import queue
import threading
import traceback
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Union
import numpy as np
import re
import requests
import math
import ast
from openai import OpenAI
from torch.utils.data import Dataset, DataLoader

from devil.constants import DEFAULT_IMAGE_TOKEN
from devil.mm_utils import load_video_new


def qwen_api_evaluation(question: str, gt_answer: str, pred_answer: str) -> int:
    """
    Evaluates the predicted answer against the ground truth using the Qwen API (via Aliyun Dashscope).
    Returns a score from 0-3, or -1 on failure.
    """
    api_key = "112"
    if not api_key or "your_api_key" in api_key or "xxxx" in api_key:
        print("Warning: Aliyun Dashscope API key is not set. VQA scoring will be skipped, returning score 0.")
        return 0

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        system_prompt = (
            "As an AI assistant, your task is to evaluate a candidate answer in comparison to a given correct answer.\n"
            "The question itself, the correct 'groundtruth' answer, and the candidate answer will be provided to you.\n"
            "Your assessment should range from 0 to 3, based solely on the semantic similarity between the groundtruth "
            "and the candidate answer, disregarding any grammatical differences.\n"
            "A rating of 0 suggests no similarity, implying the candidate answer is entirely incorrect.\n"
            "A rating of 1 suggests low similarity, meaning the candidate answer is largely incorrect.\n"
            "A rating of 2 suggests high similarity, meaning the candidate answer is largely correct.\n"
            "Lastly, a rating of 3 indicates complete similarity, which means the candidate answer is entirely correct.\n"
            "Your response should be a single integer from 0, 1, 2, or 3."
        )
        user_prompt = f'Question: {question}\nGroundtruth answer: {gt_answer}\nCandidate answer: {pred_answer}\nYour response: '
        
        completion = client.chat.completions.create(
            model="qwen2.5-72b-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=10,
            temperature=0.0,
        )

        content = completion.choices[0].message.content
        score = int(re.search(r'\d+', content).group())
        return score
    except Exception as e:
        print(f"Error calling Qwen API (Dashscope): {e}")
        traceback.print_exc()
        return -1

def filter_metadata(data: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            if isinstance(data[key], (dict, list)):
                new_data[key] = filter_metadata(value)
            elif isinstance(data[key], (int, float, bool, str)):
                new_data[key] = value
        return new_data
    elif isinstance(data, list):
        new_data = []
        for item in data:
            if isinstance(item, (dict, list)):
                new_data.append(filter_metadata(item))
            elif isinstance(item, (int, float, bool, str)):
                new_data.append(item)
        return new_data
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    
def convert_bbox_to_xyxy(box, format, img_w, img_h):
    """Converts bounding boxes from different formats to [x1, y1, x2, y2]."""
    if box is None:
        return None
    if format == 'xywh': # for HC-STVG or model output
        x, y, w, h = box
        return [x, y, x + w, y + h]
    elif format == 'xyxy': # for VidSTG
        return box
    else:
        raise ValueError(f"Unknown box format: {format}")

def compute_iou(boxA, boxB):
    """Calculates IoU for two bounding boxes in [x1, y1, x2, y2] format."""
    if boxA is None or boxB is None:
        return 0.0

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    union_area = float(boxA_area + boxB_area - inter_area)
    
    return inter_area / union_area if union_area > 0 else 0.0

def compute_giou(boxA, boxB):
    """
    Generalized IoU for [x1,y1,x2,y2].
    """
    if boxA is None or boxB is None:
        return 0.0

    # IoU
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter_w = max(0.0, xB - xA); inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    iou = inter / union if union > 0 else 0.0

    xC1 = min(boxA[0], boxB[0]); yC1 = min(boxA[1], boxB[1])
    xC2 = max(boxA[2], boxB[2]); yC2 = max(boxA[3], boxB[3])
    areaC = max(0.0, xC2 - xC1) * max(0.0, yC2 - yC1)
    if areaC <= 0:
        return iou

    giou = iou - (areaC - union) / areaC
    return float(max(-1.0, min(1.0, giou)))


class BaseEvalDataset(Dataset, metaclass=ABCMeta):

    BENCHMARK_TYPE: str = None
    TASK_TYPES: List[str] = None
    MODAL: str = None

    def __init__(
        self,
        data_root: str,
        processor: Callable,
        num_splits: int = 1,
        split_idx: int = 0,
        fps: int = 1,
        max_frames: int = 180,
    ) -> None:
        assert split_idx < num_splits, f"split_idx ({split_idx}) should be less than num_splits ({num_splits})"
        self.processor = processor
        self.data_dict = self.load_data(data_root)


    @property
    def n_samples(self) -> int:
        return sum([len(x["data_ids"]) for x in self._aggregated_data_list])

    def __len__(self) -> int:
        return len(self._aggregated_data_list)

    @abstractmethod
    def load_data(self, data_root) -> Dict[Union[int, str], Any]:
        """
        Load the dataset meta data.

        Args:
            data_root (str): path to the dataset.

        Returns:
            data_dict (Dict[Union[int, str], Any]): dataset meta data, with data_id as key.
            example:
            {
                0: {
                    # required fields for data loading
                    "video_path": os.path.join(video_folder, data["video"]),
                    "start_time": data["start"] if task_info[3] else None,
                    "end_time": data["end"] if task_info[3] else None,
                    # required fields for evaluation
                    "task_type": task_name,
                    "ground_truth": answer_idx,
                    # custom fields for instruction generation and post processing
                    "question": data["question"],
                    "options": options,
                    "option_letters": option_letters,
                }
                ...
            }
        """
        pass

    @abstractmethod
    def generate_instruction(self, data_id: Union[int, str]) -> Union[str, Dict[str, str]]:
        """
        Generate instruction(s) for model inference.

        Args:
            data_id (Union[int, str]): identifier of the data.

        Returns:
            instruction (Union[str, Dict[str, str]]): instruction(s) for model inference.
        """
        pass

    @abstractmethod
    def process_response(self, data_id: Union[int, str], response: str) -> Any:
        """
        Process the original model responses to desired format for evaluation and visualization.

        Args:
            data_id (Union[int, str]): identifier of the data.
            response (str): model response.

        Returns:
            result (Any): processed model response for evaluation.
        """
        pass

    def evaluate(self, results: List[Dict[str, Any]]) -> (Dict[str, float], List[Dict[str, Any]]):
        """
        Compute the evaluation metrics according to predictions and ground-truths.

        Args:
            results (List[Dict[str, Any]]): list of processed model responses.

        Returns:
            metrics (Dict[str, float]): evaluation metrics.
            infos (List[Dict[str, Any]]): evaluation information for visualization.
        """
        assert self.BENCHMARK_TYPE is not None, "BENCHMARK_TYPE is not defined."
        if self.TASK_TYPES is None:
            warnings.warn("TASK_TYPES is not defined. It will be automatically inferred from metadata.")
        if self.BENCHMARK_TYPE == "mcqa":
            return self._eval_mcqa(results)
        elif self.BENCHMARK_TYPE == "oqa":
            return self._eval_oqa(results)
        elif self.BENCHMARK_TYPE == "temporal_grounding":
            return self._eval_temporal_grounding(results)
        elif self.BENCHMARK_TYPE == "spatio_temporal_grounding":
            return self._eval_stvg_corrected(results)
        elif self.BENCHMARK_TYPE == "vstar_spatio_temporal_grounding":
            return self._eval_vstar(results)
        elif self.BENCHMARK_TYPE == "spatial_grounding":
            return self._eval_spatial_grounding(results)
        elif self.BENCHMARK_TYPE == "st_align": 
            return self._eval_st_align(results)
        elif self.BENCHMARK_TYPE == "stvg_miop":
            return self._eval_miop_quintile(results)
        else:
            raise NotImplementedError(f"Unsupported benchmark type: {self.BENCHMARK_TYPE}")

    def _eval_mcqa(self, results: List[Dict[str, Any]]) -> (Dict[str, float], List[Dict[str, Any]]):
        """
        Compute the evaluation metrics for multiple-choice question answering tasks.

        Args:
            results (List[Dict[str, Any]]): list of processed model responses.

        Returns:
            metrics (Dict[str, float]): evaluation metrics.
            infos (List[Dict[str, Any]]): evaluation information for visualization.
        """
        if self.TASK_TYPES is None:
            samples = defaultdict(list)
        else:
            samples = {task_type: [] for task_type in self.TASK_TYPES}

        overall_samples = []
        infos = []

        for data in results:
            data = deepcopy(data)
            meta_data = deepcopy(self.data_dict[data["data_id"]])
            ground_truth = meta_data["ground_truth"]
            task_type = meta_data["task_type"]
            matching = data["prediction"] == meta_data["ground_truth"]

            if isinstance(task_type, (list, tuple)):
                for t in task_type:
                    samples[t].append(int(matching))
            else:
                samples[task_type].append(int(matching))

            overall_samples.append(int(matching))

            infos.append(
                {
                    **data,
                    "ground_truth": ground_truth,
                    "matching": matching,
                    "task_type": task_type,
                    "meta_data": filter_metadata(meta_data),
                }
            )

        task_types = samples.keys()
        metrics = {x: sum(samples[x]) / len(samples[x]) * 100 for x in task_types}

        # overall_samples = sum(samples.values(), [])
        overall_acc = sum(overall_samples) / len(overall_samples) * 100
        metrics["Overall"] = overall_acc

        infos = [metrics] + infos
        return metrics, infos

    def _eval_oqa(self, results: List[Dict[str, Any]]) -> (Dict[str, float], List[Dict[str, Any]]):
        """
        Compute the evaluation metrics for open-ended question answering tasks.

        Args:
            results (List[Dict[str, Any]]): list of processed model responses.

        Returns:
            metrics (Dict[str, float]): evaluation metrics.
            infos (List[Dict[str, Any]]): evaluation information for visualization.
        """
        samples = []
        infos = []

        for data in results:
            data = deepcopy(data)
            meta_data = deepcopy(self.data_dict[data["data_id"]])
            score = data["score"]

            samples.append(score)
            infos.append(
                {
                    **data,
                    "score": score,
                    "meta_data": filter_metadata(meta_data),
                }
            )

        metrics = {"Overall": sum(samples) / len(samples)}

        infos = [metrics] + infos
        return metrics, infos

    def _eval_temporal_grounding(self, results: List[Dict[str, Any]]) -> (Dict[str, float], List[Dict[str, Any]]):
        ious, infos = [], []

        for data in results:
            data = deepcopy(data)
            meta_data = deepcopy(self.data_dict[data["data_id"]])
            gt_interval = meta_data["ground_truth"]

            intersection = 0
            union = gt_interval[1] - gt_interval[0]
            for pred_interval in data["prediction"]:
                start_time, end_time = min(pred_interval), max(pred_interval)
                intersection += max(0, min(end_time, gt_interval[1]) - max(start_time, gt_interval[0]))
                union += end_time - start_time
            union = union - intersection
            iou = intersection / union

            ious.append(iou)
            infos.append(
                {
                    **data,
                    "ground_truth": gt_interval,
                    "iou": iou,
                    "meta_data": filter_metadata(meta_data),
                }
            )

        metrics = {
            "mIoU": sum(ious) / len(ious) * 100,
        }
        for thred in [0.3, 0.5, 0.7]:
            metrics[f"R1@{thred}"] = sum(iou >= thred for iou in ious) / len(ious) * 100

        infos = [metrics] + infos
        return metrics, infos
    
    def _eval_stvg_corrected(self, results: List[Dict[str, Any]]) -> (Dict[str, float], List[Dict[str, Any]]):
        """
        Compute the evaluation metrics for spatio-temporal video grounding tasks.
        This version is corrected based on user feedback to explicitly use the temporal union
        of GT and prediction to determine the set of frames for vIoU calculation.
        """
        infos = []
        
        # Group by query type (for VidSTG) or use 'all' for datasets like HC-STVG
        samples_by_qtype = defaultdict(lambda: {'tiou': [], 'viou': []})
        
        for data in results:
            data_copy = deepcopy(data)
            data_id = data_copy["data_id"]
            meta_data = deepcopy(self.data_dict[data_id])
            
            # --- Ground Truth ---
            gt_st_time, gt_ed_time = meta_data["ground_truth_time"]
            gt_boxes_list = meta_data["ground_truth_boxes"]
            gt_st_frame = meta_data["gt_start_frame"]
            gt_bbox_format = meta_data["gt_bbox_format"]
            img_w, img_h = meta_data["width"], meta_data["height"]
            qtype = meta_data.get("qtype", "all")
            gt_fps = meta_data.get('fps')
            if gt_fps is None:
                print(f"Warning: FPS not found for data_id {data_id}. Skipping.")
                continue

            # --- Prediction ---
            prediction = data_copy["prediction"]
            pred_time = prediction.get("pred_time")
            pred_boxes_dict = prediction.get("pred_boxes", {}) # {frame_idx_in_sampled_frames: [x,y,w,h]}
            # `timestamps` are the timestamps of the frames sampled by the processor
            sampled_timestamps = data_copy.get("timestamps", [])

            # --- 1. Calculate tIoU ---
            tiou = 0.0
            pred_st_time, pred_ed_time = (None, None)
            if pred_time and pred_time[0] is not None and pred_time[1] is not None:
                pred_st_time, pred_ed_time = pred_time
                inter_start = max(pred_st_time, gt_st_time)
                inter_end = min(pred_ed_time, gt_ed_time)
                intersection = max(0, inter_end - inter_start)
                if intersection > 0:
                    union = (pred_ed_time - pred_st_time) + (gt_ed_time - gt_st_time) - intersection
                    tiou = intersection / union if union > 0 else 0.0
            
            # --- 2. Calculate vIoU (New, Refined Logic) ---
            viou = 0.0
            
            # Create a lookup map for GT boxes {original_frame_number: box}
            gt_boxes_map = {i + gt_st_frame: box for i, box in enumerate(gt_boxes_list) if box is not None}
            
            # Create a lookup map for predicted boxes {original_frame_number: box}
            # This requires converting the sampled frame index to an original frame number
            pred_boxes_map = {}
            if pred_boxes_dict and sampled_timestamps:
                for sampled_idx, box in pred_boxes_dict.items():
                    if sampled_idx < len(sampled_timestamps):
                        ts = sampled_timestamps[sampled_idx]
                        # Convert timestamp to its corresponding original frame number
                        # This alignment is crucial. Assume gt_st_frame corresponds to gt_st_time.
                        time_offset = ts - gt_st_time
                        frame_offset = int(round(time_offset * gt_fps))
                        original_frame_num = gt_st_frame + frame_offset
                        pred_boxes_map[original_frame_num] = box

            # Define the set of frames that fall within the TEMPORAL UNION
            # This set will form the denominator for vIoU calculation.
            union_temporal_frames_set = set()
            if sampled_timestamps:
                for idx, ts in enumerate(sampled_timestamps):
                    # Check if the timestamp is in the GT temporal range
                    in_gt_time = (gt_st_time <= ts <= gt_ed_time)
                    # Check if the timestamp is in the predicted temporal range
                    in_pred_time = (pred_st_time is not None and pred_st_time <= ts <= pred_ed_time)
                    
                    if in_gt_time or in_pred_time:
                        time_offset = ts - gt_st_time
                        frame_offset = int(round(time_offset * gt_fps))
                        original_frame_num = gt_st_frame + frame_offset
                        union_temporal_frames_set.add(original_frame_num)
            
            # Denominator: The total number of unique frames in the temporal union
            denominator = len(union_temporal_frames_set)
            
            # Numerator: Sum of IoUs for frames that have BOTH a GT and a prediction
            # We only need to iterate through frames where a prediction exists, 
            # and check if a GT also exists for that frame.
            viou_sum = 0.0
            # Get the intersection of frames that have predictions and frames in the temporal union
            relevant_pred_frames = set(pred_boxes_map.keys()).intersection(union_temporal_frames_set)

            for frame_num in relevant_pred_frames:
                gt_box = gt_boxes_map.get(frame_num)
                if gt_box: # If a GT box also exists for this frame
                    pred_box_xywh = pred_boxes_map[frame_num]
                    pred_box_xyxy = convert_bbox_to_xyxy(pred_box_xywh, 'xywh', img_w, img_h)
                    gt_box_xyxy = convert_bbox_to_xyxy(gt_box, gt_bbox_format, img_w, img_h)
                    viou_sum += compute_iou(pred_box_xyxy, gt_box_xyxy)
            
            if denominator > 0:
                viou = viou_sum / denominator

            samples_by_qtype[qtype]['tiou'].append(tiou)
            samples_by_qtype[qtype]['viou'].append(viou)

            infos.append({
                "data_id": data_id,
                "prediction": prediction,
                "response": data_copy["response"],
                "ground_truth_time": [gt_st_time, gt_ed_time],
                "ground_truth_boxes_count": len([b for b in gt_boxes_list if b]),
                "tiou": tiou,
                "viou": viou,
                "qtype": qtype,
                "meta_data": filter_metadata(meta_data),
            })
        
        # --- 3. Calculate and aggregate final metrics ---
        # (This part remains unchanged)
        metrics = {}
        overall_tiou = []
        overall_viou = []

        for qtype, data in samples_by_qtype.items():
            if not data['tiou']: continue
            viou_arr = np.array(data['viou'])
            
            prefix = "" if len(samples_by_qtype) == 1 and qtype == "all" else f"{qtype}_"
            
            metrics[f'{prefix}mTIoU'] = np.mean(data['tiou'])
            metrics[f'{prefix}mVIoU'] = np.mean(viou_arr)
            metrics[f'{prefix}vIoU@0.3'] = np.mean(viou_arr > 0.3)
            metrics[f'{prefix}vIoU@0.5'] = np.mean(viou_arr > 0.5)

            overall_tiou.extend(data['tiou'])
            overall_viou.extend(data['viou'])
            
        if len(samples_by_qtype) > 1:
            viou_arr_overall = np.array(overall_viou)
            metrics['Overall_mTIoU'] = np.mean(overall_tiou)
            metrics['Overall_mVIoU'] = np.mean(viou_arr_overall)
            metrics['Overall_vIoU@0.3'] = np.mean(viou_arr_overall > 0.3)
            metrics['Overall_vIoU@0.5'] = np.mean(viou_arr_overall > 0.5)
        
        infos = [metrics] + infos
        return metrics, infos
    
    
    def _eval_vstar(self, results: List[Dict[str, Any]]) -> (Dict[str, float], List[Dict[str, Any]]):
        
        # --- Helper functions for evaluation, ported from original v_star.py for consistency ---
        def calculate_temporal_iou(gt_range, pred_range):
            if not pred_range:
                return 0.0
            
            if isinstance(pred_range, str):
                try:
                    pred_range = ast.literal_eval(pred_range)
                except (ValueError, SyntaxError):
                    return 0.0

            if not isinstance(pred_range, (list, tuple)) or len(pred_range) != 2 or \
            not all(isinstance(x, (int, float)) for x in pred_range):
                return 0.0

            gt_start, gt_end = gt_range
            pred_start, pred_end = min(pred_range), max(pred_range) # Ensure order
            intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
            union = (gt_end - gt_start) + (pred_end - pred_start) - intersection
            return intersection / union if union > 0 else 0.0

        def _compute_spatial_iou(gt_bbox_dict, pred_bbox_xywh, img_w, img_h):
            # V-STaR GT bboxes are in xyxy format
            gt_box_xyxy = [gt_bbox_dict['xmin'], gt_bbox_dict['ymin'], gt_bbox_dict['xmax'], gt_bbox_dict['ymax']]
            # Convert prediction from xywh to xyxy
            # Assuming convert_bbox_to_xyxy and compute_iou are defined elsewhere
            pred_box_xyxy = convert_bbox_to_xyxy(pred_bbox_xywh, 'xywh', img_w, img_h)
            return compute_iou(gt_box_xyxy, pred_box_xyxy)

        def calculate_spatial_metrics(gt_bboxes, pred_bboxes_interpolated, img_w, img_h):
            if not pred_bboxes_interpolated:
                return [0.0] * 5, 0.0

            iou_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
            ious = []
            for gt_bbox_data in gt_bboxes:
                # V-STaR GT bboxes use timestamp as key
                frame_id = float(gt_bbox_data["timestamp"])
                frame_id_str = str(frame_id)
                if frame_id_str in pred_bboxes_interpolated:
                    pred_bbox_xywh = pred_bboxes_interpolated[frame_id_str]
                    ious.append(_compute_spatial_iou(gt_bbox_data, pred_bbox_xywh, img_w, img_h))
                else:
                    ious.append(0.0)
            
            mIoU = np.mean(ious) if ious else 0.0
            aps = [np.mean([1 if iou >= th else 0 for iou in ious]) for th in iou_thresholds]
            return aps, mIoU
            
        def _interpolate_boxes(pred_boxes_dict, pred_timestamps, gt_bboxes_data):
            interpolated_boxes = {}
            if not pred_boxes_dict or not pred_timestamps:
                return {}
                
            pred_points = sorted(
                [(pred_timestamps[idx], box) for idx, box in pred_boxes_dict.items() if idx < len(pred_timestamps)]
            )

            if not pred_points:
                return {}

            for gt_bbox in gt_bboxes_data:
                target_ts = float(gt_bbox["timestamp"])
                before = [p for p in pred_points if p[0] <= target_ts]
                after = [p for p in pred_points if p[0] >= target_ts]
                
                p1, p2 = None, None
                if before:
                    p1 = max(before, key=lambda x: x[0])
                if after:
                    p2 = min(after, key=lambda x: x[0])

                if p1 and p2 and p1[0] == p2[0]:
                    interpolated_box = p1[1]
                elif p1 and p2:
                    t1, b1 = p1
                    t2, b2 = p2
                    if t2 == t1:
                        ratio = 0.0
                    else:
                        ratio = (target_ts - t1) / (t2 - t1)
                    
                    b1 = np.array(b1)
                    b2 = np.array(b2)
                    interpolated_box = (b1 + ratio * (b2 - b1)).tolist()

                elif p1:
                    interpolated_box = p1[1]
                elif p2:
                    interpolated_box = p2[1]
                else:
                    continue
                
                interpolated_boxes[str(target_ts)] = interpolated_box
            
            return interpolated_boxes

        # --- Main Evaluation Logic ---
        # 1. Group results by base_id
        grouped_results = defaultdict(dict)
        for r in results:
            meta = self.data_dict[r["data_id"]]
            base_id = meta["base_id"]
            task_type = meta["task_type"]
            grouped_results[base_id][task_type] = r
            grouped_results[base_id]['meta'] = meta['full_data']
            if 'timestamps' in r:
                grouped_results[base_id]['pred_timestamps'] = r['timestamps']
        
        # 2. Iterate through grouped results and calculate metrics
        all_vqa_scores, all_temporal_ious, all_spatial_mious = [], [], []
        all_spatial_aps = [[] for _ in range(5)] # For 5 thresholds
        infos = []

        for base_id, item_results in grouped_results.items():
            meta = item_results['meta']
            
            # VQA Score
            pred_vqa = item_results.get('vqa', {}).get('prediction', "")
            # Assuming qwen_api_evaluation is defined elsewhere
            vqa_score = qwen_api_evaluation(meta['question'], meta['answer'], pred_vqa)
            all_vqa_scores.append(vqa_score if vqa_score != -1 else 0)
            
            # Temporal IoU
            pred_temporal = item_results.get('temporal', {}).get('prediction')
            temporal_iou = calculate_temporal_iou(meta['timestamps'], pred_temporal)
            all_temporal_ious.append(temporal_iou)
            
            # Spatial Metrics
            pred_spatial = item_results.get('spatial', {}).get('prediction', {})
            pred_boxes_dict = pred_spatial.get("pred_boxes", {})
            pred_timestamps = item_results.get('pred_timestamps', [])
            
            interpolated_boxes = _interpolate_boxes(pred_boxes_dict, pred_timestamps, meta['bboxes'])
            
            img_w, img_h = meta.get("width"), meta.get("height")
            aps, mIoU = calculate_spatial_metrics(meta['bboxes'], interpolated_boxes, img_w, img_h)
            
            for i in range(5):
                all_spatial_aps[i].append(aps[i])
            all_spatial_mious.append(mIoU)

            infos.append({
                "base_id": base_id,
                "vqa_score": vqa_score,
                "temporal_iou": temporal_iou,
                "spatial_mIoU": mIoU,
                "spatial_aps": aps,
                "vqa_pred": pred_vqa,
                "temporal_pred": pred_temporal,
                "spatial_pred_interpolated": interpolated_boxes,
            })

        # 3. Aggregate final metrics
        metrics = {}
        
        # --- VQA Metrics ---
        valid_vqa_scores = [s for s in all_vqa_scores if s != -1]
        metrics["VQA Avg Score"] = np.mean(valid_vqa_scores) if valid_vqa_scores else 0.0
        # <<< FIX 1: Correctly calculate the percentage/accuracy >>>
        # The original code `np.mean([1 for s ... if ...])` was incorrect.
        acc_vqa = np.mean([1 if s >= 2 else 0 for s in valid_vqa_scores]) if valid_vqa_scores else 0.0
        metrics["VQA Accuracy"] = acc_vqa
        
        # --- Temporal Metrics ---
        mean_temporal_iou = np.mean(all_temporal_ious)
        metrics["Temporal Mean IoU"] = mean_temporal_iou
        # <<< FIX 2: Correctly calculate the recall at different IoU thresholds >>>
        # The original logic was flawed, similar to the VQA accuracy calculation.
        metrics["Temporal R1@IoU=0.3"] = np.mean([1 if iou >= 0.3 else 0 for iou in all_temporal_ious])
        metrics["Temporal R1@IoU=0.5"] = np.mean([1 if iou >= 0.5 else 0 for iou in all_temporal_ious])
        metrics["Temporal R1@IoU=0.7"] = np.mean([1 if iou >= 0.7 else 0 for iou in all_temporal_ious])
        
        # --- Spatial Metrics ---
        mean_spatial_miou = np.mean(all_spatial_mious)
        metrics["Spatial Mean mIoU"] = mean_spatial_miou
        metrics["Spatial mAP@0.1"] = np.mean(all_spatial_aps[0])
        metrics["Spatial mAP@0.3"] = np.mean(all_spatial_aps[1])
        metrics["Spatial mAP@0.5"] = np.mean(all_spatial_aps[2])
        metrics["Spatial mAP@0.7"] = np.mean(all_spatial_aps[3])
        metrics["Spatial mAP@0.9"] = np.mean(all_spatial_aps[4])
        
        # --- Combined Metrics ---
        # Note: all_vqa_scores already treats API errors (-1) as 0, so we can use it directly.
        correct_vqa = [s >= 2 for s in all_vqa_scores]
        correct_temp = [iou >= 0.3 for iou in all_temporal_ious]
        correct_spat = [miou >= 0.1 for miou in all_spatial_mious]
        
        metrics["VQA & Temp (R@0.3)"] = np.mean([v and t for v, t in zip(correct_vqa, correct_temp)])
        metrics["VQA & Spat (mIoU@0.1)"] = np.mean([v and s for v, s in zip(correct_vqa, correct_spat)])
        metrics["Temp & Spat"] = np.mean([t and s for t, s in zip(correct_temp, correct_spat)])
        metrics["VQA & Temp & Spat"] = np.mean([v and t and s for v, t, s in zip(correct_vqa, correct_temp, correct_spat)])

        # --- AM and LGM Metrics ---
        epsilon = 1e-9
        safe_acc_vqa = np.clip(acc_vqa, 0, 1 - epsilon)
        safe_mean_temporal_iou = np.clip(mean_temporal_iou, 0, 1 - epsilon)
        safe_mean_miou = np.clip(mean_spatial_miou, 0, 1 - epsilon)

        metrics["AM"] = (safe_acc_vqa + safe_mean_temporal_iou + safe_mean_miou) / 3
        
        log_sum = -(
            math.log(1 - safe_acc_vqa) +
            math.log(1 - safe_mean_temporal_iou) +
            math.log(1 - safe_mean_miou)
        )
        metrics["LGM"] = log_sum / 3

        infos = [metrics] + infos
        return metrics, infos
    
    def _eval_spatial_grounding(self, results: List[Dict[str, Any]]) -> (Dict[str, float], List[Dict[str, Any]]):
        """
        Compute evaluation metrics for spatial grounding (visual grounding) tasks.
        """
        samples_by_task = defaultdict(list)
        infos = []

        for data in results:
            data_copy = deepcopy(data)
            data_id = data_copy["data_id"]
            meta_data = deepcopy(self.data_dict[data_id])
            
            gt_box_xywh = meta_data["ground_truth"]
            img_w, img_h = meta_data["width"], meta_data["height"]
            task_type = meta_data["task_type"] # e.g., 'testA', 'testB', 'val'
            
            pred_box_xywh = data_copy["prediction"]

            gt_box_xyxy = convert_bbox_to_xyxy(gt_box_xywh, 'xywh', img_w, img_h)
            pred_box_xyxy = convert_bbox_to_xyxy(pred_box_xywh, 'xywh', img_w, img_h)
            
            iou = compute_iou(gt_box_xyxy, pred_box_xyxy)
            
            samples_by_task[task_type].append(iou)

            infos.append({
                **data_copy,
                "ground_truth": gt_box_xywh,
                "iou": iou,
                "task_type": task_type,
                "meta_data": filter_metadata(meta_data),
            })
            
        metrics = {}
        all_ious = []
        for task, ious in samples_by_task.items():
            if not ious: continue
            metrics[f"{task}_mIoU"] = np.mean(ious) * 100
            metrics[f"{task}_Acc@0.5"] = np.mean([1 if iou >= 0.5 else 0 for iou in ious]) * 100
            all_ious.extend(ious)
        
        if len(samples_by_task) > 1:
            metrics["Overall_mIoU"] = np.mean(all_ious) * 100
            metrics["Overall_Acc@0.5"] = np.mean([1 if iou >= 0.5 else 0 for iou in all_ious]) * 100

        infos = [metrics] + infos
        return metrics, infos
    
    def _eval_st_align(self, results: List[Dict[str, Any]]) -> (Dict[str, float], List[Dict[str, Any]]):
        """
        Computes evaluation metrics for ST-Align (STVG & SVG) tasks.
        """
        infos = []
        samples_by_task = defaultdict(list)

        # 1. Group results by task type
        for data in results:
            data_copy = deepcopy(data)
            data_id = data_copy["data_id"]
            meta_data = deepcopy(self.data_dict[data_id])
            task_type = meta_data["task_type"]
            samples_by_task[task_type].append(data_copy)

        metrics = {}

        # 2. Evaluate STVG task
        if 'stvg' in samples_by_task:
            tious, sious, stious = [], [], []
            for data in samples_by_task['stvg']:
                data_id = data["data_id"]
                meta_data = self.data_dict[data_id]
                
                # GT data
                gt_st_time, gt_ed_time = meta_data["ground_truth_time"]
                gt_boxes_list = meta_data["ground_truth_boxes"]
                gt_st_frame = meta_data["gt_start_frame"]
                gt_bbox_format = meta_data["gt_bbox_format"]
                img_w, img_h = meta_data["width"], meta_data["height"]
                gt_fps = meta_data.get('fps')
                
                # Prediction data
                prediction = data["prediction"]
                pred_time = prediction.get("pred_time")
                pred_boxes_dict = prediction.get("pred_boxes", {}) # {sampled_frame_idx: box}
                sampled_timestamps = data.get("timestamps", [])

                # --- tIoU Calculation ---
                tiou = 0.0
                pred_st_time, pred_ed_time = (None, None)
                if pred_time:
                    pred_st_time, pred_ed_time = pred_time
                    inter_start = max(pred_st_time, gt_st_time)
                    inter_end = min(pred_ed_time, gt_ed_time)
                    intersection = max(0, inter_end - inter_start)
                    if intersection > 0:
                        union = (pred_ed_time - pred_st_time) + (gt_ed_time - gt_st_time) - intersection
                        tiou = intersection / union if union > 0 else 0.0
                tious.append(tiou)
                
                # --- vIoU/sTIoU and sIoU Calculation ---
                gt_boxes_map = {i + gt_st_frame: box for i, box in enumerate(gt_boxes_list)}
                pred_boxes_map = {} # {original_frame_num: box}
                if pred_boxes_dict and sampled_timestamps and gt_fps:
                    for sampled_idx, box in pred_boxes_dict.items():
                        if sampled_idx < len(sampled_timestamps):
                            ts = sampled_timestamps[sampled_idx]
                            original_frame_num = int(round(ts * gt_fps))
                            pred_boxes_map[original_frame_num] = box

                union_frames = set(gt_boxes_map.keys()) | set(pred_boxes_map.keys())
                inter_frames = set(gt_boxes_map.keys()) & set(pred_boxes_map.keys())
                
                iou_sum_inter = 0.0
                for frame_num in inter_frames:
                    pred_box_xywh = pred_boxes_map[frame_num]
                    gt_box = gt_boxes_map[frame_num]
                    pred_box_xyxy = convert_bbox_to_xyxy(pred_box_xywh, 'xywh', img_w, img_h)
                    gt_box_xyxy = convert_bbox_to_xyxy(gt_box, gt_bbox_format, img_w, img_h)
                    iou_sum_inter += compute_iou(pred_box_xyxy, gt_box_xyxy)
                
                siou = iou_sum_inter / len(inter_frames) if inter_frames else 0.0
                stiou = iou_sum_inter / len(union_frames) if union_frames else 0.0
                sious.append(siou)
                stious.append(stiou)

                infos.append({**data, "tiou": tiou, "siou": siou, "stiou": stiou, "meta_data": filter_metadata(meta_data)})

            # Aggregate STVG metrics
            stvg_metrics = {
                "stvg_mtiou": np.mean(tious),
                "stvg_tiou@0.3": np.mean([1 if i >= 0.3 else 0 for i in tious]),
                "stvg_tiou@0.5": np.mean([1 if i >= 0.5 else 0 for i in tious]),
                "stvg_tiou@0.7": np.mean([1 if i >= 0.7 else 0 for i in tious]),
                "stvg_msiou": np.mean(sious),
                "stvg_siou@0.3": np.mean([1 if i >= 0.3 else 0 for i in sious]),
                "stvg_siou@0.5": np.mean([1 if i >= 0.5 else 0 for i in sious]),
                "stvg_siou@0.7": np.mean([1 if i >= 0.7 else 0 for i in sious]),
                "stvg_mstiou": np.mean(stious),
                "stvg_viou@0.3": np.mean([1 if i >= 0.3 else 0 for i in stious]),
                "stvg_viou@0.5": np.mean([1 if i >= 0.5 else 0 for i in stious]),
                "stvg_viou@0.7": np.mean([1 if i >= 0.7 else 0 for i in stious]),
            }
            metrics.update(stvg_metrics)

        # 3. Evaluate SVG task
        if 'svg' in samples_by_task:
            sious, stious = [], []
            for data in samples_by_task['svg']:
                data_id = data["data_id"]
                meta_data = self.data_dict[data_id]

                # GT data
                gt_boxes_list = meta_data["ground_truth_boxes"]
                gt_st_frame = meta_data["gt_start_frame"]
                gt_bbox_format = meta_data["gt_bbox_format"]
                img_w, img_h = meta_data["width"], meta_data["height"]
                gt_fps = meta_data.get('fps')
                
                # Prediction data
                prediction = data["prediction"]
                pred_boxes_dict = prediction.get("pred_boxes", {}) # {sampled_frame_idx: box}
                sampled_timestamps = data.get("timestamps", [])

                # Map GT and Pred boxes to original frame numbers
                gt_boxes_map = {i + gt_st_frame: box for i, box in enumerate(gt_boxes_list)}
                pred_boxes_map = {} # {original_frame_num: box}
                if pred_boxes_dict and sampled_timestamps and gt_fps:
                    for sampled_idx, box in pred_boxes_dict.items():
                        if sampled_idx < len(sampled_timestamps):
                            ts = sampled_timestamps[sampled_idx]
                            original_frame_num = int(round(ts * gt_fps))
                            # For SVG, we only care about predictions within the GT time window
                            if gt_st_frame <= original_frame_num < (gt_st_frame + len(gt_boxes_list)):
                                pred_boxes_map[original_frame_num] = box
                
                union_frames = set(gt_boxes_map.keys()) | set(pred_boxes_map.keys())
                inter_frames = set(gt_boxes_map.keys()) & set(pred_boxes_map.keys())
                
                iou_sum_inter = 0.0
                for frame_num in inter_frames:
                    pred_box_xywh = pred_boxes_map[frame_num]
                    gt_box = gt_boxes_map[frame_num]
                    pred_box_xyxy = convert_bbox_to_xyxy(pred_box_xywh, 'xywh', img_w, img_h)
                    gt_box_xyxy = convert_bbox_to_xyxy(gt_box, gt_bbox_format, img_w, img_h)
                    iou_sum_inter += compute_iou(pred_box_xyxy, gt_box_xyxy)
                
                siou = iou_sum_inter / len(inter_frames) if inter_frames else 0.0
                stiou = iou_sum_inter / len(union_frames) if union_frames else 0.0
                sious.append(siou)
                stious.append(stiou)
                
                infos.append({**data, "siou": siou, "stiou": stiou, "meta_data": filter_metadata(meta_data)})

            # Aggregate SVG metrics
            svg_metrics = {
                "svg_msiou": np.mean(sious),
                "svg_siou@0.3": np.mean([1 if i >= 0.3 else 0 for i in sious]),
                "svg_siou@0.5": np.mean([1 if i >= 0.5 else 0 for i in sious]),
                "svg_siou@0.7": np.mean([1 if i >= 0.7 else 0 for i in sious]),
                "svg_mstiou": np.mean(stious),
            }
            metrics.update(svg_metrics)

        infos = [metrics] + infos
        return metrics, infos

    def _eval_miop_quintile(self, results: List[Dict[str, Any]]) -> (Dict[str, float], List[Dict[str, Any]]):
        import numpy as np
        from collections import defaultdict

        def avg(xs):
            return float(np.mean(xs)) if xs else 0.0

        infos = []
        dataset_bins = {i: [] for i in range(5)}
        dataset_overall = []

        for data in results:
            data_id = data["data_id"]
            meta = self.data_dict[data_id]

            gt_st_time, gt_ed_time = meta["ground_truth_time"]
            if gt_ed_time <= gt_st_time:
                continue

            gt_boxes_list = meta["ground_truth_boxes"]
            gt_st_frame = meta["gt_start_frame"]
            gt_fmt = meta["gt_bbox_format"]                # "xyxy"
            img_w, img_h = meta["width"], meta["height"]
            fps = meta.get("fps", 30.0)

            timestamps = data.get("timestamps", [])
            pred_boxes_dict = (data.get("prediction") or {}).get("pred_boxes", {})  # {sample_idx: [x,y,w,h]}

            gt_map = {}
            for i, box in enumerate(gt_boxes_list):
                if box is None:
                    continue
                fnum = gt_st_frame + i
                gt_map[fnum] = convert_bbox_to_xyxy(box, gt_fmt, img_w, img_h)

            pred_map = {}
            if pred_boxes_dict and timestamps:
                for sidx, box_xywh in pred_boxes_dict.items():
                    if sidx >= len(timestamps):
                        continue
                    ts = timestamps[sidx]
                    if not (gt_st_time <= ts <= gt_ed_time):
                        continue
                    frame_offset = int(round((ts - gt_st_time) * fps))
                    fnum = gt_st_frame + frame_offset
                    pred_map[fnum] = (convert_bbox_to_xyxy(box_xywh, 'xywh', img_w, img_h), ts)

            dt = (gt_ed_time - gt_st_time) / 5.0
            sample_bin_ious = {i: [] for i in range(5)}
            sample_all_ious = []

            for fnum, (pred_xyxy, ts) in pred_map.items():
                gt_xyxy = gt_map.get(fnum, None)
                if gt_xyxy is None:
                    continue
                iou = compute_iou(gt_xyxy, pred_xyxy)
                sample_all_ious.append(iou)

                if dt > 0:
                    rel = (ts - gt_st_time) / dt
                    bin_idx = int(rel)
                    if bin_idx < 0: bin_idx = 0
                    if bin_idx > 4: bin_idx = 4
                else:
                    bin_idx = 0

                sample_bin_ious[bin_idx].append(iou)

            sample_bin_means = {}
            for i in range(5):
                if sample_bin_ious[i]:
                    m = float(np.mean(sample_bin_ious[i]))
                    sample_bin_means[i] = m
                    dataset_bins[i].append(m)

            if sample_all_ious:
                sample_overall = float(np.mean(sample_all_ious))
                dataset_overall.append(sample_overall)
            else:
                sample_overall = None

            infos.append({
                "data_id": data_id,
                "Q1_miOP": sample_bin_means.get(0, None),
                "Q2_miOP": sample_bin_means.get(1, None),
                "Q3_miOP": sample_bin_means.get(2, None),
                "Q4_miOP": sample_bin_means.get(3, None),
                "Q5_miOP": sample_bin_means.get(4, None),
                "Overall_miOP": sample_overall,
                "meta_data": filter_metadata(meta),
            })

        metrics = {
            "Q1_miOP": avg(dataset_bins[0]),
            "Q2_miOP": avg(dataset_bins[1]),
            "Q3_miOP": avg(dataset_bins[2]),
            "Q4_miOP": avg(dataset_bins[3]),
            "Q5_miOP": avg(dataset_bins[4]),
            "Overall_miOP": avg(dataset_overall),
        }

        infos = [metrics] + infos
        return metrics, infos





class BaseImageEvalDataset(BaseEvalDataset):
    
    MODAL: str = "image"
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        aggregated_data = dict()
        for data_id, meta_data in self.data_dict.items():
            image_path = meta_data["image_path"]
            if image_path not in aggregated_data:
                aggregated_data[image_path] = {
                    "image_path": image_path,
                    "data_ids": [data_id],
                }
            else:
                aggregated_data[image_path]["data_ids"].append(data_id)

        aggregated_data_list = [x for _, x in aggregated_data.items()]
        self._aggregated_data_list = aggregated_data_list[kwargs["split_idx"]::kwargs["num_splits"]]
        
    # REPLACEMENT for BaseImageEvalDataset.__getitem__ method
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        aggregated_data = self._aggregated_data_list[idx]
        
        try:
            images = self.processor.load_images(aggregated_data["image_path"])
            image_inputs = self.processor.process_images(images, merge_size=1, return_tensors="pt")
        except Exception:
            traceback.print_exc()
            print(f"Failed to load image: {aggregated_data}")
            return {
                "data_ids": [], "image_inputs": {}, "text_inputs": [], "raw_images": []
            }
            
        text_inputs = []
        for data_id in aggregated_data["data_ids"]:
            instruction = self.generate_instruction(data_id)
            content = [{"type": "image"}]
            conversation = [
                {
                    "role": "user",
                    "content": content + [{"type": "text", "text": instruction}],
                }
            ]
            prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            text_inputs.append(
                self.processor.process_text(
                    prompt, 
                    image_inputs, 
                    padding=False,
                    padding_side=None,
                    return_tensors="pt"
                )
            )

        data = {
            "data_ids": aggregated_data["data_ids"],
            "image_inputs": image_inputs,
            "text_inputs": text_inputs,
            "raw_images": [images[0]], # Pass the raw PIL image, assuming one image per sample
        }

        return data
        
        
class BaseVideoEvalDataset(BaseEvalDataset):
    
    MODAL: str = "video"
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fps = kwargs["fps"]
        self.max_frames = kwargs["max_frames"]

        aggregated_data = dict()
        for data_id, meta_data in self.data_dict.items():
            video_path = meta_data["video_path"]
            start_time = meta_data["start_time"]
            end_time = meta_data["end_time"]
            aggregated_data_id = f"{video_path}_{start_time}_{end_time}"
            if aggregated_data_id not in aggregated_data:
                aggregated_data[aggregated_data_id] = {
                    "video_path": video_path,
                    "start_time": start_time,
                    "end_time": end_time,
                    "data_ids": [data_id],
                }
            else:
                aggregated_data[aggregated_data_id]["data_ids"].append(data_id)

        aggregated_data_list = [x for _, x in aggregated_data.items()]
        self._aggregated_data_list = aggregated_data_list[kwargs["split_idx"]::kwargs["num_splits"]]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        aggregated_data = self._aggregated_data_list[idx]

        try:
            frames, timestamps, imgs = self.processor.load_video( # Note: variable name is 'imgs' here
                aggregated_data["video_path"],
                start_time=aggregated_data["start_time"],
                end_time=aggregated_data["end_time"],
                make_pil=True,
            )
            image_inputs = self.processor.process_images(
                [frames],
                merge_size=2,
                return_tensors="pt"
            )
        except:
            traceback.print_exc()
            print(f"Failed to load video: {aggregated_data}")
            return {
                "data_ids": [], "image_inputs": {}, "text_inputs": [], "timestamps": [], "raw_images": []
            }

        text_inputs = []
        for data_id in aggregated_data["data_ids"]:
            instruction = self.generate_instruction(data_id, timestamps)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "num_frames": len(timestamps),
                            "timestamps": timestamps,
                        },
                        {"type": "text", "text": instruction},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            text_inputs.append(
                self.processor.process_text(
                    prompt,
                    image_inputs,
                    padding=False,
                    padding_side=None,
                    return_tensors="pt"
                )
            )

        data = {
            "data_ids": aggregated_data["data_ids"],
            "image_inputs": image_inputs,
            "text_inputs": text_inputs,
            "timestamps": timestamps,
            "raw_images": imgs,  # <-- ADDED: Pass the raw PIL images from the video
        }

        return data
