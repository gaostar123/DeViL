# evaluation/benchmarks/vidstg.py

import json
import os
import re
from typing import Any, Dict, List, Union
from .base import BaseVideoEvalDataset

class VidSTGDataset(BaseVideoEvalDataset):
    
    BENCHMARK_TYPE: str = "spatio_temporal_grounding"

    def load_data(self, data_root: str) -> Dict[str, Any]:
        data_dict = {}
        video_folder = os.path.join(data_root, "video")
        json_file = os.path.join(data_root, "annotations/test_vidstv.json")
        with open(json_file, "r") as f:
            raw_data = json.load(f)

        for data_id, ann in raw_data.items():
            video_id = ann['vid']
            video_name = f"{video_id}.mp4"
            gt_fps = ann.get('fps', 30.0)
            
            clip_start_time = ann['used_segment']['begin_fid'] / gt_fps
            clip_end_time = ann['used_segment']['end_fid'] / gt_fps
            gt_st_frame_in_clip = ann['temp_gt']['begin_fid']
            gt_ed_frame_in_clip = ann['temp_gt']['end_fid']
            gt_st_time_in_clip = gt_st_frame_in_clip / gt_fps
            gt_ed_time_in_clip = gt_ed_frame_in_clip / gt_fps
            
            ground_truth_boxes_list = [
                [b['xmin'], b['ymin'], b['xmax'], b['ymax']] if b else None 
                for b in ann['target_bboxs']
            ]

            data_dict[data_id] = {
                # required fields for data loading
                "video_path": os.path.join(video_folder, video_name),
                "start_time": clip_start_time,
                "end_time": clip_end_time,
                # required fields for evaluation
                "ground_truth_time": [gt_st_time_in_clip, gt_ed_time_in_clip],
                "ground_truth_boxes": ground_truth_boxes_list,
                "gt_start_frame": gt_st_frame_in_clip,
                "gt_bbox_format": "xyxy",
                "width": ann.get("width", 640),
                "height": ann.get("height", 360),
                "fps": gt_fps,
                "qtype": ann.get("qtype", "all"),
                # custom fields
                "question": ann['sentence']['description'],
            }
        return data_dict
    
    def generate_instruction(self, data_id: str, timestamps: List[float]) -> str:
        # This function is correct, no changes needed.
        query_text = self.data_dict[data_id]["question"]
        instruction = (
            "Locate the visual content described by the given textual query "
            f"<query>{query_text}</query> in the video. "
            "Please output the start and end timestamps in seconds and the spatial location of the object."
        )
        return instruction

    def process_response(self, data_id: str, response: str, dino_output: Dict = None, timestamps: List[float] = None) -> Dict:
        # This function is correct, no changes needed.
        # 1. Parse timestamps from text response
        pred_start_time, pred_end_time = None, None
        match = re.search(r"from (\d+\.?\d*) to (\d+\.?\d*)", response)
        if match:
            pred_start_time = float(match.group(1))
            pred_end_time = float(match.group(2))

        # 2. Process spatial output (bounding boxes)
        pred_boxes_dict = {}
        if dino_output and pred_start_time is not None and timestamps:
            pred_logits = dino_output["pred_logits"][0]
            pred_boxes_normalized = dino_output["pred_boxes"][0]
            
            meta_data = self.data_dict[data_id]
            img_w, img_h = meta_data["width"], meta_data["height"]
            
            relevant_frame_indices = [i for i, ts in enumerate(timestamps) if pred_start_time <= ts <= pred_end_time]
            
            for frame_idx in relevant_frame_indices:
                if frame_idx >= len(pred_logits): continue
                
                frame_logits = pred_logits[frame_idx]
                frame_boxes_norm = pred_boxes_normalized[frame_idx]
                pred_scores = frame_logits.sigmoid()
                max_scores_per_query, _ = pred_scores.max(-1)
                _, best_query_idx = max_scores_per_query.max(-1)
                best_box_coords_norm = frame_boxes_norm[best_query_idx].cpu().numpy()
                
                x_center, y_center, width, height = best_box_coords_norm
                w_abs = width * img_w
                h_abs = height * img_h
                x1_abs = (x_center - width / 2) * img_w
                y1_abs = (y_center - height / 2) * img_h
                
                pred_boxes_dict[frame_idx] = [x1_abs, y1_abs, w_abs, h_abs]

        return {
            "pred_time": [pred_start_time, pred_end_time] if pred_start_time is not None else None,
            "pred_boxes": pred_boxes_dict
        }
