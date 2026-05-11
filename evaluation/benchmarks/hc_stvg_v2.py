import json
import os
import re
from typing import Any, Dict, List, Union
import torch
from .base import BaseVideoEvalDataset


class HCSTVGv2Dataset(BaseVideoEvalDataset):
    
    BENCHMARK_TYPE: str = "spatio_temporal_grounding"

    def load_data(self, data_root: str) -> Dict[str, Any]:
        data_dict = {}
        video_folder = os.path.join(data_root, "videos")
        json_file = os.path.join(data_root, "anno_v2/val_hc_stvg_v2.json") 
        with open(json_file, "r") as f:
            raw_data = json.load(f)

        for video_name, ann in raw_data.items():
            data_dict[video_name] = {
                "video_path": os.path.join(video_folder, video_name),
                "start_time": None,
                "end_time": None,
                "ground_truth_time": [ann["st_time"], ann["ed_time"]],
                "ground_truth_boxes": ann["bbox"],
                "gt_start_frame": ann["st_frame"],
                "gt_bbox_format": "xywh",
                "width": ann.get("img_size", [360, 640])[1],
                "height": ann.get("img_size", [360, 640])[0],
                "fps": ann['img_num'] / 20.0,
                "question": ann.get("caption") or ann.get("English"),
            }
        return data_dict
    
    def generate_instruction(self, data_id: str, timestamps: List[float]) -> str:
        query_text = self.data_dict[data_id]["question"]
        instruction = (
            "Locate the visual content described by the given textual query "
            f"<query>{query_text}</query> in the video. "
            "Please output the start and end timestamps in seconds and the spatial location of the object."
        )
        return instruction

    def process_response(self, data_id: str, response: str, dino_output: Dict = None, timestamps: List[float] = None) -> Dict:
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
            relevant_frame_indices = [i for i in relevant_frame_indices if i < len(pred_logits)]

            if relevant_frame_indices:

                per_frame_query_scores = []
                for frame_idx in relevant_frame_indices:
                    frame_logits = pred_logits[frame_idx]              
                    frame_scores = frame_logits.sigmoid().max(dim=-1).values  
                    per_frame_query_scores.append(frame_scores)

                scores_tq = torch.stack(per_frame_query_scores, dim=0)  
                tube_scores = scores_tq.mean(dim=0)                     
                best_query_idx = int(tube_scores.argmax().item())     

                for frame_idx in relevant_frame_indices:
                    frame_boxes_norm = pred_boxes_normalized[frame_idx]
                    best_box_coords_norm = frame_boxes_norm[best_query_idx].detach().cpu().numpy()

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
