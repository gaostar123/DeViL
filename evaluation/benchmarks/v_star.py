# evaluation/benchmarks/v_star.py

import json
import os
import re
import math
from typing import Any, Dict, List

import numpy as np

from .base import BaseVideoEvalDataset


class VStarDataset(BaseVideoEvalDataset):

    BENCHMARK_TYPE = "vstar_spatio_temporal_grounding"
    chain_mode: str = "no_chain"  # "no_chain" | "chain1" | "chain2"

    def load_data(self, data_root: str) -> Dict[str, Any]:
        data_dict = {}
        video_folder = os.path.join(data_root, "videos")
        json_file = os.path.join(data_root, "V_STaR_test.json")
        with open(json_file, "r") as f:
            raw_data = json.load(f)

        for idx, item in enumerate(raw_data):
            base_id = f"{item['vid']}_{idx}"
            video_path = os.path.join(video_folder, f"{item['vid']}.mp4")
            
            common_meta = {
                "video_path": video_path,
                "start_time": None,
                "end_time": None,
                "base_id": base_id,
                "full_data": item,
            }

            data_dict[f"{base_id}_vqa"] = {**common_meta, "task_type": "vqa"}
            data_dict[f"{base_id}_temporal"] = {**common_meta, "task_type": "temporal"}
            data_dict[f"{base_id}_spatial"] = {**common_meta, "task_type": "spatial"}

        return data_dict

    def _format_spatial_hints(self, full_data: Dict[str, Any], max_hints: int = 30) -> str:
        bboxes = full_data.get("bboxes", []) or []
        if not bboxes:
            return ""

        n = len(bboxes)
        step = max(1, math.ceil(n / max_hints))
        sampled = bboxes[::step][:max_hints]

        parts = []
        for b in sampled:
            xmin = int(b.get("xmin", 0))
            ymin = int(b.get("ymin", 0))
            xmax = int(b.get("xmax", 0))
            ymax = int(b.get("ymax", 0))
            parts.append(f"[{xmin},{ymin},{xmax},{ymax}]")
        return "; ".join(parts)

    def generate_instruction(self, data_id: str, timestamps: List[float]) -> str:
        meta = self.data_dict[data_id]
        task_type = meta["task_type"]
        full_data = meta["full_data"]

        if task_type == "vqa":
            question = full_data["question"]
            prompt = (
                "You are an expert AI assistant specializing in video analysis. "
                "Based on the provided video, give an answer of about 10 words to the following question."
                f" Question: {question}"
            )
            return prompt
        
        elif task_type == "temporal":
            query = full_data["temporal_question"]
            prompt = (
                f"Locate the visual content described by the query: <query>{query}</query> "
                f"Output the start and end timestamps in seconds."
            )
            # chain2:
            if getattr(self, "chain_mode", "no_chain") == "chain2":
                spatial_hints = self._format_spatial_hints(full_data, max_hints=50)
                if spatial_hints:
                    prompt += (
                        " Use the following ground-truth spatial hints (pixel boxes at annotated timestamps) "
                        f"to help determine the temporal window: {spatial_hints}. "
                    )
            return prompt
        
        elif task_type == "spatial":
            query = full_data["spatial_question_2"]
            prompt = (
                "Locate the visual content described by the given textual query "
                f"<query>{query}</query> in the video. "
            )
            # chain1:
            if getattr(self, "chain_mode", "no_chain") == "chain1":
                gt_ts = full_data.get("timestamps", None)
                if isinstance(gt_ts, list) and len(gt_ts) == 2:
                    t0, t1 = float(min(gt_ts)), float(max(gt_ts))
                    prompt += f" Temporal hint: the correct time window is from {t0:.2f} to {t1:.2f} seconds."
            return prompt

        else:
            raise ValueError(f"Unknown task type for V-STaR: {task_type}")

    def process_response(self, data_id: str, response: str, dino_output: Dict = None, timestamps: List[float] = None) -> Any:
        meta = self.data_dict[data_id]
        task_type = meta["task_type"]

        if task_type == "vqa":
            return response

        elif task_type == "temporal":
            pattern = re.compile(r'(\d+\.?\d*|\d*\.\d+)\s*(?:-|to)\s*(\d+\.?\d*|\d*\.\d+)')
            matches = pattern.findall(response)
            if matches:
                t1, t2 = float(matches[0][0]), float(matches[0][1])
                return [min(t1, t2), max(t1, t2)]
            return None

        elif task_type == "spatial":
            pred_boxes_dict = {}
            if dino_output and timestamps:
                full_data = meta["full_data"]
                img_w, img_h = full_data["width"], full_data["height"]
                pred_logits = dino_output["pred_logits"][0]
                pred_boxes_normalized = dino_output["pred_boxes"][0]
                
                for frame_idx in range(len(pred_logits)):
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
                "pred_time": None,
                "pred_boxes": pred_boxes_dict,
            }
