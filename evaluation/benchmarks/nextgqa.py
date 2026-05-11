import json
import os
import re
import random
import string
import requests
from copy import deepcopy
from typing import Any, Dict, List, Union
import pandas as pd
from .base import BaseVideoEvalDataset, filter_metadata

def cal_ioup(gt_data, pred_data):
    pred=pred_data[0]
    pred_len = max(1e-6, pred[1] - pred[0])
    best_iou=0.0
    best_iop=0.0
    for gt in gt_data:
        start_i = max(gt[0], pred[0])
        end_i = min(gt[1], pred[1])
        inter = max(0, end_i - start_i)
        
        start_u = min(gt[0], pred[0])
        end_u = max(gt[1], pred[1])
        union = max(1e-6, end_u - start_u)
        
        best_iou = max(best_iou, inter / union)
        best_iop = max(best_iop, inter / pred_len)
    
    return best_iou, best_iop
    
    
class NEXTGQADataset(BaseVideoEvalDataset):

    BENCHMARK_TYPE: str = "gqa"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_data(self, data_root: str) -> Dict[int, Any]:
        data_dict = {}

        video_folder = os.path.join(data_root, "NExTVideo")
        mapping_file = os.path.join(data_root, "datasets/nextgqa/map_vid_vidorID.json")
        gsub_file = os.path.join(data_root, "datasets/nextgqa/gsub_test.json")
        data_file = os.path.join(data_root, "datasets/nextgqa/test.csv")
        json_list = [json.dumps(row) for row in pd.read_csv(data_file,dtype=str).to_dict('records')]
        
        with open(mapping_file, "r") as f:
            mapping = json.load(f)
        with open(gsub_file, "r") as f:
            gsub = json.load(f)
        
        json_list = [json.loads(row) for row in json_list]
        idx=0
        
        for data in json_list:
            data_dict[idx] = {
                "video_path": os.path.join(video_folder,mapping[data["video_id"]]+".mp4"),
                "start_time": None,
                "end_time": None,
                "question": data["question"],
                "ground_truth": data["answer"],
                "task_type":"qa",
                "choices": [data["a0"],data["a1"],data["a2"],data["a3"],data["a4"]]
            }
            idx+=1
            data_dict[idx] = {
                "video_path": os.path.join(video_folder,mapping[data["video_id"]]+".mp4"),
                "start_time": None,
                "end_time": None,
                "question": data["question"],
                "ground_truth": gsub[mapping[data["video_id"]][5:]]["location"][data["qid"]],
                "task_type":"temporal_grounding",
            }
            idx+=1

        return data_dict

    def generate_instruction(self, data_id: Union[int, str], video: Any) -> str:
        question = self.data_dict[data_id]["question"]
        instruction=question
        t_template = "Locate the visual content described by the query: <query>{}</query> Output the start and end timestamps in seconds."
        
        qa_template = """
            Please review the following options and select the ONE correct answer (single selection only).
            Question: {}
            
            Available Options:
            {}
            {}
            {}
            {}
            {}

            Output Requirements:
            - Directly output the CONTENT of your selected option
            - Only ONE option can be selected
            - Do not include any explanatory text

            Your choice:
        """
        if self.data_dict[data_id]["task_type"] == "temporal_grounding":
            instruction=t_template.format(question)
        else:
            instruction = qa_template.format(question,self.data_dict[data_id]["choices"][0],self.data_dict[data_id]["choices"][1],self.data_dict[data_id]["choices"][2],self.data_dict[data_id]["choices"][3],self.data_dict[data_id]["choices"][4])
        
        return instruction

    def process_response(self, data_id: Union[int, str], response: str) -> int:
        
        if self.data_dict[data_id]["task_type"] == "temporal_grounding":
            # match "from x.x to y.y" or "x.x - y.y" 
            pattern = re.compile(r'(\d+\.?\d*|\d*\.\d+)\s*(?:-|to)\s*(\d+\.?\d*|\d*\.\d+)')
            matches = pattern.findall(response)
            if len(matches) > 0:
                intervals = [[float(start), float(end)] for start, end in matches]
            else:
                pattern = r'\d*\.?\d+'
                intervals = re.findall(pattern, response)
                if len(intervals) % 2 != 0:
                    intervals.pop(0)
                intervals = [[float(intervals[i * 2]), float(intervals[i * 2 + 1])] for i in range(len(intervals) // 2)] 
            return intervals
        else:
            return response

    def evaluate(self, results):
        for data in results:
            id=data["data_id"]
            if self.data_dict[id]["task_type"] == "temporal_grounding":
                data["iou"],data["iop"]=cal_ioup(self.data_dict[data["data_id"]]["ground_truth"],data["prediction"])
            else:
                data["match"]=(self.data_dict[data["data_id"]]["ground_truth"]==data["prediction"])
                
        metrics = {}
        infos = {}
        
        temporal_res=[item for item in results if self.data_dict[item["data_id"]]["task_type"] == "temporal_grounding"]
        qa_res=[item for item in results if self.data_dict[item["data_id"]]["task_type"] == "qa"]
        
        infos["infer"] = [{k: v for k, v in item.items() if k != 'timestamps'} for item in results]
        infos["data"] = self.data_dict
        
        metrics["mIoU"] = sum([res["iou"] for res in temporal_res]) * 100 / len(temporal_res)
        metrics["mIoP"] = sum([res["iop"] for res in temporal_res]) * 100 / len(temporal_res)
        metrics["Acc"] = sum([res["match"] for res in qa_res]) * 100 / len(qa_res)
        metrics["TIoU@0.3"]=sum(1 for res in temporal_res if res["iou"] >= 0.3) * 100 / len(temporal_res)
        metrics["TIoU@0.5"]=sum(1 for res in temporal_res if res["iou"] >= 0.5) * 100 / len(temporal_res)
        
        metrics["TIoP@0.3"] = sum(1 for res in temporal_res if res["iop"] >= 0.3) * 100 / len(temporal_res)
        metrics["TIoP@0.5"] = sum(1 for res in temporal_res if res["iop"] >= 0.5) * 100 / len(temporal_res)
        
        temporal_dict = {res['data_id']: res for res in temporal_res}
        count = 0
        for qa in qa_res:
            t_id = qa['data_id'] + 1
            if t_id in temporal_dict:
                if qa['match'] and temporal_dict[t_id]['iop'] >= 0.5:
                    count += 1
        metrics["Acc@GQA"] = count * 100 / len(qa_res)

        return metrics,infos
