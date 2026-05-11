import json
import os
import re
from typing import Any, Dict, List, Union

from .base import BaseVideoEvalDataset


class ActivityNetTGDataset(BaseVideoEvalDataset):
    """
    Dataset class for ActivityNet Temporal Grounding task.
    """

    BENCHMARK_TYPE: str = "temporal_grounding"

    def load_data(self, data_root: str) -> Dict[str, Any]:
        """
        Loads the ActivityNet temporal grounding data from a JSON file and video directory.

        Args:
            data_root (str): The root directory containing 'test.json' and a 'videos' folder.

        Returns:
            Dict[str, Any]: A dictionary where keys are unique data IDs (e.g., 'video_id_sentence_index')
                            and values are dictionaries with video metadata.
        """
        data_dict = {}
        
        # Define paths for the annotation file and the video folder
        json_file_path = os.path.join(data_root, "test.json")
        video_folder_path = os.path.join(data_root, "test")

        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"Annotation file not found at: {json_file_path}")
        if not os.path.exists(video_folder_path):
            raise FileNotFoundError(f"Video folder not found at: {video_folder_path}")

        with open(json_file_path, "r") as f:
            annotations = json.load(f)

        for video_id, video_data in annotations.items():
            # Check for both .mp4 and .mkv video files
            video_path_mp4 = os.path.join(video_folder_path, f"{video_id}.mp4")
            video_path_mkv = os.path.join(video_folder_path, f"{video_id}.mkv")
            
            video_path = None
            if os.path.exists(video_path_mp4):
                video_path = video_path_mp4
            elif os.path.exists(video_path_mkv):
                video_path = video_path_mkv
            
            if video_path is None:
                # print(f"Warning: Video file for ID '{video_id}' not found. Skipping.")
                continue

            sentences = video_data.get("sentences", [])
            timestamps = video_data.get("timestamps", [])
            
            # Each sentence corresponds to a timestamp, creating a unique sample
            for i, sentence in enumerate(sentences):
                if i < len(timestamps):
                    # Create a unique ID for each video-sentence pair
                    data_id = f"{video_id}_{i}"
                    
                    data_dict[data_id] = {
                        # Required fields for data loading
                        "video_path": video_path,
                        "start_time": None,  # Process the entire video
                        "end_time": None,
                        
                        # Custom fields for instruction generation and evaluation
                        "question": sentence,
                        "ground_truth": timestamps[i],
                    }

        return data_dict

    def generate_instruction(self, data_id: Union[int, str], video_info: Any) -> str:
        """
        Generates a text prompt for the model to perform temporal grounding.

        Args:
            data_id (Union[int, str]): The unique identifier for the data sample.
            video_info (Any): Timestamps from the loaded video, not used here but required by the method signature.

        Returns:
            str: The formatted instruction string.
        """
        question = self.data_dict[data_id]["question"]
        instruction_template = "Locate the visual content described by the query: <query>{}</query> Output the start and end timestamps in seconds."
        instruction = instruction_template.format(question)
        
        return instruction

    def process_response(self, data_id: Union[int, str], response: str) -> List[List[float]]:
        """
        Parses the model's text response to extract temporal intervals.
        This implementation is robust and handles various formats, identical to the one in charades_sta.py.

        Args:
            data_id (Union[int, str]): The unique identifier for the data sample.
            response (str): The natural language response from the model.

        Returns:
            List[List[float]]: A list of predicted temporal intervals, e.g., [[start1, end1], [start2, end2]].
        """
        # Regex to find patterns like "X to Y", "X-Y", "X - Y"
        # It handles both integers and floating-point numbers.
        pattern = re.compile(r'(\d+\.?\d*|\d*\.\d+)\s*(?:-|to)\s*(\d+\.?\d*|\d*\.\d+)')
        matches = pattern.findall(response)
        
        if matches:
            # If patterns like "X to Y" are found, use them
            intervals = [[float(start), float(end)] for start, end in matches]
        else:
            # Fallback: find all numbers in the response and pair them up
            pattern = r'\d+\.?\d*'
            numbers = re.findall(pattern, response)
            # Ensure we have an even number of floats to form pairs
            if len(numbers) % 2 != 0:
                # This is a simple heuristic: if odd, remove the first number.
                # It might be part of the text but not a timestamp.
                numbers.pop(0)
            
            intervals = []
            for i in range(len(numbers) // 2):
                start = float(numbers[i * 2])
                end = float(numbers[i * 2 + 1])
                intervals.append([start, end])
                
        return intervals
