import argparse
import os
import os.path as osp
import random
import traceback
from typing import Any, Dict, List, Union
import time
import datetime
import json
import numpy as np
import torch
import torch.distributed as dist
from prettytable import PrettyTable
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import sys
sys.path.append(".")
from devil import disable_torch_init, model_init, mm_infer
from evaluation.benchmarks import build_dataset
from evaluation.register import INFERENCES
from evaluation.utils import CUDADataLoader
all_infer_time=[]

def _to_jsonable(obj):
    
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "--model_path", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)

    parser.add_argument("--data-root", "--data_root", type=str, required=True) 
    parser.add_argument("--num-workers", "--num_workers", type=int, default=8)

    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max-frames", "--max_frames", type=int, default=180)
    parser.add_argument("--max-visual-tokens", "--max_visual_tokens", type=int, default=None)

    parser.add_argument("--save-path", "--save_path", type=str, default=None)

    return parser.parse_args()


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def show_metrics(metrics: Dict[str, Any], benchmark: str):
    """
    Displays evaluation metrics in a formatted table, with all float values
    rounded to four decimal places.
    """
    if all(isinstance(metric, dict) for metric in metrics.values()):
        for task_name, metric in metrics.items():
            show_metrics(metric, f"{benchmark}_{task_name}")
    elif all(isinstance(metric, (int, float, np.integer, np.floating)) for metric in metrics.values()):
        table = PrettyTable(["Metric", "Value"])
        for metric_name, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            # Format all float values to 4 decimal places
            formatted_value = f"{value:.4f}"
            table.add_row([metric_name, formatted_value])
        table.align["Metric"] = "l"
        print(f"Results on {benchmark}:")
        print(table)
        print("\n")
    else:
        # Allow mixed types, but only format floats
        table = PrettyTable(["Metric", "Value"])
        all_basic_types = True
        for metric_name, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
                table.add_row([metric_name, formatted_value])
            elif isinstance(value, (int, str, bool)):
                 table.add_row([metric_name, value])
            else:
                all_basic_types = False
                break # Break if we find a complex type like a dict

        if all_basic_types:
            table.align["Metric"] = "l"
            print(f"Results on {benchmark}:")
            print(table)
            print("\n")
        else:
            # Fallback for complex/unexpected structures
            print(f"Complex metrics structure for {benchmark}:")
            print(json.dumps(_to_jsonable(metrics), indent=2))


def main():
    dist.init_process_group(backend="gloo", timeout=datetime.timedelta(minutes=120))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()

    seed_everything()
    args = parse_args()

    disable_torch_init()
    model_init, mm_infer_fn = INFERENCES(args.model_path)
    model, processor = model_init(
        args.model_path,
        args.max_visual_tokens,
        device_map={"": f"cuda:{local_rank}"}
    )

    dataset = build_dataset(
        args.benchmark,
        data_root=args.data_root,
        processor=processor,
        num_splits=dist.get_world_size(),
        split_idx=global_rank,
        fps=args.fps,
        max_frames=args.max_frames,
    )

    is_vstar = getattr(dataset, "BENCHMARK_TYPE", "") == "vstar_spatio_temporal_grounding"
    run_modes = ["chain1", "chain2"] if is_vstar else ["no_chain"]

    all_metrics: Dict[str, Dict[str, Any]] = {}
    merged_by_mode: Dict[str, List[Dict[str, Any]]] = {}
    base_save = args.save_path
    base_name, base_ext = (os.path.splitext(base_save) if base_save else (None, None))

    def index_results(merged_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        idx: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for r in merged_list:
            data_id = r["data_id"]
            meta = dataset.data_dict[data_id]
            b = meta["base_id"]; tt = meta["task_type"]
            if b not in idx:
                idx[b] = {}
            idx[b][tt] = r
        return idx

    for mode in run_modes:
        if hasattr(dataset, "chain_mode"):
            dataset.chain_mode = mode 

        dataloader = CUDADataLoader(
            dataset,
            batch_size=1,
            num_workers=args.num_workers,
            shuffle=True,
            collate_fn=lambda x: x[0],
            pin_memory=True,
        )
        modal = dataset.MODAL

        results: List[Dict[str, Any]] = []

        for idx, data in enumerate(
            tqdm(dataloader, desc=f"[{mode}] Rank {global_rank}", total=len(dataloader), position=local_rank)
        ):
            if not data["data_ids"]:
                continue

            data_ids = data["data_ids"]
            text_inputs = data["text_inputs"]

            for data_id, text_input in zip(data_ids, text_inputs):

                try:
                    data_dict = {**data["image_inputs"], **text_input}

                    if "raw_images" in data and data["raw_images"]:
                        data_dict["images"] = [data["raw_images"]]

                    if hasattr(model, "infer_dino_output"):
                        model.infer_dino_output = None
                    start_time=time.time()
                    response = mm_infer_fn(
                        data_dict,
                        model=model,
                        tokenizer=processor.tokenizer,
                        modal=modal,
                        do_sample=False,
                    )

                    if any(key in dataset.BENCHMARK_TYPE for key in ["spatio_temporal_grounding", "st_align", "stvg_miop"]):
                        dino_output = getattr(model, "infer_dino_output", None)
                        prediction = dataset.process_response(
                            data_id, response, dino_output=dino_output, timestamps=data.get("timestamps", [])
                        )
                    elif "spatial_grounding" in dataset.BENCHMARK_TYPE:
                        dino_output = getattr(model, "infer_dino_output", None)
                        prediction = dataset.process_response(data_id, response, dino_output=dino_output)
                    else:
                        prediction = dataset.process_response(data_id, response)
                    end_time=time.time()
                    all_infer_time.append(end_time-start_time)
                    if len(all_infer_time) == 1000:
                        print(f"Average time cost: {sum(all_infer_time)/len(all_infer_time)}")
                        

                except Exception as e:
                    traceback.print_exc()
                    print(f"Error in data_id: {data_id}")
                    prediction = None
                    response = f"ERROR: {e}"

                results.append(
                    {
                        "data_id": data_id,
                        "response": response,
                        "prediction": prediction,
                        "timestamps": data.get("timestamps", []),
                    }
                )

        gathered = [None for _ in range(dist.get_world_size())]
        dist.gather_object(
            obj=results,
            object_gather_list=gathered if global_rank == 0 else None,
            dst=0,
        )

        if global_rank == 0:
            merged = sum(gathered, [])

            if is_vstar and mode == "chain2" and "chain1" in merged_by_mode:
                idx1 = index_results(merged_by_mode["chain1"])
                existing_ids = {r["data_id"] for r in merged}
                for b, tasks in idx1.items():
                    if "vqa" in tasks:
                        vid = tasks["vqa"]["data_id"]
                        if vid not in existing_ids:
                            merged.append(tasks["vqa"])

            valid = [r for r in merged if r["prediction"] is not None]
            print(f"\n[{mode}] Total samples: {len(merged)}, Valid samples for evaluation: {len(valid)}")

            metrics, infos = dataset.evaluate(valid)
            merged_by_mode[mode] = merged 

            print("\n" * dist.get_world_size())
            show_metrics(metrics, f"{args.benchmark}_{mode}")

            all_metrics[f"{mode}"] = metrics

            if base_save:
                os.makedirs(osp.dirname(base_save), exist_ok=True)
                mode_path = f"{base_name}_{mode}{base_ext}"
                with open(mode_path, "w") as f:
                    json.dump(_to_jsonable({"metrics": metrics, "results": infos}), f, indent=4)
                print(f"[{mode}] Results saved to {mode_path}")

        del dataloader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dist.barrier()

    if is_vstar and global_rank == 0 and "chain1" in merged_by_mode and "chain2" in merged_by_mode:
        idx1 = index_results(merged_by_mode["chain1"])
        idx2 = index_results(merged_by_mode["chain2"])

        all_base_ids = set(list(idx1.keys()) + list(idx2.keys()))
        merged_nc: List[Dict[str, Any]] = []

        for b in all_base_ids:
            if "vqa" in idx1.get(b, {}):
                merged_nc.append(idx1[b]["vqa"])
            if "temporal" in idx1.get(b, {}):
                merged_nc.append(idx1[b]["temporal"])
            if "spatial" in idx2.get(b, {}):
                merged_nc.append(idx2[b]["spatial"])

        valid_nc = [r for r in merged_nc if r.get("prediction") is not None]
        print(f"\n[no_chain] Total samples: {len(merged_nc)}, Valid samples for evaluation: {len(valid_nc)}")

        metrics_nc, infos_nc = dataset.evaluate(valid_nc)
        show_metrics(metrics_nc, f"{args.benchmark}_no_chain")
        all_metrics["no_chain"] = metrics_nc

        if base_save:
            mode_path = f"{base_name}_no_chain{base_ext}"
            with open(mode_path, "w") as f:
                json.dump(_to_jsonable({"metrics": metrics_nc, "results": infos_nc}), f, indent=4)
            print(f"[no_chain] Results saved to {mode_path}")

    if global_rank == 0 and len(all_metrics) > 1:
        print("\n=== Summary across modes ===")
        for m, met in all_metrics.items():
            print(f"\nMode: {m}")
            show_metrics(met, f"{args.benchmark}_{m}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()
    avg_time = sum(all_infer_time) / len(all_infer_time) if all_infer_time else 0.0
    print('avg_time is ',avg_time,'s')


if __name__ == "__main__":
    main()
