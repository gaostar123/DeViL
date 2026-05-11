from .base import BaseEvalDataset
from .charades_sta import CharadesSTADataset
from .mvbench import MVBenchDataset
from .tempcompass import TempCompassDataset
from .videomme import VideoMMEDataset
from .hc_stvg_v1 import HCSTVGv1Dataset
from .hc_stvg_v2 import HCSTVGv2Dataset
from .vidstg import VidSTGDataset
from .v_star import VStarDataset
from .activitynet_tg import ActivityNetTGDataset
from .nextgqa import NEXTGQADataset

DATASET_REGISTRY = {
    "videomme": VideoMMEDataset,
    "mvbench": MVBenchDataset,
    "tempcompass": TempCompassDataset,
    "charades_sta": CharadesSTADataset,
    "hc_stvg_v1": HCSTVGv1Dataset,
    "hc_stvg_v2": HCSTVGv2Dataset,
    "vidstg": VidSTGDataset,
    "v-star": VStarDataset,
    "activitynet_tg": ActivityNetTGDataset,
    "next_gqa": NEXTGQADataset,
}


def build_dataset(benchmark_name: str, **kwargs) -> BaseEvalDataset:
    assert benchmark_name in DATASET_REGISTRY, f"Unknown benchmark: {benchmark_name}, available: {DATASET_REGISTRY.keys()}"
    return DATASET_REGISTRY[benchmark_name](**kwargs)
