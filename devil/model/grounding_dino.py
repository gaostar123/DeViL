# grounding_dino.py

import copy
import torch

from .g_dino.GroundingDINO import build_groundingdino
from .g_dino.matcher import build_matcher
from .g_dino.criterion import SetCriterion

def build_g_dino(config, num_classes=1, **kwargs):
    g_dino = build_groundingdino(config, num_classes=num_classes)
    return g_dino

def build_g_dino_criterion(config, num_classes=1):
    matcher = build_matcher(config, num_classes=num_classes)
    losses = ['labels', 'boxes', 'temporal_consistency']

    weight_dict = {
        'loss_ce': config.cls_loss_coef,
        'loss_bbox': config.bbox_loss_coef,
        'loss_giou': config.giou_loss_coef,
        'loss_feat_consistency': config.feat_consistency_loss_coef,
        'loss_geom_consistency': config.geom_consistency_loss_coef,
    }

    if config.GroundingDINO.aux_loss:
        aux_weight_dict = {}
        for i in range(config.GroundingDINO.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if config.GroundingDINO.two_stage_type != 'no':
        interm_weight_dict = {}
        interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items()})
        weight_dict.update(interm_weight_dict)

    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=config.eos_coef,
        losses=losses
    )
    return criterion
