import torch
import torch.nn.functional as F
from torch import nn

from .dino_util import box_ops
from .misc import (nested_tensor_from_tensor_list,
                  get_world_size, is_dist_avail_and_initialized)

from .GroundingDINO.utils import sigmoid_focal_loss

from einops import rearrange


class SetCriterion(nn.Module):
    """ This class computes the loss for ReferFormer.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.focal_alpha = focal_alpha
        self.mask_out_stride = 4
        
    def loss_temporal_consistency(self, outputs, targets, indices, num_boxes, log=True):
        """
        Computes cross-frame consistency loss for the matched tracks.
        This loss is calculated based on the matching result from the HungarianMatcher.
        """
        if 'pred_inst_embeds' not in outputs:
            return {}

        inst_embeds = outputs['pred_inst_embeds'] # (B, T, N, D)
        pred_boxes = outputs['pred_boxes']       # (B, T, N, 4)
        
        if inst_embeds.ndim == 3:
            inst_embeds = inst_embeds.unsqueeze(0)
        
        B, T, N, _ = inst_embeds.shape
        if T <= 1: # No temporal loss for single frames
            return {}

        total_loss_feat = 0.0
        total_loss_geom = 0.0
        
        # Loop over each video in the batch
        for i in range(B):
            # Get the index of the query that was matched to the ground truth for this video
            # indices[i] is a tuple (src_indices, tgt_indices)
            # For single-object tracking, src_indices will have one element
            if len(indices[i][0]) == 0:
                continue # No match found for this batch item
            
            matched_query_idx = indices[i][0][0]

            # Get the entire predicted track (embeddings and boxes) for the matched query
            query_embeds_track = inst_embeds[i, :, matched_query_idx, :] # (T, D)
            query_boxes_track = pred_boxes[i, :, matched_query_idx, :]   # (T, 4)
            
            # Loop over adjacent frames for this specific track
            for t in range(T - 1):
                # 1. Feature consistency loss (1 - cosine similarity)
                f_t = query_embeds_track[t]
                f_t_plus_1 = query_embeds_track[t+1]
                # Add a small epsilon to avoid NaN for zero vectors
                cos_sim = F.cosine_similarity(f_t, f_t_plus_1, dim=0, eps=1e-8)
                total_loss_feat += (1.0 - cos_sim)
                
                # 2. Geometry consistency loss (1 - GIoU)
                b_t = query_boxes_track[t].unsqueeze(0)
                b_t_plus_1 = query_boxes_track[t+1].unsqueeze(0)
                giou = box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(b_t),
                    box_ops.box_cxcywh_to_xyxy(b_t_plus_1)
                )
                total_loss_geom += (1.0 - giou.squeeze())

        # Normalize the loss by the total number of adjacent pairs across the batch
        num_pairs = B * (T - 1)
        if num_pairs == 0:
            return {}
            
        losses = {
            'loss_feat_consistency': total_loss_feat / num_pairs,
            'loss_geom_consistency': total_loss_geom / num_pairs,
        }
        
        return losses
    
    
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        _, nf, nq = src_logits.shape[:3]
        src_logits = rearrange(src_logits, 'b t q k -> b (t q) k')

        # judge the valid frames
        valid_indices = []
        valids = [target['valid'] for target in targets]
        for valid, (indice_i, indice_j) in zip(valids, indices):
            valid_ind = valid.nonzero().flatten()   # vilid frame ids
            valid_i = valid_ind * nq + indice_i
            valid_j = valid_ind + indice_j * nf
            valid_indices.append((valid_i, valid_j))
        idx = self._get_src_permutation_idx(valid_indices) # NOTE: use valid indices
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, valid_indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        if self.num_classes == 1: # binary referred
            target_classes[idx] = 0
        else:
            target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            pass
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']
        bs, nf, nq = src_boxes.shape[:3]
        src_boxes = src_boxes.transpose(1, 2)

        idx = self._get_src_permutation_idx(indices)
        src_boxes = src_boxes[idx]
        src_boxes = src_boxes.flatten(0, 1)  # [b*t, 4]

        target_boxes = torch.cat([t['boxes'] for t in targets], dim=0)  # [b*t, 4]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'temporal_consistency': self.loss_temporal_consistency,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        target_valid = torch.stack([t["valid"] for t in targets], dim=0).reshape(-1) # [B, T] -> [B*T]
        num_boxes = target_valid.sum().item()
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            indices = self.matcher(outputs['interm_outputs_for_matching_pre'], targets)
            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, outputs['interm_outputs'], targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


