#
# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------
#

from __future__ import annotations

from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

# TODO(Guarin, 07/25): Move transformers classes into LightlyTrain. We should actually
# be able to use the original Mask2Former classes which don't depend on transformers:
# - https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py
# - https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py
# If we replace transformers we'll still have to add SciPy as a dependency as parts
# of the loss depend on it.
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerHungarianMatcher,
    Mask2FormerLoss,
    sample_point,
    sigmoid_cross_entropy_loss,
    dice_loss
)

class Mask2FormerAdaptiveLoss(Mask2FormerLoss):
    def sample_points_using_uncertainty(
            self,
            logits: torch.Tensor,
            uncertainty_function,
            num_points: int,
            oversample_ratio: int,
            importance_sample_ratio: float,
            invalids: torch.Tensor = None,  # New parameter: bool tensor of shape [num_boxes, 1, W, H]
    ) -> torch.Tensor:
        """
        This function is meant for sampling points in [0, 1] * [0, 1] coordinate space based on their uncertainty. The
        uncertainty is calculated for each point using the passed `uncertainty function` that takes points logit
        prediction as input.

        Args:
            logits (`float`):
                Logit predictions for P points.
            uncertainty_function:
                A function that takes logit predictions for P points and returns their uncertainties.
            num_points (`int`):
                The number of points P to sample.
            oversample_ratio (`int`):
                Oversampling parameter.
            importance_sample_ratio (`float`):
                Ratio of points that are sampled via importance sampling.
            invalids (`torch.Tensor`, optional):
                Boolean tensor of shape [num_boxes, W, H] where False indicates valid points and True indicates invalid points.
                Coordinates are normalized and consistent with logits coordinate system.

        Returns:
            point_coordinates (`torch.Tensor`):
                Coordinates for P sampled points.
        """

        num_boxes = logits.shape[0]
        num_points_sampled = int(num_points * oversample_ratio)

        def sample_from_valid_regions(num_points_to_sample):
            """Sample points from valid regions defined by invalids mask"""
            if invalids is None:
                # If no invalids mask provided, use uniform random sampling
                return torch.rand(num_boxes, num_points_to_sample, 2, device=logits.device)

            point_coordinates_list = []

            for i in range(num_boxes):
                # Get valid pixel indices for current box
                valid_mask = ~invalids[i]  # Shape: [W, H]
                valid_indices = torch.nonzero(valid_mask, as_tuple=False)  # Shape: [N_valid, 2]

                if len(valid_indices) == 0:
                    # If no valid points, fall back to uniform sampling
                    points = torch.rand(num_points_to_sample, 2, device=logits.device)
                    point_coordinates_list.append(points)
                    continue
                    # raise NotImplementedError

                # Get spatial dimensions of invalids mask
                W, H = invalids.shape[-2], invalids.shape[-1]

                # Randomly select center points from valid pixels (with replacement)
                selected_indices = torch.randint(0, len(valid_indices), (num_points_to_sample,),
                                                 device=logits.device)
                center_coords = valid_indices[selected_indices]  # Shape: [num_points_to_sample, 2]

                # Convert discrete coordinates to normalized center coordinates
                # Add 0.5 to get pixel center coordinates, then normalize
                center_coords_normalized = (center_coords.float() + 0.5) / torch.tensor([W, H],
                                                                                        device=logits.device)

                # Add random offset in range [-0.5, +0.5] pixels around center
                random_offset = (torch.rand(num_points_to_sample, 2, device=logits.device) - 0.5) / torch.tensor([W, H],
                                                                                                                 device=logits.device)

                # Final coordinates: center + random offset
                points = center_coords_normalized + random_offset

                # Clamp to ensure coordinates stay in [0, 1] range
                points = torch.clamp(points, 0.0, 1.0)

                point_coordinates_list.append(points)

            return torch.stack(point_coordinates_list, dim=0)

        # Get point coordinates using the new sampling method
        point_coordinates = sample_from_valid_regions(num_points_sampled)

        # Get sampled prediction value for the point coordinates
        point_logits = sample_point(logits, point_coordinates, align_corners=False)
        # Calculate the uncertainties based on the sampled prediction values of the points
        point_uncertainties = uncertainty_function(point_logits)

        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points

        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_points_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
        idx += shift[:, None]
        point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)

        if num_random_points > 0:
            # Sample additional random points from valid regions
            random_points = sample_from_valid_regions(num_random_points)
            point_coordinates = torch.cat([point_coordinates, random_points], dim=1)

        return point_coordinates


    def loss_masks(self, masks_queries_logits, mask_labels, indices, num_masks=1, invalids=None):
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        # shape (batch_size * num_queries, height, width)
        pred_masks = masks_queries_logits[src_idx]
        # shape (batch_size, num_queries, height, width)
        # pad all and stack the targets to the num_labels dimension
        if invalids is None:
            invalids = [torch.zeros_like(mask_label, dtype=torch.bool) for mask_label in mask_labels]

        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        invalids, _ = self._pad_images_to_max_in_batch(invalids)

        target_masks = target_masks[tgt_idx]
        invalids = invalids[(tgt_idx[0], torch.zeros_like(tgt_idx[1]))]

        # No need to upsample predictions as we are using normalized coordinates
        pred_masks = pred_masks[:, None]
        target_masks = target_masks[:, None]

        # Sample point coordinates
        with torch.no_grad():
            point_coordinates = self.sample_points_using_uncertainty(
                pred_masks,
                lambda logits: self.calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
                invalids
            )

            point_labels = sample_point(target_masks, point_coordinates, align_corners=False).squeeze(1)
            valids = sample_point(invalids[:, None].float(), point_coordinates, align_corners=False).squeeze(1) < 0.5

        point_logits = sample_point(pred_masks, point_coordinates, align_corners=False).squeeze(1)
        point_logits = point_logits[valids].unsqueeze(dim=0)
        point_labels = point_labels[valids].unsqueeze(dim=0)
        losses = {
            "loss_mask": sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

        del pred_masks
        del target_masks
        return losses


class MaskClassificationLoss(Mask2FormerAdaptiveLoss):  # type: ignore[misc]
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        num_labels: int,
        no_object_coefficient: float,
    ):
        nn.Module.__init__(self)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.num_labels = num_labels
        self.eos_coef = no_object_coefficient
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.matcher = Mask2FormerHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=class_coefficient,
        )

    @torch.compiler.disable  # type: ignore[misc]
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        targets: List[dict[str, torch.Tensor]],
        class_queries_logits: Optional[torch.Tensor] = None,
        invalids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        mask_labels = [
            target["masks"].to(masks_queries_logits.dtype) for target in targets
        ]
        class_labels = [target["labels"].long() for target in targets]

        indices = self.matcher(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_queries_logits,
            class_labels=class_labels,
        )

        loss_masks = self.loss_masks(masks_queries_logits, mask_labels, indices, invalids)  # type: ignore[no-untyped-call]
        loss_classes = self.loss_labels(class_queries_logits, class_labels, indices)  # type: ignore[arg-type]

        return {**loss_masks, **loss_classes}

    def loss_masks(self, masks_queries_logits, mask_labels, indices, invalids):  # type: ignore
        loss_masks = super().loss_masks(masks_queries_logits, mask_labels, indices, 1, invalids)

        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=masks_queries_logits.device
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        num_masks_tensor = torch.clamp(num_masks_tensor / world_size, min=1)

        for key in loss_masks.keys():
            loss_masks[key] = loss_masks[key] / num_masks_tensor

        return loss_masks

    def loss_total(self, losses_all_layers: dict[str, torch.Tensor]) -> torch.Tensor:
        loss_total = None
        for loss_key, loss in losses_all_layers.items():
            if "mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient
            elif "cross_entropy" in loss_key:
                weighted_loss = loss * self.class_coefficient
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)
        return loss_total  # type: ignore
