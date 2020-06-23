import random
import torch
import numpy as np
from ..builder import BBOX_SAMPLERS


@BBOX_SAMPLERS.register_module()
class EmbedSampler(object):

    def __init__(self,
                 num_by_instance,
                 add_gt_as_proposals,
                 ):
        self.num_by_instance = num_by_instance
        self.add_gt_as_proposals = add_gt_as_proposals

    def sample(self,
               assign_results,
               bboxes,
               gt_bboxes,
               gt_labels,
               **kwargs
               ):
        sampling_results = []
        bboxes = bboxes[:, :4]
        used_instances = []
        for instance, gt_bbox in zip(gt_labels, gt_bboxes):
            proposals = bboxes[assign_results.labels == instance]
            if len(proposals) == 0:
                continue
            else:
                used_instances.append(instance)
            if self.add_gt_as_proposals:
                gt_bbox = gt_bbox.unsqueeze(0)
                proposals = torch.cat([proposals, gt_bbox], dim=0)
            if len(proposals) >= self.num_by_instance:
                selected_idx = np.random.choice(list(range(len(proposals))), self.num_by_instance, replace=False)
            else:
                selected_idx = np.random.choice(list(range(len(proposals))), self.num_by_instance, replace=True)
            selected_idx = torch.Tensor(selected_idx).type(torch.long)
            selected_proposals = proposals[selected_idx]
            sampling_results.append(selected_proposals)
        if sampling_results == [] or len(used_instances) == 1:
            sampling_results =  torch.Tensor([]).cuda()
        else:
            sampling_results = torch.cat(sampling_results, dim=0)

        return sampling_results