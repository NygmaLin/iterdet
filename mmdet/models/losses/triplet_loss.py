import torch
import torch.nn as nn
from ..builder import LOSSES


@LOSSES.register_module()
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2, nu=0.0, loss_weight=1.0):
        super(TripletLoss, self).__init__()
        self._margin = margin
        self._nu = nu
        self.criterion = torch.nn.MarginRankingLoss(margin=self._margin)
        self.loss_weight = loss_weight

    def forward(self, anchors, positives, negatives):
        d_ap = torch.sqrt(torch.sum((positives - anchors) ** 2, dim=1) + 1e-8)
        d_an = torch.sqrt(torch.sum((negatives - anchors) ** 2, dim=1) + 1e-8)

        target = torch.FloatTensor(d_ap.size()).fill_(1)
        target = target.cuda()
        # criterion = torch.nn.MarginRankingLoss(margin=self._margin)
        loss_triplet = self.criterion(d_an, d_ap, target)
        # loss_embedd = embedded[0].norm(2) + embedded[1].norm(2) + embedded[2].norm(2)

        pair_cnt = int(torch.sum((d_ap - d_an + self._margin) > 0.0))
        pair_cnt = torch.Tensor(pair_cnt)
        loss_triplet *= self.loss_weight

        return loss_triplet, pair_cnt
