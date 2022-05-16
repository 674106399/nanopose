import torch
import torch.nn as nn

class L1Loss(nn.Module):
    def __init__(self, use_jc_ohkm=False):
        super(L1Loss, self).__init__()
        self.use_jc_ohkm = use_jc_ohkm

    def ohkm(self, loss, topk=8):
        ohkm_loss = 0.
        for i in range(loss.size(0)):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / topk
        ohkm_loss /= loss.size(0)
        return ohkm_loss

    def forward(self, depth_out, depth_gt, weights=None, balance_weight=False):
        loss = torch.abs(depth_out - depth_gt)

        if weights is not None:
            loss = loss * weights

        if balance_weight:
            loss[:, :, :2] *= 2

        res = loss.sum() / depth_out.shape[0]
        if self.use_jc_ohkm:
            return res + self.ohkm(loss)
        else:
            return res