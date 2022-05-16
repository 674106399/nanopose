import torch
import torch.nn as nn

class KLDiscretLoss(nn.Module):
    def __init__(self):
        super(KLDiscretLoss, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim=1)  # [B,LOGITS]
        self.criterion_ = nn.KLDivLoss(reduction='none')

    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        loss = torch.mean(self.criterion_(scores, labels), dim=1)
        return loss

    def forward(self, outputs, targets, is_3d=False, target_weight=None):
        if is_3d:
            output_x, output_y, output_z = outputs
            target_x, target_y, target_z = targets
        else:
            output_x, output_y = outputs
            target_x, target_y = targets

        num_joints = output_x.size(1)
        loss = 0

        for idx in range(num_joints):
            coord_x_pred = output_x[:, idx].squeeze()
            coord_y_pred = output_y[:, idx].squeeze()
            coord_x_gt = target_x[:, idx].squeeze()
            coord_y_gt = target_y[:, idx].squeeze()

            weight = target_weight[:,idx].squeeze()

            loss += (self.criterion(coord_x_pred, coord_x_gt).mul(weight).mean())
            loss += (self.criterion(coord_y_pred, coord_y_gt).mul(weight).mean())
            if is_3d:
                coord_z_pred = output_z[:, idx].squeeze()
                coord_z_gt = target_z[:, idx].squeeze()
                loss += (self.criterion(coord_z_pred, coord_z_gt).mul(weight).mean())
        return loss / num_joints