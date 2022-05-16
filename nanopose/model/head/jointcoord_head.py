import numpy as np
import torch
import torch.nn as nn

import torch.distributions as distributions

from ..module.init_weights import normal_init, constant_init
from ..module.real_nvp import RealNVP, nets, nett, nets3d, nett3d
from ..head.tokenpose_head import TokenPose
from ..loss.rle_loss import RLELoss3D
from ..loss.simdr_loss import KLDiscretLoss
from ..loss.bone_loss import JointBoneLoss

class JointCoordHead(nn.Module):
    '''Head returns joint coordinates, such as :
        regression head with fully connected layers,
        soft-argmax head,
        residual log-likelihood head with flow models

    Args:

    '''

    def __init__(
        self,
        fc_type,
        in_channels,
        in_res,
        num_joints=21,
        use_rle=False,
        pred_hand_type=True,
        pred_hand_valid=True,
        fc_cfg=None,
        loss_cfg=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.in_res = in_res
        self.num_joints = num_joints
        self.use_rle = use_rle
        self.pred_hand_type = pred_hand_type
        self.pred_hand_valid = pred_hand_valid
        self.fc_cfg = fc_cfg
        self.loss_cfg = loss_cfg

        if hasattr(fc_cfg, 'tokenpose'):
            self.tokenpose_cfg = fc_cfg.pop('tokenpose')
        if hasattr(fc_cfg, 'simdr'):
            self.simdr_cfg = fc_cfg.pop('simdr')

        self.jc_loss = RLELoss3D(use_jc_ohkm=self.loss_cfg.use_jc_ohkm)
        self.simdr_loss = KLDiscretLoss()
        self.hand_loss = nn.BCEWithLogitsLoss()
        self.bone_loss = JointBoneLoss(self.num_joints)

        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        if self.pred_hand_type:
            self.hand_type_head = nn.Conv2d(self.in_channels, 1, kernel_size=self.in_res, stride=1, padding=0)

        if self.pred_hand_valid:
            self.hand_type_head = nn.Conv2d(self.in_channels, 1, kernel_size=self.in_res, stride=1, padding=0)

        if self.fc_type == 'fc':
            self.fc_coords = nn.Linear(self.in_channels, self.num_joints * 3)
        elif self.fc_type == 'conv':
            self.fc_coords = nn.Conv2d(self.in_channels, self.num_joints * 3, kernel_size=self.in_res, stride=1, padding=0)
        elif self.fc_type == 'TokenPose+SimDR':
            feature_size = [self.in_res, self.in_res]
            feat_dim = self.simdr_cfg.feat_dim
            heatmap_dim = feat_dim * 3
            self.fc_coords = TokenPose(feature_size=feature_size,
                                       num_joints=self.num_joints,
                                       channels=self.in_channels,
                                       heatmap_dim=heatmap_dim,
                                       **self.tokenpose_cfg)
            self.linspace = torch.arange(feat_dim).float() / feat_dim

        if self.use_rle:
            self.fc_sigma = nn.Conv2d(self.in_channels, self.num_joints * 3, kernel_size=self.in_res, stride=1, padding=0)

            prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2), validate_args=False)
            masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
            prior3d = distributions.MultivariateNormal(torch.zeros(3), torch.eye(3), validate_args=False)
            masks3d = torch.from_numpy(np.array([[0, 0, 1], [1, 1, 0]] * 3).astype(np.float32))

            self.flow2d = RealNVP(nets, nett, masks, prior)
            self.flow3d = RealNVP(nets3d, nett3d, masks3d, prior3d)


    def init_weights(self):
        if isinstance(self.fc_coords, nn.Conv2d):
            normal_init(self.fc_coords, std=0.01)
            constant_init(self.fc_coords, 0)
        elif isinstance(self.fc_coords, nn.Linear):
            normal_init(self.fc_coords, std=0.01)
            constant_init(self.fc_coords, 0)

        if isinstance(self.fc_sigma, nn.Conv2d):
            normal_init(self.fc_sigma, std=0.01)
            constant_init(self.fc_sigma, 0)
        elif isinstance(self.fc_sigma, nn.Linear):
            normal_init(self.fc_sigma, std=0.01)
            constant_init(self.fc_sigma, 0)

        print("Finish initialize NanoPose Head.")

    def loss(self, preds, gt_meta):
        """Compute losses.

        Args:
            preds (tuple): Prediction outputs.
            gt_meta (dict): Ground truth information.

        Returns:
            loss (Tensor): Loss tensor.
            loss_states (dict): State dict of each loss.
        """

        target_coord, target_simdr, gt_hand_type, gt_hand_valid, is_3d, gesture_label, is_double, _, img_arr, img_path = gt_meta

        joint_coord3d, pred_xyz, featuremap, scores, nf_loss, sigma, pred_hand_type, pred_hand_valid = preds

        # post process
        joint_coord3d = joint_coord3d.reshape(-1, 21, 3)
        pred_x, pred_y, pred_z = pred_xyz
        target_x, target_y, target_z = target_simdr
        pred_hand_type = pred_hand_type.reshape(-1, 1)
        pred_hand_valid = pred_hand_valid.reshape(-1, 1)

        hand_mask = (gt_hand_valid == 1.0)

        joint_coord3d = joint_coord3d[hand_mask]
        pred_x, pred_y, pred_z = pred_x[hand_mask], pred_y[hand_mask], pred_z[hand_mask]
        target_x, target_y, target_z = target_x[hand_mask], target_y[hand_mask], target_z[hand_mask]
        nf_loss = nf_loss[hand_mask]
        sigma = sigma[hand_mask]
        target_coord = target_coord[hand_mask]
        pred_hand_type = pred_hand_type[hand_mask & ~is_double].squeeze()
        if len(pred_hand_type.shape) == 0:
            pred_hand_type = pred_hand_type.unsqueeze(0)
        gt_hand_type = gt_hand_type[hand_mask & ~is_double]
        pred_hand_valid = pred_hand_valid.squeeze()
        if len(pred_hand_valid.shape) == 0:
            pred_hand_valid = pred_hand_valid.unsqueeze(0)
        is_3d = is_3d[hand_mask]
        is_2d = ~is_3d
        num_3d = is_3d.sum()
        num_2d = is_2d.sum()
        avg2d3d = int(num_3d > 0) + int(num_2d > 0)

        ### calc loss ###
        jc_loss, simdr_loss, bone_loss, hand_loss, valid_loss = 0., 0., 0., 0., 0.
        if num_3d > 0:
            jc_loss += self.joint_loss(joint_coord3d[is_3d], nf_loss[is_3d], sigma[is_3d], target_coord[is_3d],
                                       balance_weight=True)
            simdr_loss += self.kl_loss((pred_x, pred_y, pred_z), (target_x, target_y, target_z), is_3d=True)
            bone_loss += self.bone_loss(joint_coord3d[is_3d], target_coord[is_3d])
        if num_2d > 0:
            jc_loss += self.joint_loss(joint_coord3d[is_2d, :, :2], nf_loss[is_2d, :, :2], sigma[is_2d, :, :2],
                                       target_coord[is_2d, :, :2])
            simdr_loss += self.kl_loss((pred_x, pred_y), (target_x, target_y), is_3d=False)
            bone_loss += self.bone_loss(joint_coord3d[is_2d, :, :2], target_coord[is_2d, :, :2])

        hand_loss = self.hand_loss(pred_hand_type, gt_hand_type)
        valid_loss = self.hand_loss(pred_hand_valid, gt_hand_valid)

        jc_loss /= avg2d3d
        simdr_loss /= avg2d3d
        bone_loss /= avg2d3d

        loss = 0.
        loss += jc_loss * self.loss_cfg.jc_loss.loss_weight
        loss += simdr_loss * self.loss_cfg.simdr_loss.loss_weight
        loss += bone_loss * self.loss_cfg.bone_loss.loss_weight
        loss += hand_loss * self.loss_cfg.hand_loss.loss_weight
        loss += valid_loss * self.loss_cfg.hand_loss.loss_weight

        loss_states = {
            'tjc': jc_loss,
            'tkl': simdr_loss,
            'tbo': bone_loss,
            'thd': hand_loss,
            'tva': valid_loss,
            'tloss': loss
        }

    def eval(self, preds, gt_meta):
        target_coord, target_simdr, gt_hand_type, gt_hand_valid, is_3d, gesture_label, is_double, _, img_arr, img_path = gt_meta
