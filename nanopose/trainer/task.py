# Modifications Copyright 2022 Tau.
# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

import os.path as osp

import torch
import torch.distributed as dist
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only

from ..model.arch import build_model
from ..model.weight_averager import build_weight_averager
from ..utils.data_aug import DataAugmentation, mkdir

class TrainingTask(LightningModule):
    """
    Pytorch Lightning module of a general training task.
    Including training, evaluating and testing.
    Args:
        cfg: Training configurations
        evaluator: Evaluator for evaluating the model performance.
    """

    def __init__(self, cfg, evaluator=None):
        super(TrainingTask, self).__init__()
        self.cfg = cfg
        self.model = build_model(cfg.model)
        self.evaluator = evaluator
        self.save_flag = -10
        self.weight_averager = None
        if "weight_averager" in cfg.model:
            self.weight_averager = build_weight_averager(
                cfg.model.weight_averager, device=self.device
            )
            self.avg_model = copy.deepcopy(self.model)
        self.aug = DataAugmentation().to(self.model.device)

    def _preprocess_input_batch(self, batch):
        x, y = batch
        target_coord, target_simdr, hand_type, hand_valid, is_3d, gesture_label, is_double, limb_target, img_arr, img_path = y

        target_coord = target_coord / self.cfg.output_hm_shape[0] - 0.5
        target_coord[:, :, 2] = target_coord[:, :, 2] - target_coord[:,
                                                        self.cfg.root_idx:self.cfg.root_idx + 1, 2]

        if self.training:
            x = self.aug(x)
            mask3d = torch.as_tensor([a for a in is_3d for _ in range(self.cfg.num_joints)], dtype=is_3d.dtype)
            labels = {
                'target_coord': target_coord,
                'mask3d': mask3d
            }
        else:
            labels = None

        gt_meta = {
            'y': (target_coord, target_simdr, hand_type, hand_valid, is_3d, gesture_label, is_double, limb_target, img_arr, img_path),
            'labels': labels
        }

        return x, gt_meta

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, gt_meta = self._preprocess_input_batch(batch)

        preds, loss, loss_states = self.model.forward_train(x, gt_meta)

        # log train losses
        if self.global_step % self.cfg.log.interval == 0:
            lr = self.optimizers().param_groups[0]['lr']
            log_msg = 'Train| Epoch{}/{}|Iter{}({})| lr:{:.2e}| '.format(
                self.current_epoch + 1,
                self.cfg.schedule.total_epochs,
                self.global_step,
                batch_idx,
                lr,
            )
            self.scalar_summary('Train_loss/lr', 'Train', lr, self.global_step)
            for loss_name in loss_states:
                log_msg += '{}:{:.4f}| '.format(
                    loss_name, loss_states[loss_name].mean().item()
                )
                self.scalar_summary(
                    'Train_loss/' + loss_name,
                    'Train',
                    loss_states[loss_name].mean().item(),
                    self.global_step,
                )
            self.logger.info(log_msg)

        return loss

    def training_epoch_end(self, outputs) -> None:
        self.trainer.save_checkpoint(osp.join(self.cfg.save_dir, 'model_last.ckpt'))
        self.lr_schedulers.step()

    def validation_step(self, batch, batch_idx):
        x, gt_meta = self._preprocess_input_batch(batch)

        if self.weight_averager is not None:
            preds, loss, loss_states = self.avg_model.forward_train(batch, gt_meta)
        else:
            preds, loss, loss_states = self.model(batch, gt_meta)

        if batch_idx % self.cfg.log.interval == 0:
            lr = self.optimizers().param_groups[0]['lr']
            log_msg = 'Val|Epoch{}/{}|Iter{}({})| lr:{:.2e}| '.format(
                self.current_epoch + 1,
                self.cfg.schedule.total_epochs,
                self.global_step,
                batch_idx,
                lr,
            )
            for loss_name in loss_states:
                log_msg += '{}:{:.4f}| '.format(
                    loss_name, loss_states[loss_name].mean().item()
                )
            self.logger.info(log_msg)

        dets = self.model.head.post_process(preds, batch)
        return dets

    def validation_epoch_end(self, outputs):
        results = {}
        for res in outputs:
            results.update(res)
        all_results = (
            gather_results(results)
            if dist.is_available() and dist.is_initialized()
            else results
        )
        if all_results:
            eval_results = self.evaluator.evaluate(
                all_results, self.cfg.save_dir, rank=self.local_rank
            )
            metric = eval_results[self.cfg.evaluator.save_key]
            # save best model
            if metric >= self.save_flag:
                self.save_flag = metric
                best_save_path = osp.join(self.cfg.save_dir, 'modle_best')
                mkdir(self.local_rank, best_save_path)
                self.trainer.save_checkpoint(
                    osp.join(best_save_path, 'model_best.ckpt')
                )
                self.save_model_state(
                    osp.join(best_save_path, 'nanopose_model_best.pth')
                )