# Copyright 2022 Tau.
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

import time

import torch
import torch.nn as nn

from ..backbone import build_backbone
from ..neck import build_neck
from ..head import build_head

class TopDown(nn.Module):
    def __init__(
        self,
        backbone_cfg,
        neck_cfg=None,
        head_cfg=None,
    ):
        super(TopDown, self).__init__()

        self.backbone = build_backbone(backbone_cfg)

        if neck_cfg is not None:
            self.neck = build_neck(neck_cfg)

        if head_cfg is not None:
            self.head = build_head(head_cfg)

        self.epoch = 0

    def forward(self, x, labels=None):
        x = self.backbone(x)
        if hasattr(self, 'neck'):
            x = self.neck(x)
        if hasattr(self, 'head'):
            if self.training:
                x = self.head(x, labels)
            else:
                x = self.head(x)
        return x

    def inference(self, meta):
        with torch.no_grad():
            # cuda kernel是异步的，不能直接time.time()测试时间
            # 要先用torch.cuda.synchronize等待所有线程执行完毕
            torch.cuda.synchronize()
            time1 = time.time()
            preds = self(meta['img'])
            torch.cuda.synchronize()
            time2 = time.time()
            print('forward time: {:.3f}s'.format((time2-time1)), end=' | ')
            results = self.head.post_process(preds, meta)
            torch.cuda.synchronize()
            print('decode time: {:.3f}s'.format((time.time()-time2)), end=' | ')
        return results

    def forward_train(self, x, gt_meta):
        preds = self(x, gt_meta['labels'])
        loss, loss_states = self.head.loss(preds, gt_meta['labels'])

        return preds, loss, loss_states

    def set_epoch(self, epoch):
        self.epoch = epoch