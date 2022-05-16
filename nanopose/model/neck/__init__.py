import copy

from .gap_neck import GlobalAveragePooling

def build_neck(cfg):
    neck_cfg = copy.deepcopy(cfg)
    name = neck_cfg.pop('name')
    if name in ['AdaptiveAvgPool2d', 'GAP', 'GlobalAveragePooling']:
        return GlobalAveragePooling(**neck_cfg)
    else:
        raise NotImplementedError