import copy

from .jointcoord_head import JointCoordHead

def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop('name')
    if name == 'JointCoordHead':
        return JointCoordHead(**head_cfg)
    else:
        raise NotImplementedError