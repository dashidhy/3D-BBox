from . import backbones, heads, losses

__all__ = [
    'build_from', 'build_backbone', 'build_head', 'build_loss'
]

def build_from(module, cfg):
    attr = getattr(module, cfg.pop('type'))
    return attr(**cfg)


def build_backbone(cfg):
    return build_from(backbones, cfg)


def build_head(cfg):
    return build_from(heads, cfg)


def build_loss(cfg):
    return build_from(losses, cfg)