from . import backbones, heads, losses

__all__ = [
    'build_backbone', 'build_head', 'build_loss'
]


def build_backbone(cfg):
    attr = getattr(backbones, cfg.pop('type'))
    return attr(**cfg)


def build_head(cfg):
    attr = getattr(heads, cfg.pop('type'))
    return attr(**cfg)


def build_loss(cfg):
    attr = getattr(losses, cfg.pop('type'))
    return attr(**cfg)