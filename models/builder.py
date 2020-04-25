from . import backbones, heads

__all__ = [
    'build_backbone', 'build_head'
]


def build_backbone(cfg):
    attr = getattr(backbones, cfg.pop('type'))
    return attr(**cfg)


def build_head(cfg):
    attr = getattr(heads, cfg.pop('type'))
    return attr(**cfg)