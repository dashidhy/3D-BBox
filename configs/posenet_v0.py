# model settings
model = dict(
    backbone_type='resnet34',
    backbone_cfg=dict(pretrained=True),
    head_type='BoxHead',
    head_cfg=dict(in_size=512*7*7,
                  num_bins=2,
                  d_hidden_sizes=[512],
                  a_hidden_sizes=[256],
                  c_hidden_sizes=[256])
)