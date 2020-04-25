# model settings
model = dict(
    backbone_cfg=dict(type='resnet34',
                      pretrained=True),
    head_cfg=dict(type='BoxHead',
                  in_size=512*7*7,
                  num_bins=2,
                  d_hidden_sizes=[512],
                  a_hidden_sizes=[256],
                  c_hidden_sizes=[256])
)