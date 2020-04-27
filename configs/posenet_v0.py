# global
NUM_BINS = 4

# model settings
model = dict(
    type = 'PoseNet',
    backbone_cfg = dict(type = 'resnet34', 
                        pretrained = True),
    head_cfg = dict(type = 'BoxHead',
                    in_size = 512 * 7 * 7,
                    num_bins = NUM_BINS,
                    d_hidden_sizes = [512],
                    a_hidden_sizes = [256],
                    c_hidden_sizes = [256])
)

# loss settings
dimension_loss = dict(
    type = 'Dimension_Loss',
    base_loss_cfg = dict(type = 'MSE'),
    avg_dim = [1.61057209, 1.47745965, 3.52359498] # (h, w, l)
)

pose_loss = dict(
    type = 'Pose_Loss',
    base_conf_cfg = dict(type = 'CrossEntropy'),
    base_reg_cfg = dict(type = 'Cosine_Expansion', 
                        normalize = True),
    num_bins = NUM_BINS,
    bin_range_degree = 120
)