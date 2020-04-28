# global
__NUM_BINS = 4

# dataset settings
dataset_cfg = dict(
    kitti_root = './data/kitti',
    img_norm = {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
    del_labels = ('sample', 'type', 'class')
)

# model settings
model_cfg = dict(
    
    type = 'PoseNet',
    
    backbone_cfg = dict(type = 'resnet34',
                        pretrained = True,
                        progress = False),
    
    head_cfg = dict(type = 'BoxHead',
                    in_size = 512 * 7 * 7,
                    num_bins = __NUM_BINS,
                    dim_reg_hide_sizes = [512],
                    bin_conf_hide_sizes = [256],
                    bin_reg_hide_sizes = [256])
)

# loss settings
loss_cfg = dict(
    
    dimension_loss_cfg = dict(type = 'Dimension_Loss',
                              base_loss_cfg = dict(type = 'MSE'),
                              avg_dim = [1.61057209, 1.47745965, 3.52359498]),
    
    pose_loss_cfg = dict(type = 'Pose_Loss',
                         base_conf_cfg = dict(type = 'CrossEntropy'),
                         base_reg_cfg = dict(type = 'Cosine_Expansion', 
                                             normalize = True),
                         num_bins = __NUM_BINS,
                         bin_range_degree = 120),
    
    loss_weights = {'dim_reg': 1.0, 'bin_conf': 1.0, 'bin_reg': 1.0}
)

# training settings
training_cfg = dict(
    
    loader_cfg = dict(batch_size = 8, 
                      num_workers = 4,
                      pin_memory = True,
                      drop_last = True),
    
    optimizer_cfg = dict(type ='SGD',
                         lr = 1e-4, 
                         momentum = 0.9, 
                         dampening = 0, 
                         weight_decay = 1e-4, 
                         nesterov = False),
    
    total_epoch = 10
    
)

log_cfg = dict(
    log_dir = './run'
)