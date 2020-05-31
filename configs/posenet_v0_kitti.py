# global
__NUM_BINS = 4

# dataset settings
dataset_cfg = dict(
    kitti_root = './data/kitti',
    img_norm = {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
    del_labels = ('sample', 'type', 'class', 'bbox2D', 'location')
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
                    bin_reg_hide_sizes = [256],
                    cos_sin_encode = False,
                    init_weights = True)
)

# loss settings
loss_cfg = dict(
    
    dimension_loss_cfg = dict(type = 'Dimension_Loss',
                              base_loss_cfg = dict(type = 'Smooth_L1'),
                              avg_dim = (1.61057209, 1.47745965, 3.52359498)),
    
    pose_loss_cfg = dict(type = 'Pose_Loss',
                         base_conf_cfg = dict(type = 'CrossEntropy'),
                         base_reg_cfg = dict(type = 'Smooth_L1'),
                         num_bins = __NUM_BINS,
                         bin_range_degree = 100.0),
    
    loss_weights = {'dim_reg': 5.0, 'bin_conf': 1.0, 'bin_reg': 3.0}
)

# training settings
training_cfg = dict(
    
    loader_cfg = dict(batch_size = 128, 
                      num_workers = 32,
                      pin_memory = True,
                      drop_last = True),
    
    optimizer_cfg = dict(type ='SGD',
                         lr = 1.6e-3, 
                         momentum = 0.9, 
                         dampening = 0, 
                         weight_decay = 1e-4, 
                         nesterov = False),
    
    total_epoch = 50,
    lr_decay_epochs = [25],
    lr_decay_rates = [0.1]
    
)

log_cfg = dict(
    log_dir = './run',
    log_loss_every = 10, # unit: iteration
    show_loss_every = 20, # unit: iteration
    ckpt_every = 1, # unit: epoch
    eval_every = 1 # unit: epoch
)