# global
__NUM_BINS = 4

# dataset settings
dataset_cfg = dict(
    nusc_root = './data/nusc',
    img_norm = {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
    del_labels = ('attribute_tokens', 'bbox_corners', 'box_image_file', 'category_name', \
                  'class', 'filename', 'instance_token', 'location', 'next', \
                  'num_lidar_pts', 'num_radar_pts', 'prev', 'sample_annotation_token', \
                  'sample_data_token', 'type', 'visibility_token')
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
                              avg_dim = (1.80496203, 3.55914433, 1.77130574)),
    
    pose_loss_cfg = dict(type = 'Pose_Loss',
                         base_conf_cfg = dict(type = 'CrossEntropy'),
                         base_reg_cfg = dict(type = 'Smooth_L1'),
                         num_bins = __NUM_BINS,
                         bin_range_degree = 100.0),
    
    loss_weights = {'dim_reg': 0.5, 'bin_conf': 1.0, 'bin_reg': 3.0}
)

# training settings
training_cfg = dict(
    
    loader_cfg = dict(batch_size = 192, 
                      num_workers = 24,
                      pin_memory = True,
                      drop_last = True),
    
    optimizer_cfg = dict(type ='SGD',
                         lr = 1e-4, 
                         momentum = 0.9, 
                         dampening = 0, 
                         weight_decay = 0, 
                         nesterov = False),
    
    total_epoch = 10
    
)

log_cfg = dict(
    log_dir = './run',
    log_loss_every = 10, # unit: iteration
    show_loss_every = 100, # unit: iteration
    ckpt_every = 1, # unit: epoch
    eval_every = 1 # unit: epoch
)