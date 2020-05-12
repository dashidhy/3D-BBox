import torch
import numpy as np


def dimensions_to_corners(dimensions):
    """
    Note that the center of the 3D bounding box is the center of the bottom surface,
    not the geometric center of the box, following KITTI's defination.
    """
    h, w, l = dimensions[:, 0], dimensions[:, 1], dimensions[:, 2]
    zeros = torch.zeros(dimensions.size(0)).to(dimensions.device)
    corner_x = torch.stack([ l/2.0, -l/2.0, -l/2.0,  l/2.0,  l/2.0, -l/2.0, -l/2.0,  l/2.0])
    corner_y = torch.stack([ zeros,  zeros,  zeros,  zeros,     -h,     -h,     -h,     -h])
    corner_z = torch.stack([ w/2.0,  w/2.0, -w/2.0, -w/2.0,  w/2.0,  w/2.0, -w/2.0, -w/2.0])
    corners = torch.stack([corner_x, corner_y, corner_z]) # 3x8xN

    return corners.permute(2, 1, 0).contiguous() # Nx8x3


def solve_3d_bbox_single(bbox2D, corners, theta_l, calib):
    """
    Input:
        bbox2D: Tensor(4), [x1, y1, x2, y2]
        corners: Tensor(8, 3), aligned corners without rotation
        theta_l: camera direction [-pi, pi]
        calib: calibration metrices in KITTI
    """

    x1, y1, x2, y2 = bbox2D

    # useful calibrations
    P2 = calib['P2']
    R0_rect = torch.eye(4)
    R0_rect[:3, :3] = calib['R0_rect']
    K = torch.matmul(P2, R0_rect)

    # use 2D bbox to estimate global rotation
    theta_ray = torch.atan2(P2[0, 0], (x1 + x2) * 0.5 - P2[0, 2])
    ry = np.pi - theta_ray - theta_l

    Ry_T = torch.tensor([[ torch.cos(ry), 0.0, -torch.sin(ry)],
                         [     0.0      , 1.0,       0.0     ],
                         [ torch.sin(ry), 0.0,  torch.cos(ry)]])

    corners = torch.matmul(corners, Ry_T) # rotated corners

    # adjust front side
    if theta_l >= np.pi / 2.0 and theta_l < np.pi:
        corners = corners[[3, 0, 1, 2, 7, 4, 5, 6]].contiguous()
    elif theta_l >= -np.pi and theta_l < -np.pi / 2.0:
        corners = corners[[2, 3, 0, 1, 6, 7, 4, 5]].contiguous()
    elif theta_l >= -np.pi / 2.0 and theta_l < 0.0:
        corners = corners[[1, 2, 3, 0, 5, 6, 7, 4]].contiguous()

    # start solve constrains
    X = torch.eye(4)
    A = torch.zeros(4, 3)
    b = torch.zeros(4)

    # prepare constrains
    constrains = {}

    # x1 -> 7, 6
    constrains['x1'] = {}

    for i in [7, 6]:
        constrains['x1'][i] = {}
        X[:3, 3] = corners[i]
        K_X = torch.matmul(K, X)
        constrains['x1'][i]['A'] = K_X[0, :3] - x1 * K_X[2, :3]
        constrains['x1'][i]['b'] = x1 * K_X[2, 3] - K_X[0, 3]
    
    # x2 -> 4, 7
    constrains['x2'] = {}

    for i in [4, 7]:
        constrains['x2'][i] = {}
        X[:3, 3] = corners[i]
        K_X = torch.matmul(K, X)
        constrains['x2'][i]['A'] = K_X[0, :3] - x2 * K_X[2, :3]
        constrains['x2'][i]['b'] = x2 * K_X[2, 3] - K_X[0, 3]
    
    # y1 -> 4, 5, 6, 7
    constrains['y1'] = {}

    for i in [4, 5, 6, 7]:
        constrains['y1'][i] = {}
        X[:3, 3] = corners[i]
        K_X = torch.matmul(K, X)
        constrains['y1'][i]['A'] = K_X[1, :3] - y1 * K_X[2, :3]
        constrains['y1'][i]['b'] = y1 * K_X[2, 3] - K_X[1, 3]
    
    # y2 -> 2, 3, 0
    constrains['y2'] = {}

    for i in [2, 3, 0]:
        constrains['y2'][i] = {}
        X[:3, 3] = corners[i]
        K_X = torch.matmul(K, X)
        constrains['y2'][i]['A'] = K_X[1, :3] - y2 * K_X[2, :3]
        constrains['y2'][i]['b'] = y2 * K_X[2, 3] - K_X[1, 3]

    # solving linear functions
    error = float('inf')

    # case 1: only see front side
    A[0] = constrains['x1'][7]['A']
    b[0] = constrains['x1'][7]['b']
    A[1] = constrains['x2'][4]['A']
    b[1] = constrains['x2'][4]['b']

    for i in [3, 0]:
        A[2] = constrains['y2'][i]['A']
        b[2] = constrains['y2'][i]['b']
        for j in [4, 5, 6, 7]:
            A[3] = constrains['y1'][j]['A']
            b[3] = constrains['y1'][j]['b']

            trans_t = torch.matmul(torch.pinverse(A), b)
            error_t = torch.norm(torch.matmul(A, trans_t) - b)

            if error_t < error:
                trans = trans_t
                error = error_t
    
    # case 2: see both front side and lateral side
    A[0] = constrains['x1'][6]['A']
    b[0] = constrains['x1'][6]['b']

    for i in [2, 3, 0]:
        A[2] = constrains['y2'][i]['A']
        b[2] = constrains['y2'][i]['b']
        for j in [4, 5, 6, 7]:
            A[3] = constrains['y1'][j]['A']
            b[3] = constrains['y1'][j]['b']

            trans_t = torch.matmul(torch.pinverse(A), b)
            error_t = torch.norm(torch.matmul(A, trans_t) - b)

            if error_t < error:
                trans = trans_t
                error = error_t
    
    # case 2: only see lateral side
    A[1] = constrains['x2'][7]['A']
    b[1] = constrains['x2'][7]['b']

    for i in [2, 3]:
        A[2] = constrains['y2'][i]['A']
        b[2] = constrains['y2'][i]['b']
        for j in [4, 5, 6, 7]:
            A[3] = constrains['y1'][j]['A']
            b[3] = constrains['y1'][j]['b']

            trans_t = torch.matmul(torch.pinverse(A), b)
            error_t = torch.norm(torch.matmul(A, trans_t) - b)

            if error_t < error:
                trans = trans_t
                error = error_t
    
    return trans


# debug
if __name__ == '__main__':
    import os
    from datasets.kitti import kitti_utils as ku
    kitti_root = '/home/srip19-pointcloud/datasets/KITTI/'
    box_root = os.path.join(kitti_root, 'boxes', 'train')
    scene_root = os.path.join(kitti_root, 'training')
    
    box_id = '%08d' % 321
    box_label = ku.read_box_label(os.path.join(box_root, 'label', box_id+'.txt'))
    scene_id = box_label['sample']
    calib = ku.read_calib(os.path.join(scene_root, 'calib', scene_id+'.txt'))

    bbox2D = torch.tensor(box_label['bbox2D'])
    corners = dimensions_to_corners(torch.tensor(box_label['dimensions']).unsqueeze(0)).squeeze()
    theta_l = torch.tensor(box_label['theta_l'])

    print(solve_3d_bbox_single(bbox2D, corners, theta_l, calib))
    print(box_label['location'])