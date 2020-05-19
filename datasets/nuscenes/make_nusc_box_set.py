# modified from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/scripts/export_2d_annotations_as_json.py by Hongyuan Du, 2020.

"""
Export 2D annotations (xmin, ymin, xmax, ymax) from re-projections of our annotated 3D bounding boxes to a .json file.
Note: Projecting tight 3d boxes to 2d generally leads to non-tight boxes.
      Furthermore it is non-trivial to determine whether a box falls into the image, rather than behind or around it.
      Finally some of the objects may be occluded by other objects, in particular when the lidar can see them, but the
      cameras cannot.
"""

import argparse
import json
import os
from collections import OrderedDict
from typing import List, Tuple, Union

from PIL import Image
import numpy as np
from pyquaternion.quaternion import Quaternion
from shapely.geometry import MultiPoint, box
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import train_detect as TRAIN_SCENES, val as VAL_SCENES

category2class = {
    'movable_object.barrier': ('barrier', 0),
    'vehicle.bicycle': ('bicycle', 1),
    'vehicle.bus.bendy': ('bus', 2),
    'vehicle.bus.rigid': ('bus', 2),
    'vehicle.car': ('car', 3),
    'vehicle.construction': ('construction_vehicle', 4),
    'vehicle.motorcycle': ('motorcycle', 5),
    'human.pedestrian.adult': ('pedestrian', 6),
    'human.pedestrian.child': ('pedestrian', 6),
    'human.pedestrian.construction_worker': ('pedestrian', 6),
    'human.pedestrian.police_officer': ('pedestrian', 6),
    'movable_object.trafficcone': ('traffic_cone', 7),
    'vehicle.trailer': ('trailer', 8),
    'vehicle.truck': ('truck', 9)
}


def post_process_coords(corner_coords: List,
                        imsize: Tuple[int, int] = (1600, 900)) -> Union[Tuple[float, float, float, float], None]:
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict,
                    x1: float,
                    y1: float,
                    x2: float,
                    y2: float,
                    sample_data_token: str,
                    filename: str) -> OrderedDict:
    """
    Generate one 2D annotation record given various informations on top of the 2D bounding box coordinates.
    :param ann_rec: Original 3d annotation record.
    :param x1: Minimum value of the x coordinate.
    :param y1: Minimum value of the y coordinate.
    :param x2: Maximum value of the x coordinate.
    :param y2: Maximum value of the y coordinate.
    :param sample_data_token: Sample data token.
    :param filename:The corresponding image file where the annotation is present.
    :return: A sample 2D annotation record.
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    # add more fields
    repro_rec['type'] = category2class[ann_rec['category_name']][0]
    repro_rec['class'] = category2class[ann_rec['category_name']][1]

    return repro_rec


def get_2d_boxes(sample_data_token: str, visibilities: List[str], dataroot: str, box_image_dir: str) -> List[OrderedDict]:
    """
    Get the 2D annotation records for a given `sample_data_token`.
    :param sample_data_token: Sample data token belonging to a camera keyframe.
    :param visibilities: Visibility filter.
    :return: List of 2D annotation record that belongs to the input `sample_data_token`
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)
    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties and in detection benchmark
    ann_recs = [nusc.get('sample_annotation', token) for token in s_rec['anns']]
    ann_recs = [ann_rec for ann_rec in ann_recs if (ann_rec['visibility_token'] in visibilities and \
                                                    ann_rec['category_name'] in category2class.keys())]

    repro_recs = []
    scene_image = Image.open(os.path.join(dataroot, sd_rec['filename']))
    for i, ann_rec in enumerate(ann_recs):
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords
            if max_x - min_x < 25.0 or max_y - min_y < 25.0:
                continue

        # TODO: add dimension, location, theta_l

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y, sample_data_token, sd_rec['filename'])
        
        # save box image
        box_image = scene_image.resize((224, 224), box=final_coords)
        box_image_file = os.path.join(box_image_dir, 'camera__{}__{}.png'.format(sample_data_token, i))
        box_image.save(box_image_file)
        repro_rec['box_image_file'] = box_image_file
        
        repro_recs.append(repro_rec)

    return repro_recs


def main(args):

    # create folders
    boxes_dir = os.path.join(args.dataroot, 'boxes')
    if os.path.exists(boxes_dir):
        raise FileExistsError('Box dataset exists!')
    train_dir = os.path.join(boxes_dir, 'train')
    val_dir = os.path.join(boxes_dir, 'val')
    os.makedirs(train_dir)
    os.makedirs(val_dir)

    def create_subset(split_dir, scenes):
        box_image_dir = os.path.join(split_dir, 'images')
        os.makedirs(box_image_dir)
        sample_data_camera_tokens = []

        print('Loading camera tokens...')
        for scene_info in nusc.scene:
            if scene_info['name'] in scenes:
                sample_token = scene_info['first_sample_token']
                while sample_token != '':
                    sample_info = nusc.get('sample', sample_token)
                    for sensor_token in sample_info['data'].values():
                        sensor_info = nusc.get('sample_data', sensor_token)
                        if sensor_info['sensor_modality'] == 'camera' and sensor_info['is_key_frame']:
                            sample_data_camera_tokens.append(sensor_token)
                    sample_token = sample_info['next']
        print('Done. {} tokens loaded.'.format(len(sample_data_camera_tokens)))

        # For debugging purposes: Only produce the first n images.
        if args.image_limit != -1:
            sample_data_camera_tokens = sample_data_camera_tokens[:args.image_limit]
            print('Debugging. {} image used.'.format(args.image_limit))

        # Loop through the records and apply the re-projection algorithm.
        reprojections = []
        print('Annotation converting...')
        for token in tqdm(sample_data_camera_tokens):
            reprojection_records = get_2d_boxes(token, args.visibilities, args.dataroot, box_image_dir)
            reprojections.extend(reprojection_records)

        # Save to a .json file.
        with open(os.path.join(split_dir, 'image_annotations.json'), 'w') as fh:
            json.dump(reprojections, fh, sort_keys=True, indent=4)

        print("Done. Saved image_annotations.json under {}".format(split_dir))
    
    print('Creating train set...')
    create_subset(train_dir, TRAIN_SCENES)
    print('Creating val set...')
    create_subset(val_dir, VAL_SCENES)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export 2D annotations from reprojections to a .json file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, default='./data/nuscenes', help="Path where nuScenes is saved.")
    parser.add_argument('--visibilities', type=str, default=['', '1', '2', '3', '4'],
                        help='Visibility bins, the higher the number the higher the visibility.', nargs='+')
    parser.add_argument('--image_limit', type=int, default=-1, help='Number of images to process or -1 to process all.')
    args = parser.parse_args()
    
    nusc = NuScenes(dataroot=args.dataroot, version='v1.0-trainval')
    main(args)