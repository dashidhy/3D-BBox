if __name__ == '__main__':
    import os
    import argparse
    from functools import partial
    from PIL import Image
    from tqdm import tqdm
    import kitti_utils as ku
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='KITTI dataset root.')
    parser.add_argument('--force', action='store_true', help='To cover existing box dataset.')
    FLAGS = parser.parse_args()

    data_dir = os.path.join(FLAGS.root, 'training')
    splits_dir = os.path.join(FLAGS.root, 'splits')
    boxes_dir = os.path.join(FLAGS.root, 'boxes')

    mkdirs = partial(os.makedirs, exist_ok=FLAGS.force)

    if os.path.exists(boxes_dir) and not FLAGS.force:
        raise FileExistsError('Box dataset exists! Use --force to cover it.')

    # make train val set
    for split in ['train', 'val']:
        image_dir = os.path.join(boxes_dir, split, 'image')
        label_dir = os.path.join(boxes_dir, split, 'label')
        mkdirs(image_dir)
        mkdirs(label_dir)
    
        # start creating
        sample_list = ku.read_txt_lines(os.path.join(splits_dir, split+'.txt'))
        print('Creating {} box dataset...'.format(split))
        box_id = 0
        for sample in tqdm(sample_list):
            scene_image = Image.open(os.path.join(data_dir, 'image_2', sample+'.png'))
            scene_labels = ku.read_scene_labels(os.path.join(data_dir, 'label_2', sample+'.txt'))

            for scene_label in scene_labels:
                if scene_label['class'] >= 0:
                    box_image = scene_image.resize((224, 224), box=scene_label['bbox2D'])
                    box_label = ku.label_scene2box(scene_label)
                    ku.write_box_image(box_image,
                                       os.path.join(image_dir, '%08d.png'% box_id),
                                       force=FLAGS.force)
                    ku.write_box_label(box_label, 
                                       os.path.join(label_dir, '%08d.txt'% box_id), 
                                       force=FLAGS.force)
                    box_id += 1