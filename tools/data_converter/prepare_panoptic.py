from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.utils.data_classes import LidarPointCloud

from nuscenes.utils.data_io import load_bin_file,panoptic_to_lidarseg

import argparse
import os
import os.path as osp
import numpy as np
import time
from nuscenes.utils.map_mask import MapMask
from nuscenes.utils.color_map import get_colormap

def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--dataroot',
        type=str,
        default='./data/nuscenes/',
        help='specify the root path of dataset')
    parser.add_argument(
        '--save_path',
        type=str,
        default='./data/nuscenes/pc_panoptic/',
        required=False,
        help='specify the target path of dataset')
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0-trainval',
        required=False,
        help='')
    parser.add_argument(
        '--gt_from',
        type=str,
        default='panoptic',
        required=False,
        help='')

    args = parser.parse_args()
    return args

class NuScenesNew(NuScenes):
    """
    Database class for nuScenes to help query and retrieve information from the database.
    """

    def __init__(self,
                 version: str = 'v1.0-mini',
                 dataroot: str = '/data/sets/nuscenes',
                 verbose: bool = True,
                 map_resolution: float = 0.1):
        """
        Loads database and creates reverse indexes and shortcuts.
        :param version: Version to load (e.g. "v1.0", ...).
        :param dataroot: Path to the tables and data.
        :param verbose: Whether to print status messages during load.
        :param map_resolution: Resolution of maps (meters).
        """
        self.version = version
        self.dataroot = dataroot
        self.verbose = verbose
        self.table_names = ['category', 'attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor',
                            'ego_pose', 'log', 'scene', 'sample', 'sample_data', 'sample_annotation', 'map']
        # self.table_names = ['attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor',
        #                     'ego_pose', 'log', 'scene', 'sample', 'sample_data', 'sample_annotation', 'map']

        assert osp.exists(self.table_root), 'Database version not found: {}'.format(self.table_root)

        start_time = time.time()
        if verbose:
            print("======\nLoading NuScenes tables for version {}...".format(self.version))

        # Explicitly assign tables to help the IDE determine valid class members.
        self.panoptic = self.__load_table__('panoptic')
        self.category = self.__load_table__('category')
        self.attribute = self.__load_table__('attribute')
        self.visibility = self.__load_table__('visibility')
        self.instance = self.__load_table__('instance')
        self.sensor = self.__load_table__('sensor')
        self.calibrated_sensor = self.__load_table__('calibrated_sensor')
        self.ego_pose = self.__load_table__('ego_pose')
        self.log = self.__load_table__('log')
        self.scene = self.__load_table__('scene')
        self.sample = self.__load_table__('sample')
        self.sample_data = self.__load_table__('sample_data')
        self.sample_annotation = self.__load_table__('sample_annotation')
        self.map = self.__load_table__('map')

        # Initialize the colormap which maps from class names to RGB values.
        self.colormap = get_colormap()

        lidar_tasks = [t for t in ['lidarseg', 'panoptic'] if osp.exists(osp.join(self.table_root, t + '.json'))]
        if len(lidar_tasks) > 0:
            self.lidarseg_idx2name_mapping = dict()
            self.lidarseg_name2idx_mapping = dict()
            self.load_lidarseg_cat_name_mapping()
        for i, lidar_task in enumerate(lidar_tasks):
            if self.verbose:
                print(f'Loading nuScenes-{lidar_task}...')
            if lidar_task == 'lidarseg':
                self.lidarseg = self.__load_table__(lidar_task)
            else:
                self.panoptic = self.__load_table__(lidar_task)

            setattr(self, lidar_task, self.__load_table__(lidar_task))
            label_files = os.listdir(os.path.join(self.dataroot, lidar_task, self.version))
            num_label_files = len([name for name in label_files if (name.endswith('.bin') or name.endswith('.npz'))])
            num_lidarseg_recs = len(getattr(self, lidar_task))
            assert num_lidarseg_recs == num_label_files, \
                f'Error: there are {num_label_files} label files but {num_lidarseg_recs} {lidar_task} records.'
            self.table_names.append(lidar_task)
            # Sort the colormap to ensure that it is ordered according to the indices in self.category.
            self.colormap = dict({c['name']: self.colormap[c['name']]
                                  for c in sorted(self.category, key=lambda k: k['index'])})

        # If available, also load the image_annotations table created by export_2d_annotations_as_json().
        if osp.exists(osp.join(self.table_root, 'image_annotations.json')):
            self.image_annotations = self.__load_table__('image_annotations')

        # Initialize map mask for each map record.
        for map_record in self.map:
            map_record['mask'] = MapMask(osp.join(self.dataroot, map_record['filename']), resolution=map_resolution)

        if verbose:
            for table in self.table_names:
                print("{} {},".format(len(getattr(self, table)), table))
            print("Done loading in {:.3f} seconds.\n======".format(time.time() - start_time))

        # Make reverse indexes for common lookups.
        self.__make_reverse_index__(verbose)

        # Initialize NuScenesExplorer class.
        self.explorer = NuScenesExplorer(self)

def generate_occupancy_data(nusc: NuScenes, cur_sample, save_path='./occupacy/', gt_from: str = 'lidarseg'):

    lidar_data = nusc.get('sample_data',
                            cur_sample['data']['LIDAR_TOP'])
    pc = LidarPointCloud.from_file(nusc.dataroot+lidar_data['filename'])
    filename = os.path.split(lidar_data['filename'])[-1]
    lidar_sd_token = cur_sample['data']['LIDAR_TOP']

    lidarseg_labels_filename = os.path.join(nusc.dataroot,
                                            nusc.get(gt_from, lidar_sd_token)['filename'])
    lidar_seg = load_bin_file(lidarseg_labels_filename, type=gt_from)
    if gt_from == 'panoptic':
        lidar_seg = panoptic_to_lidarseg(lidar_seg)
    lidar_seg = lidar_seg.reshape(-1, len(lidar_seg))

    new_points = np.concatenate((pc.points, lidar_seg), axis=0)
    new_points = new_points.astype(np.float16)
    new_points.tofile(save_path + filename)


def convert2occupy(dataroot = '../nuscenes3/', save_path='./occupancy/', version='v1.0-trainval', gt_from='panoptic'):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    nusc = NuScenesNew(version=version, dataroot=dataroot, verbose=True)
    for i, scene in enumerate(nusc.scene):
        print("scene {}".format(i))

        sample_token = scene['first_sample_token']
        cur_sample = nusc.get('sample', sample_token)
        cnt = 0
        while True:
            generate_occupancy_data(nusc, cur_sample, save_path=save_path, gt_from=gt_from)
            cnt += 1
            print(cnt)
            if cur_sample['next'] == '':
                break
            cur_sample = nusc.get('sample', cur_sample['next'])
        print("scene {}: {} samples finished.".format(i, cnt))

if __name__ == "__main__":

    args = parse_args()
    convert2occupy(args.dataroot, args.save_path, version=args.version, gt_from=args.gt_from)

    # dataroot = 'D:\Jenny\Code\\v1.0-mini\\'
    # save_path = 'D:\Jenny\Code\\v1.0-mini\pc_panoptic\\'
    # convert2occupy(dataroot, save_path, version='v1.0-mini', gt_from='panoptic')
