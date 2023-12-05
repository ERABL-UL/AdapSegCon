import numpy as np
import warnings
import os
from torch.utils.data import Dataset
from data_utils.data_map import *
from pcd_utils.pcd_preprocess import *
from pcd_utils.pcd_transforms import *
import MinkowskiEngine as ME
import torch
import json
import OSToolBox as ost

warnings.filterwarnings('ignore')

class SemanticKITTIDataLoader(Dataset):
    def __init__(self, root,  split='train', pre_training=True, resolution=0.05, percentage=None, intensity_channel=False):
        self.root = root
        self.augmented_dir = 'segmented_views'
        self.n_clusters = 50

        if not os.path.isdir(os.path.join(self.root, self.augmented_dir)):
            os.makedirs(os.path.join(self.root, self.augmented_dir))
        self.resolution = resolution
        self.intensity_channel = intensity_channel

        self.seq_ids = {}
        self.seq_ids['train'] = [ '00', '02', '03', '04', '05', '06', '07', '09', '10']
        self.seq_ids['validation'] = ['00', '02', '03', '04', '05', '06', '07', '09', '10']
        self.pre_training = pre_training
        self.split = split

        assert (split == 'train' or split == 'validation')
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath_list(split)

        if split == 'train':
            self.train_set_percent(percentage)

        print('The size of %s data is %d'%(split,len(self.points_datapath)))

    def train_set_percent(self, percentage):
        if percentage is None or percentage == 1.0:
            return

        percentage = str(percentage)
        # the stratified point clouds are pre-defined on this percentiles_split.json file
        with open('tools/percentiles_split.json', 'r') as p:
            splits = json.load(p)

            assert (percentage in splits)

            self.points_datapath = []

            for seq in splits[percentage]:
                self.points_datapath += splits[percentage][seq]['points']

        return

    def datapath_list(self, split):
        self.points_datapath = []

        for seq in self.seq_ids[split]:
            point_seq_path = os.path.join(self.root, split, 'sequences', seq)
            point_seq_bin = os.listdir(point_seq_path)
            point_seq_bin.sort()
            self.points_datapath += [ os.path.join(point_seq_path, point_file) for point_file in point_seq_bin ]


    def transforms(self, points):
        if self.pre_training:
            points = np.expand_dims(points, axis=0)
            points[:,:,:3] = rotate_point_cloud(points[:,:,:3])
            points[:,:,:3] = rotate_perturbation_point_cloud(points[:,:,:3])
            points[:,:,:3] = random_scale_point_cloud(points[:,:,:3])
            points[:,:,:3] = random_flip_point_cloud(points[:,:,:3])
            points[:,:,:3] = jitter_point_cloud(points[:,:,:3])
            points = random_drop_n_cuboids(points)
            return np.squeeze(points, axis=0)
        
        elif self.split == 'train':
            theta = torch.FloatTensor(1,1).uniform_(0, 2*np.pi).item()
            scale_factor = torch.FloatTensor(1,1).uniform_(0.95, 1.05).item()
            rot_mat = np.array([[np.cos(theta),
                                    -np.sin(theta), 0],
                                [np.sin(theta),
                                    np.cos(theta), 0], [0, 0, 1]])

            points[:, :3] = np.dot(points[:, :3], rot_mat) * scale_factor
            return points
        else:
            return points


    def __len__(self):
        return len(self.points_datapath)
    
    def read_ply(self, file_name):
        data = ost.read_ply(file_name)
        cloud_x = data['x']
        cloud_y = data['y']
        cloud_z = data['z']
        if self.pre_training == True:
            labels = (data['seg']).astype(np.int32)
        else:
            labels = (data['c']).astype(np.int32)
        labels = labels.reshape(len(labels), 1)
        return(np.c_[cloud_x,cloud_y,cloud_z,labels])
    
    def _get_augmented_item(self, index):
        cluster_path = os.path.join(self.root, self.augmented_dir, f'{index}.ply')
        points_set = self.read_ply(cluster_path)
        #generating 2 views
        points_i = random_cuboid_point_cloud(points_set.copy())
        points_i = self.transforms(points_i)
        points_j = random_cuboid_point_cloud(points_set.copy())
        points_j = self.transforms(points_j)

        return points_i, points_j

    def _get_item(self, index):
        pc = self.read_ply(self.points_datapath[index])
        points_set,label = pc[:,:3], pc[:,3:]
        labels = np.vectorize(learning_map.get)(label)
        unlabeled = labels[:,0] == 0
        # remove unlabeled points
        labels = np.delete(labels, unlabeled, axis=0)
        points_set = np.delete(points_set, unlabeled, axis=0)
        points_set[:, :3] = self.transforms(points_set[:, :3])

        return points_set, labels.astype(np.int32)

    def __getitem__(self, index):
        return self._get_augmented_item(index) if self.pre_training else self._get_item(index)
