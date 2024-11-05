import numpy as np
import warnings
import os
from torch.utils.data import Dataset
from data_utils import data_map_KITTI360, data_map_ParisLille3D, data_map_Toronto3D
from pcd_utils.pcd_preprocess import *
from pcd_utils.pcd_transforms import *
import MinkowskiEngine as ME
import torch
import json
import OSToolBox as ost

warnings.filterwarnings('ignore')

class AggregatedPCDataLoader(Dataset):
    def __init__(self, root,  split='train', dataset_name="KITTI360", pre_training=True, resolution=0.05, percentage=None, orig=False):
        self.root = root
        self.augmented_dir = 'segmented_views'
        self.n_clusters = 50
        self.dataset_name = dataset_name
        if not os.path.isdir(os.path.join(self.root, self.augmented_dir)):
            os.makedirs(os.path.join(self.root, self.augmented_dir))
        self.resolution = resolution
        self.orig = orig
        self.pre_training = pre_training
        self.split = split
        self.seq_ids = {}
        if self.dataset_name == "KITTI360":
            self.seq_ids['train'] = [ '00', '02', '03', '04', '05', '06', '07', '09', '10']
            self.seq_ids['validation'] = [ '00', '02', '03', '04', '05', '06', '07', '09', '10' ]
            self.seq_ids['test'] = ['08', '18']
            self.learning_map = data_map_KITTI360.learning_map
            self.filename_split = 37
            self.class_name = 'semantic'
        elif self.dataset_name == "ParisLille3D":
            self.seq_ids['train'] = ['01', '02', '03', '04']
            self.seq_ids['validation'] = [ '03' ]
            self.seq_ids['test'] = ['05', '06', '07']
            self.learning_map = data_map_ParisLille3D.learning_map
            self.filename_split = 41
            self.class_name = 'c'
        elif self.dataset_name == "Toronto3D":
            self.seq_ids['train'] = [ '01', '03', '04']
            self.seq_ids['validation'] = [ '02' ]
            self.learning_map = data_map_Toronto3D.learning_map 
            self.filename_split = 38
            self.class_name = 'scalar_Label'
        elif self.dataset_name == "Merged":
            self.seq_ids['train'] = [ '00', '02', '03', '04', '05', '06', '07', '09', '10',
                                     '11', '12', '13', '14', '15', '16']
        
        assert (split == 'train' or split == 'validation' or split =='test')
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
        if self.orig:
            json_filename = 'tools/percentiles_split_' + self.dataset_name + '_orig' + '.json'
        else:
            json_filename = 'tools/percentiles_split_' + self.dataset_name + '.json'
        with open(json_filename, 'r') as p:
            splits = json.load(p)

            assert (percentage in splits)

            self.points_datapath = []

            for seq in splits[percentage]:
                self.points_datapath += splits[percentage][seq]['data']

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
    
    def read_ply_cluster(self, file_name):
        data = ost.read_ply(file_name)
        cloud_x = data['x']
        cloud_y = data['y']
        cloud_z = data['z']
        cluster = (data['seg']).astype(np.int32)
        
        return(np.c_[cloud_x, cloud_y, cloud_z, cluster])
    
    def read_ply(self, file_name):
        data = ost.read_ply(file_name)
        cloud_x = data['x']
        cloud_y = data['y']
        cloud_z = data['z']
        if self.orig:
            labels = (data[self.class_name]).astype(np.int32)
        else:
            labels = (data['c']).astype(np.int32)
        if self.dataset_name == "Toronto3D" and self.orig:
            UTM_OFFSET = [627285, 4841948, 0]
            cloud_x = cloud_x - UTM_OFFSET[0]
            cloud_y = cloud_y - UTM_OFFSET[1]
            cloud_z = cloud_z - UTM_OFFSET[2]
        labels = labels.reshape(len(labels), 1)
        return(np.c_[cloud_x, cloud_y, cloud_z], labels)
    
    def read_ply_unlabel(self, file_name):
        data = ost.read_ply(file_name)
        cloud_x = data['x']
        cloud_y = data['y']
        cloud_z = data['z']
        return np.c_[cloud_x, cloud_y, cloud_z]
    def _get_augmented_item(self, index):
        # we need to preprocess the data to get the cuboids and guarantee overlapping points
        # so if we never have done this we do and save this preprocessing
        cluster_path = os.path.join(self.root, self.augmented_dir, f'{index}.ply')
        points_set = self.read_ply_cluster(cluster_path)
        points_i = random_cuboid_point_cloud(points_set.copy())
        points_i = self.transforms(points_i)
        points_j = random_cuboid_point_cloud(points_set.copy())
        points_j = self.transforms(points_j)
        
        return points_i, points_j

    def _get_item(self, index):
        if self.split != 'test':
            # print(self.points_datapath[index])
            points_set, labels = self.read_ply(self.points_datapath[index])
            labels = np.vectorize(self.learning_map.get)(labels)
            # labels = np.expand_dims(labels, axis=-1)
            unlabeled = labels[:,0] == 0

            # remove unlabeled points
            labels = np.delete(labels, unlabeled, axis=0)
            points_set = np.delete(points_set, unlabeled, axis=0)
            #remap labels to learning values
            points_set[:, :3] = self.transforms(points_set[:, :3])
    
            return points_set, labels.astype(np.int32)
        else:
            points_set = self.read_ply_unlabel(self.points_datapath[index])
            return points_set

    def __getitem__(self, index):
        return self._get_augmented_item(index) if self.pre_training else self._get_item(index)
