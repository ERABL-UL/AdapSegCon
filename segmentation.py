from pcd_utils.pcd_preprocess import *
import numpy as np
import OSToolBox as ost
import argparse
from tqdm import tqdm
import os
# from data_utils.data_map_SemanticKITTI import learning_map as learning_map_semantickitti
from data_utils.data_map_Toronto3D import learning_map
def read_ply(file_name):
    data = ost.read_ply(file_name)
    cloud_x = data['x']
    cloud_y = data['y']
    cloud_z = data['z']
    labels = data['c']
    return np.c_[cloud_x,cloud_y,cloud_z], labels

def datapath_list(root, dataset, seq_ids, split):
    points_datapath = []
    labels_datapath = []
    if dataset == "SemanticKITTI":
        for seq in seq_ids:
            point_seq_path = os.path.join(root, split, 'sequences', "{0:0=2d}".format(seq), 'velodyne')
            label_seq_path = os.path.join(root, split, 'sequences', "{0:0=2d}".format(seq), 'labels')
            
            point_seq_bin = os.listdir(point_seq_path)
            point_seq_bin.sort()
            points_datapath += [ os.path.join(point_seq_path, point_file) for point_file in point_seq_bin ]
            
            point_seq_label = os.listdir(label_seq_path)
            point_seq_label.sort()
            labels_datapath += [ os.path.join(label_seq_path, label_file) for label_file in point_seq_label ]           
        return points_datapath, labels_datapath
    else:
        for seq in seq_ids:
            point_seq_path = os.path.join(root, split, 'sequences', "{0:0=2d}".format(seq))
            point_seq_ply = os.listdir(point_seq_path)
            point_seq_ply.sort()
            points_datapath += [ os.path.join(point_seq_path, point_file) for point_file in point_seq_ply ]
        return points_datapath           

def generate_segmented_pc(index, root, dataset, save_path, points_datapath, labels_datapath, n_clusters):
    # we need to preprocess the data to get the cuboids and guarantee overlapping points
    # so if we never have done this we do and save this preprocessing
    index_save = index
    cluster_path = os.path.join(root, save_path, f'{index_save}.ply')
    if dataset == "SemanticKITTI":
        points_set = np.fromfile(points_datapath[index], dtype=np.float32)
        points_set = points_set.reshape((-1, 4))
        # remove ground and get clusters from point cloud
        points_set = clusterize_pcd(points_set[:,:3], n_clusters)

        labels = np.fromfile(labels_datapath[index], dtype=np.uint32)
        labels = labels.reshape((-1))
        labels = labels & 0xFFFF
        #remap labels to learning values
        labels = np.vectorize(learning_map_semantickitti.get)(labels)
        labels = np.expand_dims(labels, axis=-1)

        ost.write_ply(cluster_path, [points_set[:,:3],np.int32(labels),points_set[:,3]], ['x','y','z','c','seg'])
    else:
        points_set, labels = read_ply(points_datapath[index])
        # labels = np.vectorize(learning_map.get)(label)
        points_set = clusterize_pcd(points_set, n_clusters)
        ost.write_ply(cluster_path, [points_set[:,:3],np.int32(labels),points_set[:,3]], ['x','y','z','c','seg'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SparseSimCLR')
    parser.add_argument('--dataset', type=str, default='Parislille3D' ,
                        help='Name_of_the_dataset')
    parser.add_argument('--split', type=str, default='train',
                        help='split set (default: train')
    parser.add_argument('--root', type=str, default='/home/reza/PHD/Data/Parislille3D/orig',
                        help='Path to the folder of dataset')
    parser.add_argument('--save-path', type=str, default='segmented_views',
                        help='Path to the folder that you want to save the generated files')
    parser.add_argument('--n-clusters', type=int, default=50 ,
                        help='size of each block')
    parser.add_argument('--seq-ids', type=list, default=[1,2,4] ,
                        help='list of sequences #numbers')
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.root, args.save_path)):
        os.makedirs(os.path.join(args.root, args.save_path))
    if args.dataset == "SemanticKITTI":
        points_datapath, labels_datapath = datapath_list(args.root, args.dataset, args.seq_ids, args.split)
    else:
        points_datapath = datapath_list(args.root, args.dataset, args.seq_ids, args.split)
        labels_datapath = []
    for index in tqdm(range(len(points_datapath))):
        generate_segmented_pc(index, args.root, args.dataset, args.save_path, points_datapath, labels_datapath, args.n_clusters)