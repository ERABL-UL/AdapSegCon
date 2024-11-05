# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 12:16:53 2024

@author: NanBlanc
"""

import OSToolBox as ost
import numpy as np
from tqdm import tqdm

in_cloud='/home/reza/PHD/Data/Parislille3D/orig_others/test/labels/fine_tune/ajaccio_57_label_100.ply'
out_cloud='/home/reza/PHD/Data/Parislille3D/orig_others/test/labels/fine_tune/ajaccio_57_label_100_color.ply'
colors=np.genfromtxt('/home/reza/PHD/Data/Parislille3D/orig_others/test/labels/fine_tune/colors.txt',delimiter=",")


#do a for loop if needed
data=ost.read_ply(in_cloud)
cloud_x = data['x']
cloud_y = data['y']
cloud_z = data['z']
# UTM_OFFSET = [627285, 4841948, 0]
# cloud_x = cloud_x - UTM_OFFSET[0]
# cloud_y = cloud_y - UTM_OFFSET[1]
# cloud_z = cloud_z - UTM_OFFSET[2]
labels = (data['c']).astype(np.int32)
data = np.c_[cloud_x, cloud_y, cloud_z, labels]

newdata=np.zeros((data.shape[0],7))
for i,point in tqdm(enumerate(data)):
    newdata[i]=np.insert(point,3,colors[int(point[3])])
    
ost.write_ply(out_cloud, [newdata[:,:3],newdata[:,3:].astype(np.int32)], ["x","y","z","red","green","blue","class"])