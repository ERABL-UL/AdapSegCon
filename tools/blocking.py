import numpy as np
import OSToolBox as ost

def read_ply(file_name):
    data = ost.read_ply(file_name)
    cloud_x = data['x']
    cloud_y = data['y']
    cloud_z = data['z']
    # label = (data['class']).astype(np.int32)
    # UTM_OFFSET = [627285, 4841948, 0]
    # cloud_x = cloud_x - UTM_OFFSET[0]
    # cloud_y = cloud_y - UTM_OFFSET[1]
    # cloud_z = cloud_z - UTM_OFFSET[2]
    return(np.c_[cloud_x, cloud_y, cloud_z])


def partition_point_cloud(point_cloud, segment_distance):
    # Extract x, y coordinates from the point cloud
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]

    # Calculate the cumulative distance between consecutive points
    cumulative_distance = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    cumulative_distance = np.insert(cumulative_distance, 0, 0)  # Include the starting point

    # Determine the segment indices based on the desired segment distance
    segment_indices = np.searchsorted(cumulative_distance, np.arange(0, cumulative_distance[-1], segment_distance))

    # Partition the point cloud into segments
    segmented_point_clouds = np.split(point_cloud, segment_indices[1:], axis=0)

    return segmented_point_clouds

# Example usage:
# Assuming 'point_cloud' is your input point cloud with shape (N, 3) where N is the number of points
# Each row represents (x, y, z) coordinates of a point

# Replace this with your actual point cloud data

file_name = '/home/reza/PHD/Data/Parislille3D/orig_others/test/sequences/07/dijon_9.ply'
point_cloud = read_ply(file_name)

# Set the desired segment distance (200 meters in this case)
segment_distance = 5000000.0  

# Partition the point cloud into segments
segmented_point_clouds = partition_point_cloud(point_cloud, segment_distance)

# # Print the number of segments and their lengths
# print(f"Number of segments: {len(segmented_point_clouds)}")
# for i, segment in enumerate(segmented_point_clouds):
#     print(f"Segment {i + 1} length: {np.sqrt(np.sum(np.diff(segment[:, 0])**2 + np.diff(segment[:, 1])**2))):.2f} meters")
for i, segment in enumerate(segmented_point_clouds):
    ost.write_ply('/home/reza/PHD/Data/Parislille3D/orig_others/test/sequences/07seg/'+'seg'+str(i)+'.ply', segment, ['x','y','z'] )