# source: https://nbviewer.org/github/niosus/notebooks/blob/master/icp.ipynb
import numpy as np
import open3d as o3d
import copy
import matplotlib.pyplot as plt
from matplotlib import animation, rc


def returnFinites(array):
    arr_isfinite = np.all(np.isfinite(array), axis=1)
    return array[arr_isfinite, :]


def draw_registration_result(source, target, transformation=np.eye(4)):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def main():
    # saveFile = 'data_add/points_diff_scans_infV.npz'
    # saveFile = 'data/points_from_sim_ranges_inf.npz'
    saveFile = 'data_add/points_vdiff_scans.npz'

    # load the two laser scans. they are already in a 4x1 format
    diff_points_from_ranges = np.load(saveFile)
    dspoints_p1_4x1 = returnFinites(diff_points_from_ranges['points_p1'])
    dspoints_p2_4x1 = returnFinites(diff_points_from_ranges['points_p2'])

    # create a point cloud object for open3d and input your points in there
    source_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()

    source_pcd.points = o3d.utility.Vector3dVector(dspoints_p1_4x1[:, 0:3])
    target_pcd.points = o3d.utility.Vector3dVector(dspoints_p2_4x1[:, 0:3])

    dists = source_pcd.compute_point_cloud_distance(target_pcd)

    sum = 0
    for d in dists:
        sum += d
    
    print("Sum: ", sum)

if __name__ == '__main__':
    main()

