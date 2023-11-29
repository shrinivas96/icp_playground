import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
# from ranges_and_cartesian import read_scans_write_points, pol2cart
from utilities import *

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIG_SIZE = 12
BIGG_SIZE = 14
BIGGER_SIZE = 16

plt.rc('xtick', labelsize=BIG_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIG_SIZE)    # fontsize of the tick labels
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def plot_and_save(saveLocation: str, data: list, labels:list, markersizes: list):
    ax = plot_data_multiple(data, labels=labels, markersizes=markersizes)
    
    # plt.title("Comparision of real and simulated scan")
    plt.grid(True)
    plt.legend()
    plt.savefig(saveLocation, bbox_inches='tight')
    # plt.show()
    plt.close()

def sum_and_psi(target_pcd, pointCloud_4x1: np.ndarray):
    compare_pcd = o3d.geometry.PointCloud()
    compare_pcd.points = o3d.utility.Vector3dVector(pointCloud_4x1[:, 0:3])
    dist_bw_sim_and = np.array(target_pcd.compute_point_cloud_distance(compare_pcd))
    summation = dist_bw_sim_and.sum()
    psi = returnMetric(summation)
    return summation, psi



def main():
    # readFile = "data_add/logRealSimScanComparision.dat"
    readFile = "writing_data/logObsSimScan08.dat"
    imageFolder = "images/2023-08-05/"
    dataOutFolder = "npz_sava/2023-08-05/"
    rangeData = dataOutFolder + "rangesObsSim.npz"

    # needs to be done only once:
    # ranges_p1_obs, ranges_p2_sim = return_ranges(readFile)
    # np.savez(rangeData, ranges_p1=ranges_p1_obs, ranges_p2=ranges_p2_sim)

    # load data
    loadData = np.load(rangeData)
    ranges_p1_obs = loadData['ranges_p1']
    ranges_p2_sim = loadData['ranges_p2']

    ## 1. real scan at location
    points_p1_real_4x1 = returnFinites(convert_to_cartesian(np.copy(ranges_p1_obs)))
    
    # save array
    pointsRealData = dataOutFolder + "pointsReal.npz"
    np.savez(pointsRealData, points_p1_real_4x1)

    # save figure
    realScanImage = imageFolder + "realScan.png"
    plot_and_save(realScanImage, 
                  [points_p1_real_4x1[:, 0:2].T],
                  ['Scan at $(x_t, y_t, \\theta_t)$'], 
                  [5])

    ## 2. two different distorted images

    ranges_p1_dist5 = distortion(np.copy(ranges_p1_obs), mask_range_=1.5, mask_start_idx_=-1, window_size_=0, unif_min=0.05)
    points_p1_dist5_4x1 = returnFinites(convert_to_cartesian(ranges_p1_dist5))

    ranges_p1_dist8 = distortion(np.copy(ranges_p1_obs), mask_range_=1.5, mask_start_idx_=-1, window_size_=0, unif_min=0.08)
    points_p1_dist8_4x1 = returnFinites(convert_to_cartesian(ranges_p1_dist8))

    # save array
    pointsDistData = dataOutFolder + "pointsDist5n8.npz"
    np.savez(pointsDistData, points_p1_dist5_4x1, points_p1_dist8_4x1)

    # save figure
    realDDistScanImage = imageFolder + "realDistDistScan.png"
    plot_and_save(realDDistScanImage, 
                  [points_p1_real_4x1[:, 0:2].T, points_p1_dist5_4x1[:, 0:2].T, points_p1_dist8_4x1[:, 0:2].T],
                  ['Scan at $(x_t, y_t, \\theta_t)$', 'Distortions from $\\mathcal{U}(-0.05, 0.05)$', 'Distortions from $\\mathcal{U}(-0.08, 0.08)$'], 
                  [6, 4, 4])
    

    ## 3. mask at a location
    ranges_p1_mask = distortion(np.copy(ranges_p1_obs), mask_range_=3.5, mask_start_idx_=0, window_size_=55, unif_min=0)
    points_p1_mask_4x1 = returnFinites(convert_to_cartesian(ranges_p1_mask))

    # save array
    pointsMaskData = dataOutFolder + "pointsMask1.5.npz"
    np.savez(pointsMaskData, points_p1_mask_4x1)

    # save figure
    realMaskScanImage = imageFolder + "realMaskScan.png"
    plot_and_save(realMaskScanImage, 
                  [points_p1_real_4x1[:, 0:2].T, points_p1_mask_4x1[:, 0:2].T],
                  ['Scan at $(x_t, y_t, \\theta_t)$', 'Mask reporting range $3.5 \, m$'], 
                  [5, 5])

    ## 4. sim scan at location
    points_p2_sim_4x1 = returnFinites(convert_to_cartesian(ranges_p2_sim))

    # save array
    pointsSimData = dataOutFolder + "pointsSim.npz"
    np.savez(pointsSimData, points_p2_sim_4x1)

    # save figure
    realSimScanImage = imageFolder + "realSimSimple.png"
    plot_and_save(realSimScanImage, 
                  [points_p1_real_4x1[:, 0:2].T, points_p2_sim_4x1[:, 0:2].T],
                  ['Real Scan at $(x_t, y_t, \\theta_t)$', 'Simulated Scan at $(x_t, y_t, \\theta_t)$'],
                  [5, 5])

    ## 5. aligned scan
    # create an initial transformation for ICP to find the original
    fake_trans = np.eye(4)
    threshold = 0.02*1e1
    source_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()
    icp_aligned_pcd = o3d.geometry.PointCloud()

    source_pcd.points = o3d.utility.Vector3dVector(points_p1_real_4x1[:, 0:3])
    target_pcd.points = o3d.utility.Vector3dVector(points_p2_sim_4x1[:, 0:3])

    reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, fake_trans,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))

    # the final trasnformation as a 4x4 hom. trans.
    T_icp = reg_p2p.transformation
    
    print("Transformation is:")
    print(T_icp)

    # apply transformation to points of p1
    points_p1_reg_4x1 = np.matmul(points_p1_real_4x1, T_icp.T)
    icp_aligned_pcd.points = o3d.utility.Vector3dVector(points_p1_reg_4x1[:, 0:3])

    # save array
    pointsRegData = dataOutFolder + "pointsAligned.npz"
    np.savez(pointsRegData, points_p1_reg_4x1)

    # save figure
    simRealAlignedScanImage = imageFolder + "simRealAligned.png"
    plot_and_save(simRealAlignedScanImage, 
                  [points_p2_sim_4x1[:, 0:2].T, points_p1_reg_4x1[:, 0:2].T],
                  ['Simulated Scan', 'Real Scan after ICP'],
                  [5, 5])
    
    ## 6. compute metric for all scans
    sum_sim_real, psi_sim_real = sum_and_psi(target_pcd, points_p1_real_4x1)
    sum_sim_dist5, psi_sim_dist5 = sum_and_psi(target_pcd, points_p1_dist5_4x1)
    sum_sim_dist8, psi_sim_dist8 = sum_and_psi(target_pcd, points_p1_dist8_4x1)
    sum_sim_mask, psi_sim_mask = sum_and_psi(target_pcd, points_p1_mask_4x1)
    sum_sim_real_reg, psi_sim_real_reg = sum_and_psi(target_pcd, points_p1_reg_4x1)

    print(sum_sim_real, psi_sim_real)
    print(sum_sim_dist5, psi_sim_dist5)
    print(sum_sim_dist8, psi_sim_dist8)
    print(sum_sim_mask, psi_sim_mask)
    print(sum_sim_real_reg, psi_sim_real_reg)




if __name__ == '__main__':
    main()