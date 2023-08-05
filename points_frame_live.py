import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dist_bw_pcd import returnFinites
import ranges_to_cartesian_jacked as rcj


def visualise_points(saveFile):
    # saveFile = 'data_add/points_scan_dump.npz'
    # saveFile = '/home/shrini/workspace/playgrnd/icp_playground/writing_data/points_new_sim_scan_doc.npz'
    loadFile = np.load(saveFile)
    points_array_4x1 = loadFile['arr_0']
    # points_array_4x1 = returnFinites(points_array_4x1)
    points_array = points_array_4x1[:, :, 0:2]

    fig, ax = plt.subplots()
    scatter = ax.scatter([], [], s=10)

    def update(frame):
        ax.clear()
        x, y = points_array[frame].T
        scatter = ax.scatter(x, y, s=10)
        ax.set_xlim(-10, 10)  # Set appropriate limits based on your data
        ax.set_ylim(-10, 10)
        # ax.set_title(f"Frame {frame + 1}/{points_array.shape[0]}")
        return scatter,

    def init():
        ax.set_xlim(0, 1)  # Set appropriate limits based on your data
        ax.set_ylim(0, 1)
        return scatter,

    num_frames = points_array.shape[0]

    ani = FuncAnimation(fig, update, frames=num_frames, blit=True)
    plt.show()


def main():
    readFile = 'writing_data/new_sim_scan_doc.dat'
    rangesSaveFile = 'writing_data/ranges_new_sim_scan_doc.npz'
    pointsSaveFile = 'writing_data/points_new_sim_scan_doc.npz'
    rcj.get_in(readFile, rangesSaveFile, pointsSaveFile)

    visualise_points(pointsSaveFile)

if __name__ == '__main__':
    main()