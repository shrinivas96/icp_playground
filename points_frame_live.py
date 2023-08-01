import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


saveFile = 'data_add/points_scan_dump.npz'
loadFile = np.load(saveFile)
points_array_4x1 = loadFile['arr_0']
points_array = points_array_4x1[:, :, 0:2]

fig, ax = plt.subplots()
scatter = ax.scatter([], [], s=10)

def update(frame):
    ax.clear()
    x, y = points_array[frame].T
    scatter = ax.scatter(x, y, s=10)
    ax.set_xlim(-4, 4)  # Set appropriate limits based on your data
    ax.set_ylim(-4, 4)
    # ax.set_title(f"Frame {frame + 1}/{points_array.shape[0]}")
    return scatter,

def init():
    ax.set_xlim(0, 1)  # Set appropriate limits based on your data
    ax.set_ylim(0, 1)
    return scatter,

num_frames = points_array.shape[0]

ani = FuncAnimation(fig, update, frames=num_frames, blit=True)
plt.show()
