import numpy as np


if __name__ == '__main__':
    fileName = "poses_raw_graph_base"
    filePath = "export_data/{}.txt".format(fileName)
    
    all_poses = []
    with open(filePath, "r") as file:
        for line in file:
            # Split the line into individual numbers
            # Convert the numbers into a numpy array
            poses_as_transform = np.array(line.split(), dtype=float)
            all_poses.append(poses_as_transform)

    all_poses = np.array(all_poses)
    savePath = "export_data/{}.npz".format(fileName)

    np.savez(savePath, all_poses)