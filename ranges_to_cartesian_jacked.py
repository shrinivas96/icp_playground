import yaml
import numpy as np
import matplotlib.pyplot as plt

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def combine_points(ranges_arr):
    v_cart = np.vectorize(pol2cart)

    # corresponding angles in [0, 2pi) for the range scans
    angles = np.arange(0, 360) * np.pi/180
    angles = np.reshape(angles, (360, 1))

    # ones and zeros array
    zeros = np.zeros((360, 1))
    ones = np.ones((360, 1))

    # empty array to hold all 4x1 point arrays
    points_4x1_array = []

    for arr in ranges_arr:
        xy_p1 = v_cart(np.reshape(arr, (360, 1)), angles)    
        points_p1 = np.column_stack((xy_p1[0], xy_p1[1], zeros, ones))
        points_4x1_array.append(points_p1)
    
    points_4x1_array = np.array(points_4x1_array)
    return points_4x1_array

def parse_ranges(yaml_data):
    data = yaml.safe_load(yaml_data)
    ranges_array = data['ranges']
    return ranges_array

def return_all_ranges(fileName):
    with open(fileName, 'r') as file:
        yaml_data = file.read()

    sequences = yaml_data.split('header:')

    ranges_arrays = []

    for sequence in sequences[1:]:  # Skip the first empty element
        ranges_array = parse_ranges('header:' + sequence)
        ranges_arrays.append(ranges_array)

    return np.array(ranges_arrays)


def get_in(readFile, rangesSaveFile, pointsSaveFile):
    # readFile = "data_add/simulatedScan_self.dat"
    # saveFile = 'data_add/points_from_self_sim_ranges.npz'
    
    # readFile = 'writing_data/new_scan_doc.dat'
    # rangesSaveFile = 'writing_data/ranges_new_scan_doc.npz'
    # pointsSaveFile = 'writing_data/points_new_scan_doc.npz'

    # needs to be done only once
    all_ranges = return_all_ranges(readFile)
    all_ranges = all_ranges.astype(np.float64)
    np.savez(rangesSaveFile, all_ranges)

    # loadFile = np.load(rangesSaveFile)
    # all_ranges = loadFile['arr_0']
    point_4x1_array = combine_points(all_ranges)
    np.savez(pointsSaveFile, point_4x1_array)

if __name__ == '__main__':
    pass