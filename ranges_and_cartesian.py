import numpy as np
import matplotlib.pyplot as plt

def normalise_angles(value, start=0, end=2*np.pi):
    # normalises angles between 0 and 2pi. 
    # source https://stackoverflow.com/a/2021986/6609148
    # simpler implementation https://stackoverflow.com/a/37358130/6609148
    width = end - start
    offset = value - start
    return (offset - (np.floor(offset / width) * width)) + start


def cart2pol(x, y):
    ranges = np.sqrt(x*x + y*y)
    phi = np.arctan2(y, x)
    return ranges, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def read_scans_write_points(fileToRead, saveFile):
    v_cart = np.vectorize(pol2cart)
    infCount1 = 0
    infCount2 = 0

    # corresponding angles in [0, 2pi) for the range scans
    angles = np.arange(0, 360) * np.pi/180
    angles = np.reshape(angles, (360, 1))

    # scan file contains just two lines of scan copy pasted from logger on remote machine
    with open(fileToRead, "r") as file:
        ranges_p1 = []
        ranges_p2 = []
        
        # first line
        for x in file.readline().split(", "):
            if(x == "inf"):
                infCount1+=1
                # x = "0.0"
            ranges_p1.append(float(x.rstrip("\n")))
        
        # second line
        for x in file.readline().split(", "):
            if(x == "inf"):
                infCount2+=1
                # x = "0.0"
            ranges_p2.append(float(x.rstrip("\n")))
        
        ranges_p1 = np.array(ranges_p1)
        ranges_p2 = np.array(ranges_p2)
    
    print("Count 1: ", infCount1)
    print("Count 2: ", infCount2)
    # adjusting arrays to the right shape
    ranges_p1 = np.reshape(ranges_p1, (360, 1))
    ranges_p2 = np.reshape(ranges_p2, (360, 1))

    # ones and zeros array
    zeros = np.zeros((360, 1))
    ones = np.ones((360, 1))

    # converting polar to cartesian
    xy_p1 = v_cart(ranges_p1, angles)
    xy_p2 = v_cart(ranges_p2, angles)
    
    # appending 0's and 1's to make it usable with 4x4 matrices. 
    # 0's in third dimension just means it is one plane in 3D points
    points_p1 = np.column_stack((xy_p1[0], xy_p1[1], zeros, ones))
    points_p2 = np.column_stack((xy_p2[0], xy_p2[1], zeros, ones))

    np.savez(saveFile, points_p1=points_p1, points_p2=points_p2)


def main():
    # readFile = "data_add/simulatedScan_self.dat"
    # saveFile = 'data_add/points_from_self_sim_ranges.npz'
    readFile = "data_add/confirming_large_distance.dat"
    saveFile = 'data_add/confirming_large_distance.npz'

    read_scans_write_points(readFile, saveFile)

if __name__ == '__main__':
    main()
    