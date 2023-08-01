from numpy.linalg import inv
import numpy as np
from scipy.spatial.transform import Rotation

def apply_transformation(matrix, vector):
    # define vector with zeros, same size as the input vector
    result_vector = np.zeros(vector.shape[0])

    # for each row in the matrix
    for i in range(matrix.shape[0]):
        # get the row
        row_i = matrix[i, :]

        # for each element in the vector
        for j in range(vector.shape[0]):
            # multiple each element of vector with each element of row
            elem = vector[j]
            r = row_i[j]

            # if either of them are 0 just set them to 0 and move on
            if elem == 0 or r == 0:
                result_vector[i] += 0
                continue

            # if either element is inf, then the whole row is going to be inf
            if elem == np.inf or elem == -np.inf:
                result_vector[i] = elem
                break

            # finally, multiplication in case all numbers are "normal"
            result_vector[i] += r * elem
    
    # return the resulting vector
    return result_vector    
            

def pose_to_transform(pose):
    x, y, z = pose['position']
    qx, qy, qz, qw = pose['orientation']

    Rp = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

    tp = np.eye(4)

    tp[0:3, 0:3] = Rp
    tp[0, 3] = x
    tp[1, 3] = y
    tp[2, 3] = z
    
    return tp


def main():
    # poses related to time stamps 147 and 157
    pose1 = {'position': (6.358286629145617, 4.071427193736686, -0.0010075003550408732),
             'orientation': (-0.0014029327973987555, 0.0007482064529068695, 0.8819255550000044, 0.4713860280116544)}

    pose2 = {'position': (5.619354267905068, 4.211060580572698, -0.0010079803173249774),
             'orientation': (-0.001591063939694671, -4.136325179491184e-05, 0.9997038359707867, -0.024283886631176782)}
    
    HT_C1 = pose_to_transform(pose1)
    HT_C2 = pose_to_transform(pose2)

    np.savez("data/poses_as_transforms.npz", HT_C1=HT_C1, HT_C2=HT_C2)
    
    C1_invC2 = np.matmul(HT_C1, inv(HT_C2))
    invC2_C1 = np.matmul(inv(HT_C2), HT_C1)
    C2_invC1 = np.matmul(HT_C2, inv(HT_C1))
    invC1_C2 = np.matmul(inv(HT_C1), HT_C2)

    np.savez("data/transforms.npz", C1_invC2, invC2_C1, C2_invC1, invC1_C2)


if __name__ == '__main__':
    my_trans = np.eye(4)
    # c = 0.5
    # s = -0.8660254
    # my_trans[0:2, 0:2] = np.array([[c, -s], [s, c]])
    # my_trans[0, 3] = -0.8
    # my_trans[1, 3] = 0.3

    vec = np.array([1, 2,  0.,  np.inf])

    print(apply_transformation(my_trans, vec))
    