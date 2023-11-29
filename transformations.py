import numpy as np
def homo2vec(T):
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])

def vec2homo(x):
    T = np.eye(3)
    T[0:2, 2] = x[0:2]
    T[0:2, 0:2] = np.array([[np.cos(x[2]), -np.sin(x[2])], [np.sin(x[2]), np.cos(x[2])]])
    return T

xi = np.array([2, 3, np.pi/3])
Xi = vec2homo(xi)
Ri = Xi[0:2, 0:2]

xj = np.array([2+4, 3+2, (np.pi/3)+(np.pi/2)])
Xj = vec2homo(xj)

print("xi = {}\nxj = {}".format(xi, xj))

xji = xj - xi
xij = xi - xj

# print("xi - xj = ", xij)

Xij = np.dot(np.linalg.inv(Xi), Xj)
Xiiij = np.dot(np.linalg.inv(Xi), np.linalg.inv(Xj))
Xijii = np.dot(np.linalg.inv(Xj), np.linalg.inv(Xi))

# print("inv(Xi)*Xj = ", homo2vec(Xij))
# print("inv(Xi)*inv(Xj) = ", homo2vec(Xiiij))
# print("inv(Xj)*inv(Xi) = ", homo2vec(Xijii))

# print(homo2vec((np.linalg.inv(Xi))))

print(XXi)
print(np.linalg.inv(XXi))
print(XXi.T)