import numpy as np

A = np.eye(3)
T = np.array([1, 2, 3])

print(f"K^t=\n {np.cross(A, T)}")

