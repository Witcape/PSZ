import numpy as np

def defog(HazeImg, t, A, delta):
    t = np.maximum(np.abs(t), 0.00001) ** delta
    if np.isscalar(A):
        A = np.full(3, A)
    
    R = np.clip((HazeImg[:, :, 0] - A[0]) / t + A[0], 0, 1)
    G = np.clip((HazeImg[:, :, 1] - A[1]) / t + A[1], 0, 1)
    B = np.clip((HazeImg[:, :, 2] - A[2]) / t + A[2], 0, 1)
    
    rImg = np.stack([R, G, B], axis=2)
    return rImg