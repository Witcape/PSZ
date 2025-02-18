import numpy as np
from scipy.signal import convolve2d
from scipy.fftpack import fft2, ifft2

def cal_weight_fun(HazeImg, D, param):
    sigma = param
    HazeImg = HazeImg.astype(np.float64) / 255
    d_r = convolve2d(HazeImg[:, :, 0], D, mode='same', boundary='wrap')
    d_g = convolve2d(HazeImg[:, :, 1], D, mode='same', boundary='wrap')
    d_b = convolve2d(HazeImg[:, :, 2], D, mode='same', boundary='wrap')
    return np.exp(-(d_r**2 + d_g**2 + d_b**2) / (2 * sigma))

def cal_trans(HazeImg, t, lambda_, param):
    nRows, nCols = t.shape
    
    d_filters = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
        np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
        np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    ]
    
    d_filters = [d / np.linalg.norm(d) for d in d_filters]
    WFun = [cal_weight_fun(HazeImg, d, param) for d in d_filters]
    
    Tf = fft2(t)
    DS = sum(abs(fft2(d, (nRows, nCols)))**2 for d in d_filters)
    
    beta, beta_rate, beta_max = 1, 2 * np.sqrt(2), 2**8
    
    while beta < beta_max:
        gamma = lambda_ / beta
        DU = sum(
            fft2(convolve2d(
                np.maximum(np.abs(convolve2d(t, d, mode='same', boundary='wrap')) - W / (beta * len(d_filters)), 0) * np.sign(convolve2d(t, d, mode='same', boundary='wrap')),
                np.flip(d), mode='same', boundary='wrap'))
            for d, W in zip(d_filters, WFun)
        )
        t = np.abs(ifft2((gamma * Tf + DU) / (gamma + DS)))
        beta *= beta_rate
    
    return t