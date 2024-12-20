import numpy as np

########################### ( C, H, W ) ###########################

def sam(x, y):
    num = np.sum(np.multiply(x, y), 0)
    den = np.sqrt(np.multiply(np.sum(x**2, 0), np.sum(y**2, 0)))
    sam = np.sum(np.degrees(np.arccos(np.clip(num / den, -1, 1)))) / (x.shape[2]*x.shape[1])
    return sam

def psnr(x,y):
    bands = x.shape[0]
    x = np.reshape(x, [bands,-1])
    y = np.reshape(y, [bands,-1])
    msr = np.mean((x-y)**2, 1)
    maxval = np.max(y, 1)**2
    return np.mean(10*np.log10(maxval/msr))

def ergas(x, y, Resize_fact=4):
    err = y-x
    ergas=0
    for i in range(y.shape[0]):
        ergas += np.mean(np.power(err[i],2)) / (np.mean(y[i])**2)
    ergas = (100.0/Resize_fact) * np.sqrt(1.0/y.shape[0] * ergas)
    return ergas
        

def rmse(x, y, maxv=1, minv=0):
    rmse_total = np.sqrt(np.mean(np.power(x-y, 2)))
    rmse_total = rmse_total* (maxv-minv) + minv
    return rmse_total
