import enum
import numpy as np
import satellite_cloud_generator as scg
import torch
import torchvision
from s2cloudless import S2PixelCloudDetector
import os 
import matplotlib.pyplot as plt

def findClose(a, bands):
    return min(range(len(bands)), key=lambda i: abs(bands[i]-a))

def normGray(R):
    # R: (C, H, W)
    R_ = torch.zeros(R.shape, dtype=torch.float32)
    for i in range(R.shape[0]):
        mean, std = R[i].mean(), R[i].std()
        R_[i] = (R[i]-mean)/std
        R_[i] = R_[i].clip(-3.0, 3.0) # clip out reflectance which is out of range
        R_[i] = R_[i]*std + mean
        R_[i] = (R_[i]-R_[i].min())/(R_[i].max()-R_[i].min())
        R_[i] = R_[i].clip(0.0, 1.0) 
    return R_

def normHSI(R):
    # R: (C, H, W)
    rmax, rmin = np.max(R), np.min(R)
    R = (R - rmin)/(rmax - rmin)
    return R

def prime(a, b):
    prime_list = []
    for i in range(a, b):
        if i == 0 or i == 1:
            continue
        else:
            for j in range(2, int(i/2)+1):
                if i % j == 0:
                    break
            else:
                prime_list.append(i)
    return prime_list
    
def generateCloudyMags(cloud_path, threshold=0.999, q=0.85, q2=0.5): 
    cloudy_hsi = np.load(f'{cloud_path}/cloudy.npy')
    clean_hsi = np.load(f'{cloud_path}/clean.npy')

    # These are about the bandwidth of the HSI image from AVIRIS dataset and then doing some pre-processing.
    bands = [365.9298,375.5940,385.2625,394.9355,404.6129,414.2946,423.9808,433.6713,443.3662,453.0655,462.7692,472.4773,482.1898,491.9066,501.6279,511.3535,521.0836,530.8180,540.5568,550.3000,560.0477,569.7996,579.5560,589.3168,599.0819,608.8515,618.6254,628.4037,638.1865,647.9736,657.7651,667.5610,654.7923,664.5994,674.4012,684.1979,693.9894,703.7756,713.5566,723.3325,733.1031,742.8685,752.6287,762.3837,772.1335,781.8781,791.6174,801.3516,811.0805,820.8043,830.5228,840.2361,849.9442,859.6471,869.3448,879.0372,888.7245,898.4066,908.0834,917.7551,927.4214,937.0827,946.7387,956.3895,966.0351,975.6755,985.3106,994.9406,1004.565,1014.185,1023.799,1033.408,1043.012,1052.611,1062.204,1071.793,1081.376,1090.954,1100.526,1110.094,1119.656,1129.213,1138.765,1148.311,1157.853,1167.389,1176.920,1186.446,1195.966,1205.482,1214.992,1224.497,1233.996,1243.491,1252.980,1262.464,1252.773,1262.746,1272.718,1282.691,1292.662,1302.634,1312.606,1322.577,1332.548,1342.519,1352.490,1362.460,1372.430,1382.400,1392.369,1402.339,1412.308,1422.277,1432.245,1442.214,1452.182,1462.150,1472.118,1482.085,1492.052,1502.019,1511.986,1521.952,1531.918,1541.885,1551.850,1561.816,1571.781,1581.746,1591.711,1601.675,1611.640,1621.604,1631.568,1641.531,1651.494,1661.458,1671.420,1681.383,1691.345,1701.307,1711.269,1721.231,1731.192,1741.153,1751.114,1761.075,1771.036,1780.996,1790.956,1800.915,1810.875,1820.834,1830.793,1840.752,1850.710,1860.669,1870.627,1871.784,1865.964,1876.025,1886.085,1896.141,1906.196,1916.248,1926.298,1936.346,1946.391,1956.435,1966.475,1976.514,1986.550,1996.584,2006.615,2016.645,2026.672,2036.696,2046.719,2056.739,2066.756,2076.772,2086.785,2096.796,2106.804,2116.810,2126.814,2136.816,2146.815,2156.812,2166.807,2176.799,2186.789,2196.777,2206.762,2216.745,2226.726,2236.705,2246.681,2256.655,2266.626,2276.595,2286.562,2296.527,2306.490,2316.449,2326.407,2336.363,2346.316,2356.267,2366.215,2376.161,2386.105,2396.047,2405.986,2415.923,2425.858,2435.790,2445.720,2455.648,2465.573,2475.496,2485.417,2495.336]
    bands = bands[:214]
    del bands[151:170]
    del bands[103:116]
    bands = bands[10:]

    cloudy_hsi_ng = normGray(torch.tensor(cloudy_hsi.transpose((2,0,1)).astype('float32'))) # channel-wise normalization

    sentinelBands = [443,490,560,665,705,740,783,842,865,940,1375,1610,2190] # Sentinel-2 MSI bandwidth
    senlbid = [findClose(k, bands) for k in sentinelBands]

    sentlbands = cloudy_hsi_ng[senlbid].permute(1,2,0)

    cloud_detector = S2PixelCloudDetector(threshold=threshold, average_over=1, dilation_size=1, all_bands=True)
    cloud_prob = cloud_detector.get_cloud_probability_maps(sentlbands[np.newaxis, ...])
    cloud_mask = cloud_detector.get_cloud_masks(sentlbands[np.newaxis, ...])

    cloudy_hsi = torch.FloatTensor(normHSI(cloudy_hsi.transpose((2,0,1))))
    clean_hsi = torch.FloatTensor(normHSI(clean_hsi.transpose((2,0,1))))

    cmags = scg.q_mag(cloudy_hsi, torch.tensor(cloud_mask)==0.0, mask_cloudy=torch.tensor(cloud_mask)==1.0, clean=clean_hsi, q=q, q2=q2)

    return cmags

def randVilualize(path, img=None, n=9):
    if not img:
        file_names = os.listdir(path)
        rand_files = np.random.randint(0, len(file_names))
        file = np.load(os.path.join(path, file_names[rand_files]))
        img, desc = torch.FloatTensor(file['HSI']), file['Description']
        band_list = [5, 14, 24]
        i = 0
        while i < (n-3):
            k = np.random.randint(0, img.shape[0])
            if k not in band_list:
                band_list.append(k)
                i += 1
        fig, axs = plt.subplots(3, int(n/3))
        fig.supxlabel(f'Description: {desc}')
        for idx, ax in enumerate(axs.flat):
            ax.imshow(img[band_list[idx]])
            ax.set_title(f'Band {band_list[idx]}', fontsize='small', loc='center')
            ax.set_axis_off()
        plt.axis('off')
        

def spectralSmoothing(hsi, fac=4): ##DEPRECATED##
    '''
    Smoothing HSI along spectrals by bilinear method.
    inputs: 
        hsi (Tensor): (C, H, W)
        fac (integer): specify the reducing channels factor 
    return: hsi_ (Tensor): (C//fac, H, W)
    '''
    c, h, w = hsi.shape
    c_ = c//fac

    hsi_ = torch.zeros((c_,h,w), dtype=torch.float32)
    weight = torch.ones(fac//2) / torch.arange(1, 1+(fac//2)*2, 2)
    weight = torch.cat((weight.flip(dims=[0]), weight), dim=0) ## e.g. x4: (1/3, 1/1, 1/1, 1/3)
    weight = weight.view(fac, 1, 1)

    for idx, i in enumerate(range(fac//2, c-(fac//2+1)-1, fac)):
        hsi_[idx,:,:] = torch.sum(hsi[i-(fac//2-1):i+fac//2 + 1, :, :] * weight, dim=0) / torch.sum(weight)
    
    return hsi_, 'Spectral Blurring'

def spatialBlurring(hsi, downsample=4, mode='bilinear'):
    '''
    Downsampling the original HSI.
    inputs:
        hsi (Tensor): (C, H, W)
        downsample (int): specify the ratio for super resolution, e.g. 4 means x4
        mode (string): specify the interpolation method which can be 'bilinear', 'bicubic' and 'nearest'
    return: (C, H*downsample_rate, W*downsample_rate)
    '''
    c, h, w = hsi.shape
    h_, w_ = h//downsample, w//downsample
    hsi_ = torch.rand(hsi.shape).copy_(hsi)
    if mode=='bicubic':
        itpm = torchvision.transforms.InterpolationMode.BICUBIC
    elif mode=='nearest':
        itpm = torchvision.transforms.InterpolationMode.NEAREST
    else:
        itpm = torchvision.transforms.InterpolationMode.BILINEAR
    hsi_ = torchvision.transforms.Resize(size=(h_,w_), interpolation=itpm)(hsi_)
    
    return hsi_, 'Spatial Blurring'
    

def addGaussianNoise(hsi, snr=15):
    xpower = torch.sum(hsi**2) / hsi.numel()
    npower = torch.sqrt(xpower / snr)
    hsi = hsi + torch.randn(hsi.shape) * npower
    return hsi, 'Noisy'

def bandMissing(hsi, mask_rate=0.4):
    c, h, w = hsi.shape
    hsi_ = torch.rand(hsi.shape).copy_(hsi)
    desc = ''
    mode = np.random.choice([0,1,2], p=[0.35, 0.25, 0.4]) # 0: complete, 1: band-wise, 2: partial
    mt = np.random.uniform()

    if mode==0:
        if mt < 0.3:
            ## missing at continuous bands
            missing_band_num = int(np.random.randn(1)*10) - 5 + int(c*mask_rate)
            if missing_band_num > 172:
                missing_band_num = 172
            rand_init_idx = np.random.randint(0, c-missing_band_num+1) 
            missing_bands = list(range(rand_init_idx, rand_init_idx+missing_band_num, 1))
        else: 
            ## each band has 'mask_rate' probability to miss
            missing_bands = list(range(c))
            k = 1
            for i in range(c):
                if np.random.uniform() < 1-mask_rate:
                    missing_bands.pop(-k)
                else:
                    k = k+1
        hsi_[missing_bands,:,:] = 0.
        desc = 'Bands Complete Missing' 
    elif mode==1: # the probability of mode1 is less 
        # missing [2,4,6,...] or [1,3,5,...] or [3,5,7,...] liked
        stride = 2
        missing_rows = list(range(np.random.randint(0, stride), w, stride))
        missing_band_num = int(mask_rate*c*h/len(missing_rows))
        if missing_band_num > 172:
            missing_band_num = 172
        if mt < 0.3:
            ## missing at continuous bands
            rand_init_idx = np.random.randint(0, c-missing_band_num+1) 
            missing_bands = list(range(rand_init_idx, rand_init_idx+missing_band_num, 1))
        else: 
            ## each band has 'missing_band_num/c' probability to miss
            missing_bands = list(range(c))
            k = 1
            for i in range(c):
                if np.random.uniform() < 1 - missing_band_num/c:
                    missing_bands.pop(-k)
                else:
                    k = k+1

        temp = hsi_[missing_bands,:,:]
        temp[:,missing_rows,:] = 0.
        hsi_[missing_bands,:,:] = temp
        desc = 'Band-wise Missing'
    else:
        missing_rows_num = np.random.randint(int(h*mask_rate), c)
        missing_rows = list(range(h))
        
        k = 1
        for i in range(h):
            if np.random.uniform() < 1 - missing_rows_num/h:
                missing_rows.pop(-k)
            else:
                k = k+1

        missing_band_num = int(mask_rate*c*h/missing_rows_num)
        if missing_band_num > 172:
            missing_band_num = 172
        if mt < 0.3:
            ## missing at continuous bands
            rand_init_idx = np.random.randint(0, c-missing_band_num+1) 
            missing_bands = list(range(rand_init_idx, rand_init_idx+missing_band_num, 1))
        else: 
            ## each band has 'missing_band_num/c' probability to miss
            missing_bands = list(range(c))
            k = 1
            for i in range(c):
                if np.random.uniform() < missing_band_num/c:
                    missing_bands.pop(-k)
                else:
                    k = k+1
        temp = hsi_[missing_bands,:,:]
        temp[:,missing_rows,:] = 0.
        hsi_[missing_bands,:,:] = temp
        desc = 'Partial Missing'

    return hsi_, desc

def cloudGeneration(hsi, cmags, p=0.5):
    desc = ''
    if np.random.uniform() < p:
        ## thin cloud but large covering
        hsi_ = scg.add_cloud_and_shadow(hsi,
                            channel_magnitude=cmags,
                            locality_degree=1, 
                            min_lvl=[0.0, 0.4], 
                            max_lvl=[0.4, 0.6], 
                            clear_threshold=0.0, 
                            blur_scaling=2,
                            cloud_color=True,
                            channel_offset=0,
                            decay_factor=1.0,
                            return_cloud=False)
        desc = 'Thinly Cloudy'
    else:
        ## thick clouds but small covering
        hsi_ = scg.add_cloud_and_shadow(hsi,
                            channel_magnitude=cmags,
                            locality_degree=[2,4], 
                            min_lvl=0.0, 
                            max_lvl=1.0, 
                            clear_threshold=[0.0, 0.4], 
                            blur_scaling=1,
                            cloud_color=True,
                            channel_offset=0,
                            decay_factor=1.0,
                            return_cloud=False)
        desc = 'Thickly Cloudy'
    return hsi_.squeeze(), desc

def iterData(root, store_path, dagtypes, method, params, cmags=None):
    stages = ['train', 'val', 'test']
    for s in stages:
        datalist = os.listdir(os.path.join(root, s))
        for data in  datalist:
            if 'GT.npz.npy' in data:
                gt = np.load(os.path.join(root, s, data))[:256,:256,:]
                gt = torch.tensor(normHSI(np.transpose(gt, (2,0,1))))
                for dt in dagtypes:
                    for i, lv in enumerate(params[dt]):
                        if dt == 'Cloudy':
                            deg, _ = method[dt](gt, cmags, lv)
                        else:
                            deg, _ = method[dt](gt, lv)
                        np.save(os.path.join(store_path, dt, f'LV{i+1}', s, f'{data[:-10]}_{dt}.npy'), np.transpose(deg.numpy(), (1,2,0))) # shape: h, w, c
                print(f'Completing generation of deg {data} !')

if __name__ == '__main__':
    cmags = generateCloudyMags('.') # reference datas are in the same folder e.g. clean.npy and cloudy.npy
    root_path = '../data'
    store_path = '../data/data_deg' # change the path you want to save

    if not os.path.exists(store_path):
        os.makedirs(store_path)

    degtypes = ['Cloudy', 'Noisy', 'BandMissing', 'SpatialBlurring']
    methods = {
        'Cloudy': cloudGeneration, 
        'Noisy': addGaussianNoise,
        'BandMissing': bandMissing,
        'SpatialBlurring': spatialBlurring
    }
    ## multiple levels
    # degLvDict = {
    #     'Cloudy': [0.25, 0.5, 0.75], # p
    #     'Noisy': [30, 50, 70], # sigma
    #     'BandMissing': [0.2, 0.3, 0.4], # mask ratio
    #     'SpatialBlurring': [2, 4, 8] # downsample 
    # }
    ## only one level
    degLvDict = {
        'Cloudy': [0.25], # p
        'Noisy': [15], # snr
        'BandMissing': [0.2], # mask ratio
        'SpatialBlurring': [2] # downsample 
    }

    for dt in degtypes:
        for i in range(1):
            for stage in ['train', 'val', 'test']:
                if not os.path.exists(os.path.join(store_path, dt, f'LV{i+1}', stage)):
                    os.makedirs(os.path.join(store_path, dt, f'LV{i+1}', stage))
                    
    iterData(root_path, store_path, degtypes, methods, degLvDict, cmags)



    
    
