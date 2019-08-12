import numpy as np
from skimage.color import rgb2gray

def extractFeatures(im, c, patch_radius):
    r = patch_radius
    img = rgb2gray(im)
    img = np.pad(img, pad_width = r,mode='constant')
    cx = c[1,:].astype(int)
    cy = c[0,:].astype(int)
    d = (2*r+1)
    d2 = d//2
    f = []
    for x in range(0,len(cx)):
        x1 = cx[x] - d2
        x2 = cx[x] + d2
        y1 = cy[x] - d2
        y2 = cy[x] + d2
        patch = img[x1:x2, y1:y2]
        feature = patch.reshape(-1)
        f.append(feature)
    return f