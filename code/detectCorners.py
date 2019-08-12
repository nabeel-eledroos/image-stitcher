import numpy as np
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
from scipy.ndimage.filters import maximum_filter
from skimage.morphology import local_maxima
from skimage.feature import peak_local_max
import numpy as np

def gaussian(hsize=3,sigma=0.5):
    shape = (hsize, hsize)
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def nms(corner_score):
    #Pick local maxima in a window
    locs = peak_local_max(corner_score, 5)
    #Get scores at maximum peaks
    scores = corner_score[locs[:, 0], locs[:, 1]]
    #Sort scores
    idx_sorted = np.argsort(scores)

    #Get scores and coordinates according to sorting
    cs = scores[idx_sorted]
    cx = locs[idx_sorted, 1]
    cy = locs[idx_sorted, 0]

    return cx, cy, cs

def detectCorners(I, w, th):
    I = I.astype(float)
    if I.ndim > 2:
        I = rgb2gray(I)

    corner_score = simple_score(I, w)


    corner_score[corner_score < th] = 0

    cx, cy, cs = nms(corner_score)
    c = np.array([cx, cy, cs])
    isvalid = np.logical_and(
            np.logical_and(cx >= 0, cx < I.shape[1]),
            np.logical_and(cy >= 0, cy < I.shape[0]))
    return c[:, isvalid]


def simple_score(I, w):
    gs = gaussian(6*w+1, w)
    f = []

    f.append(np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]))
    f.append(np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]]))
    f.append(np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]]))
    f.append(np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]]))
    f.append(np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]]))
    f.append(np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]]))
    f.append(np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]]))
    f.append(np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]]))

    corner_score = np.zeros_like(I)
    for i in xrange(8):
        diff = convolve(I, f[i], mode='nearest')
        diff_sum = convolve(diff**2, gs, mode='nearest')
        corner_score += diff_sum
    return corner_score
