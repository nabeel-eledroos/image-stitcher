import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import plot_matches

from detectCorners import detectCorners
from extractFeatures import extractFeatures
from computeMatches import computeMatches
from mergeImages import mergeImages
from ransac import ransac

def showMatches(im1, im2, c1, c2, matches, title=""):
    disp_matches = np.array([np.arange(matches.shape[0]), matches]).T.astype(int)
    valid_matches = np.where(matches>=0)[0]
    disp_matches = disp_matches[valid_matches, :]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plot_matches(ax, im1, im2, 
            c1[[1, 0], :].astype(int).T, c2[[1,0], :].astype(int).T, disp_matches)
    ax.set_title(title)


def imread(path):
    img = plt.imread(path).astype(float)
    
    #Remove alpha channel if it exists
    if img.ndim > 2 and img.shape[2] == 4:
        img = img[:, :, 0:3]
    #Puts images values in range [0,1]
    if img.max() > 1.0:
        img /= 255.0

    return img


im1 = imread('../data/building_left.jpg') #left image
im2 = imread('../data/building_right1.jpg') #right image

sigma = 1.5
threshold = 0.0005
max_corners = 200
isSimple = False
c1 = detectCorners(im1, sigma, threshold)
c2 = detectCorners(im2, sigma, threshold)

n1 = min(max_corners, c1.shape[1])
c1 = c1[:, 0:n1]
n2 = min(max_corners, c2.shape[1])
c2 = c2[:, 0:n2]

#Compute feature descriptors
patch_radius = 5
f1 = extractFeatures(im1, c1, patch_radius)

f2 = extractFeatures(im2, c2, patch_radius)

#Compute matches
matches = computeMatches(f1, f2)
showMatches(im1, im2, c1, c2, matches, title='All correspondences')
plt.show()

#Estimate transformation
inliers, transf = ransac(matches, c1, c2)
good_matches = np.zeros_like(matches)-1
good_matches[inliers] = matches[inliers]
showMatches(im1, im2, c1, c2, good_matches, title='Inliers')
plt.show()

#Warp images
stitch_im = mergeImages(im1, im2, transf)
plt.imshow(stitch_im)
plt.title('Stitched Image')
plt.show()