import numpy as np
from random import randint
import math
import time

def ransac(matches, c1, c2):
    count = 0
    N = 3000
    thresh = 50
        # 1 - 50
        # 2 - 200
        # 3 - 50
        # 4 - 70
        # 5 - 50
    d=0
    inliers = []
    transf = []
    numIn = 0
    while count<N:
        points = []
        ppoints = []
        #pick s points at random
        for i in range(0,2):
            ind = np.random.randint(len(c1[0]))
            x = c1[1,ind].astype(int)
            y = c1[0,ind].astype(int)
            points.append((x, y))
            pInd = matches[ind].astype(int)
            px = c2[1,pInd].astype(int)
            py = c2[0,pInd].astype(int)
            ppoints.append((px, py))

        #calculate scale
        numerator = math.sqrt((ppoints[0][0] - ppoints[1][0])**2 + (ppoints[0][1] - ppoints[1][1])**2)
        denominator = math.sqrt((points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2)
        if(denominator == 0):
            continue
        scale = numerator / denominator

        #calculate translation tx and ty
        tx = ppoints[0][0] - (scale * points[0][0])
        ty = ppoints[0][1] - (scale * points[0][1])

        #check euclidean distances less than threshold
        im1x = c1[1,:].astype(int)
        im1y = c1[0,:].astype(int)

        im2x = c2[1,:].astype(int)
        im2y = c2[0,:].astype(int)
        possible = []
        d = 0
        for x in range(0, len(im1x)):
            xprime = im2x[matches[x].astype(int)]
            yprime = im2y[matches[x].astype(int)]
            Tx1 = (xprime - tx) // scale
            Ty1 = (yprime - ty) // scale
            euDis = (im1x[x] - Tx1)**2 + (im1y[x] - Ty1)**2
            if euDis < thresh:
                possible.append(x)
                d += 1
        if(d>numIn):
            transf = []
            numIn = d
            inliers = possible
            transf.append(ty)
            transf.append(tx)
            transf.append(scale)
        else:
            possible = []
            d = 0
        count += 1

    return(inliers, transf)
