import numpy as np
from numpy import linalg as LA

def computeMatches(f1, f2):
    m = np.zeros((len(f1)))
    for i in range(0,len(f1)):
        mins = 1000000000
        for j in range(0,len(f2)):
            sd = LA.norm(f1[i]-f2[j])
            if(sd<mins):
                mins = sd
                m[i] = j
    return m

