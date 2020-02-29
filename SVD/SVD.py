import numpy as np 
from PIL import Image
import math 
import os

#Function of SVD 
def SVD(sigma, u, v, k):
    m = len(u)
    n = len(v[0])
    a = np.zeros((m, n))
    for kk in range(k):
        uu = u[:,kk].reshape(m, 1)
        vv = v[kk].reshape(1, n)
        a = a + sigma[kk] * np.dot(uu, vv)
    a = a.clip(0, 255)
    return np.rint(a).astype('uint8')

#Read image
A = Image.open("1.jpg",'r')
a = np.array(A)
print('Image.size: ',a.shape)

#Deal with R G B matrixes
ur,sigmar,vr = np.linalg.svd(a[:, :, 0])
ug,sigmag,vg = np.linalg.svd(a[:, :, 1])
ub,sigmab,vb = np.linalg.svd(a[:, :, 2])

Num = 20
for k in range(1,Num+1):
    R = SVD(sigmar, ur, vr, k)
    G = SVD(sigmag, ug, vg, k)
    B = SVD(sigmab, ub, vb, k)
    new_image = np.stack((R, G, B), axis=2)
    Image.fromarray(new_image).save('SVD_out\svd_%d.jpg' % k)

