"""
Simulation studies of DSREM pipeline (step 1)
Usage: python -W ignore ./DSREM_simu_s1.py ./data/ ./result/

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2018-03-04
"""

import numpy as np
import sys
import nibabel as nib
from scipy.io import savemat
from lmm import lmm_fit
import warnings
warnings.simplefilter("ignore")

"""
installed all the libraries above
"""

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    """+++++++++++++++++++++++++++++++++++"""
    print("""\n Load dataset & preprocessing \n """)
    info_file_name = input_dir + "info.txt"
    dat = np.loadtxt(info_file_name)                        # including subject index, visit time, diagnostic info.
    m = dat.shape[0]                                        # the total number of observations
    nidx = dat[:, 0].reshape(-1, 1)                         # subject index
    n = int(nidx[m - 1, 0])                                 # number of subject
    x = dat[:, 1].reshape(-1, 1)                            # visit time
    p = x.shape[1]
    dx = dat[:, 2].reshape(-1, 1)                           # diagnostic information
    x0 = x[np.nonzero(dx == 0)[0], :]                       # covariates for normal controls
    nidx0 = nidx[np.nonzero(dx == 0)[0], :]                 # subject index for normal controls
    mask_file_name = input_dir + "mask.txt"
    mask = np.loadtxt(mask_file_name)
    sub = np.array(np.nonzero(mask == 1))
    d, n_v = sub.shape
    l, h = mask.shape
    img_file_name = input_dir + "img.nii"
    img = nib.load(img_file_name)
    img_mat = img.get_data()
    y = np.zeros(shape=(m, n_v))
    for j in range(n_v):
        y[:, j] = img_mat[sub[0, j], sub[1, j], :]
    y0 = y[np.nonzero(dx == 0)[0], :]                       # imaging intensity for normal controls

    """+++++++++++++++++++++++++++++++++++"""
    print("""\n Fit voxel-wised linear mixed model (lmm) \n """)
    beta = np.zeros(shape=(p+1, n_v))           # array for estimated fixed effect
    sigma = np.zeros(shape=(p+1, p+1, n_v))     # array for estimated covariance matrix of random effects
    scale = np.zeros(shape=(1, n_v))            # array for estimated scale parameter in error part
    for j in range(n_v):
        y0_j = y0[:, j].reshape(-1, 1)
        beta[:, j], sigma[:, :, j], scale[0, j] = lmm_fit(x0, y0_j, nidx0)
        print(j)

    """++++++++++++++++++++++++++++++++++++"""
    print("""\n Save results\n """)
    beta_file_name = output_dir + "beta.txt"
    np.savetxt(beta_file_name, beta[:, :])
    scale_file_name = output_dir + "scale.txt"
    np.savetxt(scale_file_name, scale[:, :])
    sigma_file_name = output_dir + "sigma.mat"
    savemat(sigma_file_name, mdict={'sigma': sigma})
