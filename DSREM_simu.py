"""
Simulation studies of DSREM pipeline
Usage: python ./DSREM_simu.py ./data/ ./result/ 10 5

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2018-03-04
"""

import numpy as np
import sys
import nibabel as nib
from lmm import lmm_fit
from mrf import mrf_map

"""
installed all the libraries above
"""

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    em_iter = int(sys.argv[3])
    map_iter = int(sys.argv[4])

    """+++++++++++++++++++++++++++++++++++"""
    print("""\n Load dataset & preprocessing \n """)
    info_file_name = input_dir + "/info.txt"
    dat = np.loadtxt(info_file_name)        # including subject index, gender, visit time (normalized), diagnostic info.
    m = dat.shape[0]                        # the total number of observations
    nidx = dat[:, 0].reshape(m, 1)          # subject index
    n = nidx[m - 1, 0]                      # number of subject
    x = dat[:, 1:2].reshape(m, 2)           # gender and visit time
    p = x.shape[1]
    dx = dat[:, 3].reshape(m, 1)            # diagnostic information
    x0 = x[np.nonzero(dx == 0), :]          # covariates for normal controls
    x1 = x[np.nonzero(dx == 1), :]          # covariates for patients
    nidx0 = nidx[np.nonzero(dx == 0), :]    # subject index for normal controls
    nidx1 = nidx[np.nonzero(dx == 1), :]    # subject index for patients
    mask_file_name = input_dir + "/mask.txt"
    mask = np.loadtxt(mask_file_name)
    sub = np.array(np.nonzero(mask == 1))
    n_v, d = sub.shape
    l, h = mask.shape
    img_file_name = input_dir + "/y.nii.gz"
    img = nib.load(img_file_name)
    img_mat = img.get_data()
    y = np.zeros(shape=(m, n_v))
    for j in range(n_v):
        y[:, j] = img_mat[sub[j, 0], sub[j, 1], :]
    y0 = y[np.nonzero(dx == 0), :]          # imaging intensity for normal controls
    y1 = y[np.nonzero(dx == 1), :]          # imaging intensity for patients

    """+++++++++++++++++++++++++++++++++++"""
    print("""\n Fit voxel-wised linear mixed model (lmm) \n """)
    beta = np.zeros(shape=(p+1, n_v))           # array for estimated fixed effect
    sigma = np.zeros(shape=(p+1, p+1, n_v))     # array for estimated covariance matrix of random effects
    scale = np.zeros(shape=(1, n_v))            # array for estimated scale parameter in error part
    omega = np.zeros(shape=(int(sum(dx)), n_v))
    for j in range(n_v):
        y0_data = y0[:, j].reshape(y0.shape[0], 1)
        beta[:, j], sigma[:, :, j], scale[0, j] = lmm_fit(x0, y0_data, nidx0)
        for i in range(omega.shape[0]):
            omega[i, j] = scale[0, j] * np.dot(np.dot(x1[i, :].reshape(1, -1), np.squeeze(sigma[:, :, j])),
                                               x1[i, :].reshape(-1, 1)) + scale[0, j]
    inv_omega = 1/omega

    """+++++++++++++++++++++++++++++++++++"""
    print("""\n Set the initial guess of disease regions and related additional effects \n """)
    label0 = 0*y1
    sub1_id = np.unique(nidx1)
    n1 = len(sub1_id)
    cum_idx = 0
    for i in range(n1):
        idx_i = np.nonzero(nidx1 == sub1_id[i])[0]
        x1_i = x1[idx_i, :]
        y1_i = y1[idx_i, :]
        ey1_i = np.dot(x1_i, beta)
        res_y1_i = y1_i-ey1_i
        threshold = np.percentile(res_y1_i, 5, axis=1)
        for t in range(len(idx_i)):
            data = res_y1_i[i, :].reshape(-1, 1)
            threshold_idx = np.nonzero(data <= threshold[t])[0]
            label0[cum_idx+t, threshold_idx] = 1
        cum_idx = cum_idx+len(idx_i)
    alpha0 = -0.2*beta
    mu0 = np.dot(x1, alpha0)

    """+++++++++++++++++++++++++++++++++++"""
    print("""\n Estimate the disease regions and related additional effects \n """)
    tau, eta = [0.3, 0.3]
    res_y1 = y1-np.dot(x1, beta)   # n1 * n_v
    for k in range(em_iter):
        print("iteration %d \n" % (k+1))
        label0 = mrf_map(label0, res_y1, mask, sub, mu0, inv_omega, tau, eta, map_iter)
        res_y1_b = res_y1 * label0 * inv_omega
        for j in range(n_v):
            alpha0[:, j] = np.dot(np.dot(np.dot(np.transpose(x1), np.diag(inv_omega[:, j])), x1),
                                  np.dot(np.transpose(x1), res_y1_b[:, j].reshape(-1, 1)))
        mu0 = np.dot(x1, alpha0)

    """++++++++++++++++++++++++++++++++++++"""
    print("""\n Save results\n """)
    label_file_name = output_dir + "label.txt"
    np.savetxt(label_file_name, label0)
    bbar_file_name = output_dir + "alpha.txt"
    np.savetxt(bbar_file_name, alpha0[:, :])
