"""
Simulation studies of DSREM pipeline (step 2)
Usage: python -W ignore ./DSREM_simu_s2.py ./data/ ./result/ 5 5

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2018-03-04
"""

import numpy as np
import sys
import nibabel as nib
from scipy.io import loadmat
from mrf import mrf_map
from scipy.spatial import distance
import warnings
warnings.simplefilter("ignore")

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
    info_file_name = input_dir + "info.txt"
    dat = np.loadtxt(info_file_name)                        # including subject index, visit time, diagnostic info.
    m = dat.shape[0]                                        # the total number of observations
    nidx = dat[:, 0].reshape(-1, 1)                         # subject index
    n = int(nidx[m - 1, 0])                                 # number of subject
    x = dat[:, 1].reshape(-1, 1)                            # visit time
    p = x.shape[1]
    dx = dat[:, 2].reshape(-1, 1)                           # diagnostic information
    x1 = np.hstack((np.ones(shape=(int(sum(dx)), 1)), x[np.nonzero(dx == 1)[0], :]))
    nidx1 = nidx[np.nonzero(dx == 1)[0], :]                 # subject index for patients
    mask_file_name = input_dir + "mask.txt"
    mask = np.loadtxt(mask_file_name)
    sub = np.array(np.nonzero(mask == 1)).T
    n_v, d = sub.shape
    dist = distance.cdist(sub, sub, 'euclidean')
    l, h = mask.shape
    img_file_name = input_dir + "img.nii"
    img = nib.load(img_file_name)
    img_mat = img.get_data()
    y = np.zeros(shape=(m, n_v))
    for j in range(n_v):
        y[:, j] = img_mat[sub[j, 0], sub[j, 1], :]
    y1 = y[np.nonzero(dx == 1)[0], :]                       # imaging intensity for patients

    """+++++++++++++++++++++++++++++++++++"""
    print("""\n Load result from step 1 \n """)
    beta_file_name = output_dir + "beta.txt"
    beta = np.loadtxt(beta_file_name)
    scale_file_name = output_dir + "scale.txt"
    s2 = np.loadtxt(scale_file_name)
    sigma_file_name = output_dir + "sigma.mat"
    mat = loadmat(sigma_file_name)
    sigma = mat['sigma']

    """+++++++++++++++++++++++++++++++++++"""
    print("""\n Set the initial guess of disease regions and related additional effects \n """)
    label0 = 0*y1
    pid = np.unique(nidx1)
    baseline_id = np.zeros(shape=(x1.shape[0], 1))
    n1 = len(pid)
    for i in range(n1):
        pid_i = np.nonzero(nidx1 == pid[i])[0]
        baseline_id[pid_i[0]] = 1
        m_i = len(pid_i)
        for j in range(m_i-1):
            del_y = y[pid_i[j+1], :]-y[pid_i[j], :]
            threshold = np.percentile(del_y, 3)
            label0[pid_i[j+1], :] = 1*(del_y <= threshold)
        label0[pid_i[0], :] = label0[pid_i[1], :]
    betabar0 = np.array([-0.4, -0.2])

    """+++++++++++++++++++++++++++++++++++"""
    print("""\n Estimate the disease regions and related additional effects \n """)
    tau, eta = [0.3, 0.3]
    res_y1 = y1-np.dot(x1, beta)   # n1 * n_v
    inv_sigma = np.zeros(shape=res_y1.shape)
    for i in range(res_y1.shape[0]):
        for k in range(res_y1.shape[1]):
            inv_sigma[i, k] = 1 / (s2[k]*(1+np.dot(np.dot(x1[i, :].reshape(1, -1),
                                                          np.squeeze(sigma[:, :, k])), x1[i, :].reshape(-1, 1))))
    for k in range(em_iter):
        print("iteration %d \n" % (k+1))
        label0 = mrf_map(label0, baseline_id, res_y1, dist, sub, x1, betabar0, s2, sigma, tau, eta, map_iter)
        # res_y1_b = res_y1 * label0 * inv_omega**0.5
        betabar0 = np.array([-0.4, -0.2])

    """++++++++++++++++++++++++++++++++++++"""
    print("""\n Save results\n """)
    label_file_name = output_dir + "label.txt"
    np.savetxt(label_file_name, label0)
    # bbar_file_name = output_dir + "betabar.txt"
    # np.savetxt(bbar_file_name, betabar0[:, :])
