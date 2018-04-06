"""
Estimating the hidden variables, b_ij(v), v = 1,2,..., num_vox, based on ICM algorithm
Author: Chap Huang  (chaohuang.stat@gmail.com)
Last update: 2018-02-18
"""

import numpy as np
"""
installed all the libraries above
"""


def sub2ind(array_shape, sub):
    """ Function to transfer subscripts of neighbor voxels to index """

    if len(array_shape) == 2:
        if sub[0] >= (array_shape[0]-1):
            ind2_1 = -1
        else:
            ind2_1 = (sub[0] + 1) * array_shape[1] + sub[1]
        if sub[1] >= (array_shape[1]-1):
            ind2_2 = -1
        else:
            ind2_2 = sub[0] * array_shape[1] + sub[1] + 1
        if sub[0] == 0:
            ind2_3 = -1
        else:
            ind2_3 = (sub[0] - 1) * array_shape[1] + sub[1]
        if sub[1] == 0:
            ind2_4 = -1
        else:
            ind2_4 = sub[0] * array_shape[1] + sub[1] - 1
        tmp2 = np.array([ind2_1, ind2_2, ind2_3, ind2_4])
        ind = tmp2[np.nonzero(tmp2 != -1)[0]]
    else:
        if sub[0] >= (array_shape[0]-1):
            ind3_1 = -1
        else:
            ind3_1 = (sub[0] + 1)*array_shape[1]*array_shape[2] + sub[1]*array_shape[1] + sub[2]
        if sub[1] >= (array_shape[1]-1):
            ind3_2 = -1
        else:
            ind3_2 = sub[0]*array_shape[1]*array_shape[2] + (sub[1] + 1)*array_shape[1] + sub[2]
        if sub[2] >= (array_shape[2]-1):
            ind3_3 = -1
        else:
            ind3_3 = sub[0]*array_shape[1]*array_shape[2] + sub[1]*array_shape[1] + sub[2] + 1
        if sub[0] == 0:
            ind3_4 = -1
        else:
            ind3_4 = (sub[0] - 1)*array_shape[1]*array_shape[2] + sub[1]*array_shape[1] + sub[2]
        if sub[1] == 0:
            ind3_5 = -1
        else:
            ind3_5 = sub[0]*array_shape[1]*array_shape[2] + (sub[1] - 1)*array_shape[1] + sub[2]
        if sub[2] == 0:
            ind3_6 = -1
        else:
            ind3_6 = sub[0]*array_shape[1]*array_shape[2] + sub[1]*array_shape[1] + sub[2] - 1
        tmp3 = np.array([ind3_1, ind3_2, ind3_3, ind3_4, ind3_5, ind3_6])
        ind = tmp3[np.nonzero(tmp3 != -1)[0]]

    return ind


def delta_fun(x1, x2):
    """ Delta function """

    d = len(np.asarray(x2))
    z = 0 * x2
    for i in range(d):
        if x1 == x2[i]:
            z[i] = 0
        else:
            z[i] = 1
    return sum(z)


def mrf_map(label0, res_y1, mask, coord_mat, mu, inv_s, tau, eta, map_iter):
    """ MRF-MAP estimation procedure """

    n1, n_v, m = res_y1.shape
    n_class = 2

    u1 = np.zeros(shape=(n_v, n_class))
    u2 = np.zeros(shape=(n_v, n_class))

    for ii in range(n1):
        r_i = np.squeeze(res_y1[ii, :, :])
        mu_i = mu[ii, :]
        label0_i = label0[ii, :]
        for jj in range(map_iter):
            for kk in range(n_v):
                u1[kk, 0] = 0.5 * np.sum(np.dot(np.reshape(r_i[kk, :], newshape=(1, m)),
                                                np.squeeze(inv_s[kk, :, :])) ** 2)
                u1[kk, 1] = 0.5 * np.sum(np.dot(np.reshape(r_i[kk, :] - mu_i, newshape=(1, m)),
                                                np.squeeze(inv_s[kk, :, :])) ** 2)
                sub = coord_mat[kk, :]
                ind = sub2ind(mask.shape, sub)
                u2[kk, 0] = delta_fun(0, label0_i[ind])
                u2[kk, 1] = delta_fun(1, label0_i[ind])
            u = u1 + u2*tau + u2*eta
            label0_i = np.argmin(u, axis=1)

        label0[ii, :] = label0_i

    """return label0, mu"""
    return label0
