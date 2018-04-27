"""
Estimating the hidden variables, b_ij(v), v = 1,2,..., num_vox, based on ICM algorithm
Author: Chap Huang  (chaohuang.stat@gmail.com)
Last update: 2018-02-18
"""

import numpy as np
"""
installed all the libraries above
"""


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


def mrf_map(label0, baseline_id, res_y1, dist, coord_mat, x1, betabar0, s2, sigma, tau, eta, map_iter):
    """ MRF-MAP estimation procedure """

    n1, n_v = res_y1.shape
    n_class = 2

    for ii in range(n1):
        r_i = res_y1[ii, :]
        mu_i = np.dot(x1[ii, :].reshape(1, -1), betabar0.reshape(-1, 1))
        label0_i = label0[ii, :]
        for jj in range(map_iter):
            u1 = np.zeros(shape=(n_v, n_class))
            u2 = np.zeros(shape=(n_v, n_class))
            u3 = np.zeros(shape=(n_v, n_class))
            for kk in range(n_v):
                inv_omega = 1/(s2[kk]*(1+np.dot(np.dot(x1[ii, :].reshape(1, -1),
                                                       np.squeeze(sigma[:, :, kk])), x1[ii, :].reshape(-1, 1))))
                nu_i = r_i[kk].reshape(-1, 1)-mu_i
                u1[kk, 0] = 0.5 * r_i[kk]*inv_omega*r_i[kk]
                u1[kk, 1] = 0.5 * nu_i*inv_omega*nu_i
                dist_k = dist[kk, :]
                ind = np.nonzero(dist_k <= 1.5)[0]
                u2[kk, 0] = delta_fun(0, label0_i[ind])
                u2[kk, 1] = delta_fun(1, label0_i[ind])

                if baseline_id[jj] == 0:
                    u3[kk, 0] = delta_fun(0, label0[ii-1, ind])
                    u3[kk, 1] = delta_fun(1, label0[ii-1, ind])
            u = u1 + u2*tau + u3*eta
            label0_i = np.argmin(u, axis=1)

        label0[ii, :] = label0_i

    """return label0, mu"""
    return label0
