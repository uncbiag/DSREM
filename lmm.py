"""
Step 1: fit voxel-wised linear mixed model (lmm) via EM algorithm
Estimating beta and sigma, v = 1,2,..., num_vox, based on normal subjects.

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2018-03-04
"""

import numpy as np
import statsmodels.api as sm
import pandas as pd

"""
installed all the libraries above
"""


def lmm_fit(x_data, y_data, ni):
    """
        Fit voxel-wised linear mixed model (lmm)

        :param
            x_data (matrix) : ordered design matrix (according to time for each subject, the first column is time).
            y_data (vector): ordered data vector (according to x_data).
            zcols (vector): the indices of the columns of x_data that will be considered as random effects.
            ni (vector) :  subject order list (ordered according to x_data).
        :return
            sm_y (matrix): smoothed response matrix (n*l*m)
            bw_o (matrix): optimal bandwidth (d*m)
    """

    """Setting up"""
    data = np.hstack((ni, y_data, x_data))
    # p = x_data.shape[1]
    # var_names = ["x%d" % (j+1) for j in range(p-1)]
    # col_names = ["subject", "y", "time", var_names]
    col_names = ["subject", "y", "time"]
    df = pd.DataFrame(data, columns=col_names)

    # Fit a mixed model to imaging data with random intercept per subject and random effect on time.
    model_formula = "y ~ time"
    # for j in range(p-1):
    #     model_formula = model_formula + " + x%d" % (j+1)
    # vc = {'time': '1 + C(time)'}
    model = sm.MixedLM.from_formula(model_formula, groups="subject", re_formula="time", data=df)
    result = model.fit()
    beta = np.array(result.fe_params)
    sigma = np.array(result.cov_re)
    scale = np.mean(sigma/np.array(result.cov_re_unscaled))

    """return beta, sigma, scale"""
    return beta, sigma, scale
