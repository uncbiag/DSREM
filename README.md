# DSREM
Dynamic Spatial Random Effects Model

Our dynamic spatial random effects model (DSREM) is a Python package consisting of a spatial random effects models (SREM) [Besag, 1974, Diggle and Ribeiro, 2007, Geman and Geman, 1984, Huang et al., 2015, Li and Singh, 2009] and a dynamic conditional random field (DCRF) model [Sutton et al., 2007, Wang and Ji, 2005, Wang et al., 2006, Yin et al., 2009].

To test the code, one toy dataset is provided which can be found via the link 
https://1drv.ms/f/s!AkpG5buRSVXjh88b16hjXCYrFJiFtg

DSREM_simu_s1.py and DSREM_simu_s2.py are the demo codes for testing the toy dataset. In particular, 

DSREM_simu_s1.py: run linear mixed model acorss all voxels and normal subjects

DSREM_simu_s2.py: run dynamic markov random field to derive the diseased region for each patient acoss time.
