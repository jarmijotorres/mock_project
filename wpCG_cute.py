import numpy as np
import sys 

l = sys.argv[1] # name should be like '/cosma/home/dp004/dc-armi2/sdsswork/outputs_pipelines/data/CODEX50p_LOWZ_3Dps_c2pcf_North_logr1.2_50_10bins.dat'

xi2d_CODEX = np.loadtxt(l)
#### please force the 3D_ps output to be a square in shape
Nbins_pi = 100
Nbins_sigma = int(len(xi2d_CODEX)/Nbins_pi)
pi_data = xi2d_CODEX[:Nbins_pi,0]
sigma_data = xi2d_CODEX[::Nbins_pi,1]
xi_pi_sigma = np.zeros((Nbins_pi,Nbins_sigma))
for j in range(Nbins_sigma):
    for i in range(Nbins_pi):
        xi_pi_sigma[i,j] = xi2d_CODEX[i+j*Nbins_pi,2]
#

wp_CODEX_LOWZ = 2*np.sum(xi_pi_sigma,axis=0)

np.savetxt(l.split('CODEX')[0]+'wp_'+l.split('/')[-1],np.array([sigma_data,wp_CODEX_LOWZ ]).T)