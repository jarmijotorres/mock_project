import numpy as np
import sys,h5py

Lbox = 1536.

haloes_sim = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/Haloes_MG-Gadget_GR_z0.3_L1536.0_ID_M200c_pos_R200_c200scatter_logMhalomin_11.2.hdf5','r')
haloes = haloes_sim['data']

C0 = haloes[haloes['M200']>10**(14.0)]
C1 = np.array([C0['pos'][:,0],C0['pos'][:,1],C0['pos'][:,2],C0['M200'],np.ones_like(C0['M200'])]).T

np.savetxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/cluster_catalogues/Cluster_GR_z0.3_L1536_M200c_pos_mainhaloes.dat',C1)