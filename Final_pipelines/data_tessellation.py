import numpy as np
import h5py
import sys,time
sys.path.append('/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/src/')
from hod import *
from tpcf_obs import *
from chi2 import *
from survey_geometry import *

l = sys.argv[1] #'/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/twopcf_catalogues/V2D_16_slices_galaxy_LOWZ_North_z0.24_0.36.hdf5'
cat_name = l.split('/')[-1]
redshift = float(sys.argv[2])
cat = h5py.File(l,'r')
LC_G0 = np.array(list(zip(cat['RA'][:],cat['DEC'][:],cat['Z'][:],cat['weight'][:],cat['JK_ID'][:])),dtype=[('RA',float),('DEC',float),('Z',float),('weight',float),('JK_ID',int)])

if redshift == 0.3:
    z_l = 0.24
    z_h = 0.36
    survey_dp = np.load('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/survey_catalogues/edges_12timesNgal_LOWZ_North_ran_RA_DEC_z0.24_0.36.npy')
    Ns = 8 #Number of slices to tessellate 
    
else:
    z_l = 0.46
    z_h = 0.54
    survey_dp = np.load('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/survey_catalogues/edges_5timesNgal_CMASS_North_ran_RA_DEC_z0.46_0.54.npy')
    Ns = 5

G1_V2D = survey_tessellation_MT(Ns=Ns,z_l=z_l,z_h=z_h,G0=LC_G0,survey_dp=survey_dp)
cat2 = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/twopcf_catalogues/'+cat_name.split('hdf5')[0]+'V2D_%d_zslices.hdf5'%Ns,'a')
cat2.create_dataset('RA',data=G1_V2D['RA'])
cat2.create_dataset('DEC',data=G1_V2D['DEC'])
cat2.create_dataset('Z',data=G1_V2D['Z'])
cat2.create_dataset('weight',data=G1_V2D['weight'])
cat2.create_dataset('JK_ID',data=G1_V2D['JK_ID'])
cat2.create_dataset('V2D_%d_zslices'%Ns,data=G1_V2D['V2D_%d_zslices'%Ns])
cat2.create_dataset('mark_V2D_%d_zslice'%Ns,data=G1_V2D['V2D_%d_zslices'%Ns]**-0.5)
cat2.create_dataset('weight_mark_V2D_%d_zslice'%Ns,data=G1_V2D['weight']*(G1_V2D['V2D_%d_zslices'%Ns]**-0.5))
cat2.close()