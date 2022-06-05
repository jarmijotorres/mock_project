import numpy as np
import subprocess
import h5py
import sys,time
sys.path.append('/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/src/')
from hod import *
from tpcf_obs import *
from chi2 import *
from survey_geometry import *

#from mpi4py import MPI
#comm =MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()

#if rank == 0:
#t =time.time()

M=sys.argv[1]#'GR'
Lbox = 768
redshift = float(sys.argv[2])

if redshift == 0.3:
    z_l = 0.24
    z_h = 0.36
    mask_IDs = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/survey_catalogues/mask/randoms_LOWZ_North_NS1024.mask')
    survey_dp = np.load('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/survey_catalogues/edges_12timesNgal_LOWZ_North_ran_RA_DEC_z0.24_0.36.npy')
    thetas = np.load('/cosma7/data/dp004/dc-armi2/mcmc_runs/'+M+'_An0.5_Awp0.5/confidence_interval/HODparams_DeltaChi2_1sigma_ConfInt_LOWZ.npy')
    V_survey = 419757894.7368421#Volume from survey
    Ns = 16 #Number of slices to tessellate 
    
else:
    z_l = 0.46
    z_h = 0.54
    mask_IDs = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/survey_catalogues/mask/randoms_CMASS_North_NS1024.mask')
    survey_dp = np.load('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/survey_catalogues/edges_10timesNgal_CMASS_North_ran_RA_DEC_z0.46_0.54.dat.npy')
    thetas = np.load('/cosma7/data/dp004/dc-armi2/mcmc_runs/'+M+'_An0.5_Awp0.5/confidence_interval/HODparams_DeltaChi2_1sigma_ConfInt_CMASS.npy')
    V_survey = 660246219.5467759
    Ns = 9

haloes_table = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/Haloes_MG-Gadget_'+M+'_z'+str(redshift)+'_L'+str(Lbox)+'_ID_M200c_R200c_pos_vel_Nsh_FirstSh_SubHaloList_SubHaloMass_SubHaloVel_logMmin_11.2.0.hdf5','r')

haloes = np.array(haloes_table['MainHaloes'])
subhaloes = np.array(haloes_table['SubHaloes'])


np.random.shuffle(thetas)
theta_randoms = thetas[:100]
                                
theta_chunk = np.round(theta_randoms,decimals=5)

#theta_split = np.array_split(theta,size)

#theta_chunk = theta_split[rank]
n_chunk = np.zeros(len(theta_chunk))
wp_chunk = np.zeros((len(theta_chunk),13))
MCF_chunk = np.zeros((len(theta_chunk),10))
for i,t in enumerate(theta_chunk):
    G0 = HOD_mock_subhaloes(t,haloes,subhaloes,Lbox)
    wp_sim = wp_from_box(G0,n_threads=64,Lbox=Lbox,Nsigma=13,return_rpavg=False)
    #move to LC to calculate the marked correlation function
    LC_G0 = box_to_lightcone(G0,mask_IDs,z_l,z_h,is_pec_vel=False)
    n_sim = LC_G0.shape[0]/V_survey
    G1_V2D = survey_tessellation_MT(Ns=Ns,z_l=z_l,z_h=z_h,G0=LC_G0,survey_dp=survey_dp)
    cat_File = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/marks/V2D_Galaxy_{0}_z{1}_HOD_{2:.5f}_{3:.5f}_{4:.5f}_{5:.5f}_{6:.5f}_RA_DEC_Zobs_weight_mark1.hdf5'.format(M,redshift,t[0],t[1],t[2],t[3],t[4]),'w')
    cat_File.create_dataset('RA',data=G1_V2D[:,0])
    cat_File.create_dataset('DEC',data=G1_V2D[:,1])
    cat_File.create_dataset('Z',data=G1_V2D[:,2])
    cat_File.create_dataset('weight',data=G1_V2D[:,3])
    cat_File.create_dataset('mark1',data=G1_V2D[:,4]**-0.5)
    cat_File.close()
    
    create_parfile_twopcf(M,z_l,z_h,redshift,t)
    
    subprocess.run(['/cosma/home/dp004/dc-armi2/two_pcf/TWOPCF','/cosma7/data/dp004/dc-armi2/HOD_mocks/twopcf/twopcf_unmarked_HOD_LC_z'+str(redshift)+'.ini'])
    subprocess.run(['/cosma/home/dp004/dc-armi2/two_pcf/TWOPCF','/cosma7/data/dp004/dc-armi2/HOD_mocks/twopcf/twopcf_marked_HOD_LC_z'+str(redshift)+'.ini'])
    
    xi2d_m = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/twopcf_ouputs/TWOPCF_xi2D_marked_{0}_{1}_LC_HOD_{2:.5f}_{3:.5f}_{4:.5f}_{5:.5f}_{6:.5f}_logrp_0.5_50_10logrpbins_logpi_0.5_80_50pibins_z{7:.2f}_{8:.2f}.hdf5'.format(M,redshift,t[0],t[1],t[2],t[3],t[4],z_l,z_h),'r')
    xi2d_um = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/twopcf_ouputs/TWOPCF_xi2D_unmarked_{0}_{1}_LC_HOD_{2:.5f}_{3:.5f}_{4:.5f}_{5:.5f}_{6:.5f}_logrp_0.5_50_10logrpbins_logpi_0.5_80_50pibins_z{7:.2f}_{8:.2f}.hdf5'.format(M,redshift,t[0],t[1],t[2],t[3],t[4],z_l,z_h),'r')
    
    MCF_sim = MCF_from_tpcf(xi2d_m,xi2d_um)
    
    n_chunk[i] = n_sim
    wp_chunk[i] = wp_sim
    MCF_chunk[i] = MCF_sim
    
#comm.barrier() 

#n_all = comm.gather(n_chunk,root=0)
#wp_all = comm.gather(wp_chunk,root=0)
#MCF_all = comm.gather(MCF_chunk,root=0)

#if rank == 0:
#e_t = time.time() - t
T = h5py.File('/cosma7/data/dp004/dc-armi2/mcmc_runs/'+M+'_An0.5_Awp0.5/confidence_interval/randoms_sample'+M+'_chi_2_1sigma_ConfInt_100_theta_nga_wp_MCF_z'+str(redshift)+'.hdf5','w')
T.create_dataset(name='theta',data=theta_chunk)
T.create_dataset(name='n_gal',data=n_chunk)
T.create_dataset(name='wp_rp',data=wp_chunk)
T.create_dataset(name='MCF',data=MCF_chunk)
T.close()
#print('job done in t: %.2lf'%e_t)
    