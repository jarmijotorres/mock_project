import numpy as  np
from glob import glob
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from mpi4py import MPI
import sys,h5py

comm =MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

#C0 = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/survey_galaxies/galaxy_LOWZ_North_z0.24_0.36.dat')
#R0 = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/survey_galaxies/random0_LOWZ_North_z0.24_0.36.dat')

#Vs = 404001656.8356064#survey volume

npi=100
nsi=20
sigma = np.logspace(np.log10(0.5),np.log10(50.),nsi+1)
pi = np.linspace(0,100.,npi+1)
dpi = np.diff(pi)[0]
s_l = np.log10(sigma[:-1]) + np.diff(np.log10(sigma))[0]/2.
rp = 10**s_l

NR = 50.0#len(R0)/len(C0)

limit_areas = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/survey_galaxies/limits_subvolumes.txt')

l=sys.argv[1]#'/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/marked_catalogues/V2D_8_slices_dz_0.015_galaxy_LOWZ_North_z0.24_0.36.hdf5.hdf5'
cat_name = l.split('/')[-1]

lr=sys.argv[2]#'/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/survey_catalogues/random0_LOWZ_North_z0.24_0.36.hdf5'

C0_data = h5py.File(l,'r')
R0_data = h5py.File(lr,'r')
C0_all = C0_data['data']
R0_all = R0_data['data']

cond1 = (C0_all['RA']>limit_areas[rank][0])&(C0_all['RA']<limit_areas[rank][1])&(C0_all['DEC']>limit_areas[rank][2])&(C0_all['DEC']<limit_areas[rank][3])
cond2 = (R0_all['RA']>limit_areas[rank][0])&(R0_all['RA']<limit_areas[rank][1])&(R0_all['DEC']>limit_areas[rank][2])&(R0_all['DEC']<limit_areas[rank][3])

C0 = C0_all[~cond1]
R0 = R0_all[~cond2]
#JK_gal_list = #np.sort(glob('/cosma7/data/dp004/dc-armi2/SDSS_subsamples/JK_galaxy_LOWZ_NGC_field*.dat'))
#JK_ran_list = #np.sort(glob('/cosma7/data/dp004/dc-armi2/SDSS_subsamples/JK_random0_LOWZ_NGC_field*.dat'))

#NJK=len(JK_gal_list)

#wp_JK_run = []
#n_JK_run = []

#C0 = np.loadtxt(JK_gal_list[rank])
#R0 = np.loadtxt(JK_ran_list[rank])
#
D1D2_est = DDrppi_mocks(autocorr=True,cosmology=2,nthreads=28,pimax=80,binfile=sigma,RA1=C0['RA'],DEC1=C0['DEC'],CZ1=C0['Z'],weights1=np.ones_like(C0['weight']),weight_type='pair_product')

D1R2_est = DDrppi_mocks(autocorr=False,cosmology=2,nthreads=28,pimax=80,binfile=sigma,RA1=C0['RA'],DEC1=C0['DEC'],CZ1=C0['Z'],weights1=np.ones-like(C0['weight']),RA2=R0['RA'],DEC2=R0['DEC'],CZ2=R0['Z'],weights2=R0['weight'],weight_type='pair_product')

R1R2_est = DDrppi_mocks(autocorr=True,cosmology=2,nthreads=28,pimax=80,binfile=sigma,RA1=R0['RA'],DEC1=R0['DEC'],CZ1=R0['Z'],weights1=R0['weight'],weight_type='pair_product')

D1D2 = D1D2_est['npairs']*D1D2_est['weightavg']
R1R2 = R1R2_est['npairs']*R1R2_est['weightavg'] / (NR*NR)
D1R2 = D1R2_est['npairs']*D1R2_est['weightavg'] / NR

xi2D = (D1D2 - 2*D1R2 + R1R2)/R1R2
xi_pi_sigma = np.zeros((npi,nsi))
for j in range(nsi):
    for i in range(npi):
        xi_pi_sigma[i,j] = xi2D[i+j*npi] 

wp_JK = 2*np.sum(xi_pi_sigma,axis=0)
n_JK = len(C0) / Vs

comm.barrier()

wp_JK_all = comm.gather(wp_JK,root=0)
n_JK_all = comm.gather(n_JK,root=0)
if rank == 0:
    print("gathering and saving infromation from jobs")
    #np.save('/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/clustering/all_wp_JK25.npy',wp_JK_all)
    wp_JK_mean = np.mean(wp_JK_all,axis=0)
    wp_JK_std = np.sqrt(NJK - 1)*np.std(wp_JK_all,ddof=1,axis=0)
    n_JK_mean = np.mean(n_JK_all)
    n_JK_std = np.std(n_JK_all)
    S = np.array([rp,wp_JK_mean,wp_JK_std]).T
    Sn = np.array([n_JK_mean,n_JK_std])
    np.savetxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/clustering/JK25_wp_'+cat_name+'_logrp0.5_50_10bins_pimax80_z0.24_0.36.txt',S)
    np.savetxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/number_density/JK25_numberDensity'+cat_name+'_z0.24_0.36.txt',newline=' ')
    print('End of program.')



