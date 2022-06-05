import numpy as  np
import h5py,sys
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from astropy.cosmology import Planck15 as cosmo
from mpi4py import MPI

comm =MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

def dcomov(z):
    '''
    Include little h to obtain Mpc/h units
    '''
    return (cosmo.comoving_distance(z)*cosmo.h).value

def auto_marked_correlation_function(C0,R0,sigma):
    
    dpi = 1.0#np.diff(pi)[0]
    s_l = np.log10(sigma[:-1]) + np.diff(np.log10(sigma))[0]/2.
    rp = 10**s_l
    NR = 50.0#ratio between the lenght of R0 and C0
    V0 = C0['V2D']

    m0 = C0['weight']*V0**-0.25
    m0_bar = np.mean(m0)

    D1D2_est = DDrppi_mocks(autocorr=True,cosmology=2,nthreads=16,pimax=pimax,binfile=sigma,RA1=C0['RA'],DEC1=C0['DEC'],CZ1=C0['Z'],weights1=C0['weight'],weight_type='pair_product')

    D1R2_est = DDrppi_mocks(autocorr=False,cosmology=2,nthreads=16,pimax=pimax,binfile=sigma,RA1=C0['RA'],DEC1=C0['DEC'],CZ1=C0['Z'],weights1=C0['weight'],RA2=R0['RA'],DEC2=R0['DEC'],CZ2=R0['Z'],weights2=R0['weight'],weight_type='pair_product')

    R1R2_est = DDrppi_mocks(autocorr=True,cosmology=2,nthreads=16,pimax=pimax,binfile=sigma,RA1=R0['RA'],DEC1=R0['DEC'],CZ1=R0['Z'],weights1=R0['weight'],weight_type='pair_product')

    D1D2 = D1D2_est['npairs']*D1D2_est['weightavg']
    R1R2 = R1R2_est['npairs']*R1R2_est['weightavg'] / (NR*NR)
    D1R2 = D1R2_est['npairs']*R1R2_est['weightavg'] / NR

    xi2D = (D1D2 - D1R2 - D1R2 + R1R2)/R1R2
    xi_pi_sigma = np.zeros((npi,nsi))
    for j in range(nsi):
        for i in range(npi):
            xi_pi_sigma[i,j] = xi2D[i+j*npi] 

    wp_unmarked = 2*np.sum(xi_pi_sigma,axis=0)*dpi

    W1W2_est = DDrppi_mocks(autocorr=True,cosmology=2,nthreads=16,pimax=pimax,binfile=sigma,RA1=C0['RA'],DEC1=C0['DEC'],CZ1=C0['Z'],weights1=m0,weight_type='pair_product')

    W1R2_est = DDrppi_mocks(autocorr=False,cosmology=2,nthreads=16,pimax=pimax,binfile=sigma,RA1=C0['RA'],DEC1=C0['DEC'],CZ1=C0['Z'],weights1=m0,RA2=R0['RA'],DEC2=R0['DEC'],CZ2=R0['Z'],weights2=np.ones_like(R0['RA']),weight_type='pair_product')

    W1W2 = W1W2_est['npairs']*W1W2_est['weightavg']
    W1R2 = W1R2_est['npairs']*W1R2_est['weightavg']*m0_bar / NR
    R1R2 = R1R2_est['npairs']*m0_bar*m0_bar / (NR*NR)

    xi2D = (W1W2 - W1R2 - W1R2 + R1R2)/R1R2

    xi_pi_sigma = np.zeros((npi,nsi))
    for j in range(nsi):
        for i in range(npi):
            xi_pi_sigma[i,j] = xi2D[i+j*npi]

    wp_marked = 2*np.sum(xi_pi_sigma,axis=0)*dpi

    MCF_JK = (1+wp_marked/rp)/(1+wp_unmarked/rp)
    return MCF_JK

#3D_ps binning
#npi=int(sys.argv[3])
nsi=10
sigma = np.logspace(np.log10(0.5),np.log10(50.),nsi+1)
s_l = np.log10(sigma[:-1]) + np.diff(np.log10(sigma))[0]/2.
rp = 10**s_l
pimax=80
npi = 80
#pi = np.linspace(0,pimax,npi+1)

limit_areas = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/survey_galaxies/limits_subvolumes_25.txt')

l = sys.argv[1]#'/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/marked_catalogues/V2D_8_slices_dz_0.015_galaxy_LOWZ_North_z0.24_0.36.hdf5.hdf5'
cat_name = l.split('/')[-1]
rl = sys.argv[2]#'/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/survey_catalogues/random0_LOWZ_North_z0.24_0.36.hdf5'
C0_data = h5py.File(l,'r')
R0_data = h5py.File(rl,'r')
C0_all = C0_data['data']
R0_all = R0_data['data']

cond1 = (C0_all['RA']>limit_areas[rank][0])&(C0_all['RA']<limit_areas[rank][1])&(C0_all['DEC']>limit_areas[rank][2])&(C0_all['DEC']<limit_areas[rank][3])
cond2 = (R0_all['RA']>limit_areas[rank][0])&(R0_all['RA']<limit_areas[rank][1])&(R0_all['DEC']>limit_areas[rank][2])&(R0_all['DEC']<limit_areas[rank][3])

C0 = C0_all[~cond1]
R0 = R0_all[~cond2]

MCF_JK = auto_marked_correlation_function(C0,R0,sigma)

comm.barrier()

MCF_JK_all = comm.gather(MCF_JK,root=0)

if rank == 0:
    print("gathering and saving infromation from jobs")
    np.save('/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/markedclustering/JKs/all_MCF_JK25_mVp-0.25_'+cat_name+'_logrp%.1lf_%.1lf_%dbins_pimax%d_%dpibins.npy'%(sigma[0],sigma[-1],nsi,pimax,npi),MCF_JK_all)
    MCF_JK_mean = np.mean(MCF_JK_all,axis=0)
    MCF_JK_std = np.sqrt(size - 1)*np.std(MCF_JK_all,ddof=1,axis=0)
    S = np.array([rp,MCF_JK_mean,MCF_JK_std]).T
    #G1_vols_all /= d_box**2
    np.savetxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/markedclustering/JK25_MCF_mVp-0.25_'+cat_name+'_logrp%.1lf_%.1lf_%dbins_pimax%d_%dpibins.txt'%(sigma[0],sigma[-1],nsi,pimax,npi),S)