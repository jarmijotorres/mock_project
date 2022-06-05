from halotools.mock_observables import npairs_xy_z
from halotools.mock_observables import marked_npairs_xy_z
import numpy as np
from glob import glob
import sys
from mpi4py import MPI
comm =MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

M = sys.argv[1]
redshift = sys.argv[2]
p=0.5#float(sys.argv[2])

npi=50
nsi=10
pimax=80
sigma = np.logspace(np.log10(0.5),np.log10(50.),nsi+1)
rp = 10**(np.log10(sigma[:-1]) + np.diff(np.log10(sigma))[0]/2.)
pi = np.logspace(np.log(0.5),np.log(pimax),npi+1,base=np.e)
#pi = np.linspace(0,pimax,npi+1)
pb = 10**(np.log10(pi[:-1]) + np.diff(np.log10(pi))[0]/2.)
#dpi = np.diff(pi)[0]
#pb = pi[:-1] + dpi/2.

dlogpi = np.diff(np.log(pi))[0]
#dlpi = np.mean(dlogpi)

def Marked_auto_correlation_function_box(C1,m1):
    
    m1_bar = np.mean(m1)

    DD_below_par = npairs_xy_z(sample1=C1[:,(0,1,2)],sample2=C1[:,(0,1,2)],rp_bins=sigma,pi_bins=pi,period=None,num_threads=28)
    DD = np.diff(np.diff(DD_below_par,axis=1),axis=0)/2.0

    WW_below_par =  marked_npairs_xy_z(sample1=C1[:,(0,1,2)],sample2=C1[:,(0,1,2)],rp_bins=sigma,pi_bins=pi,period=None,weights1=m1,weights2=m1,weight_func_id=1,num_threads=28)
    WW = np.diff(np.diff(WW_below_par,axis=1),axis=0)/2.0
    
    Lbox = 768
    NR=10.0
    #R0 = Lbox*np.random.random(size=(int(NR)*len(C0),3))
    R1 = Lbox*np.random.random(size=(int(NR)*len(C1),3))
    RR_below_par = npairs_xy_z(sample1=R1[:,(0,1,2)],sample2=R1[:,(0,1,2)],rp_bins=sigma,pi_bins=pi,period=None,num_threads=28)
    RR = np.diff(np.diff(RR_below_par,axis=1),axis=0)/(2.0*(NR*NR))

    xi_w = (WW)/(m1_bar*m1_bar*RR).astype(float) - 1
    wp_w = 2*np.sum(xi_w*pb,axis=1)*dlogpi
    
    xi_nw = (DD)/RR.astype(float) - 1
    wp_nw = 2*np.sum(xi_nw*pb,axis=1)*dlogpi
    
    MCF = (1 + wp_w/rp)/(1 + wp_nw/rp)
    
    return MCF

dir_marks = '/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/marks/'
theta_list = np.sort(glob(dir_marks+'V2D_Galaxy_'+M+'_'+redshift+'_*_MCMCfit_20_slices_ssize38.4_sspace.npy'))

list_all = np.array_split(theta_list,size)

list_chunk = list_all[rank]

for l in list_chunk:
    
    cat_name = l.split('/')[-1]
    theta = np.array(cat_name.split('_')[6:11],dtype=float)
    C1 = np.load(l)
    m1 = C1[:,3]**-p
    MCF_sim = Marked_auto_correlation_function_box(C1,m1)

    S = np.array([rp,MCF_sim]).T
    np.savetxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/MCF/MCF_sspace_'+M+'_z'+redshift+'_theta_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf'%tuple(theta)+'_mV2D20s_p-0.5.dat',S)


