from halotools.mock_observables import npairs_xy_z
from halotools.mock_observables import marked_npairs_xy_z
import numpy as np
import sys
from mpi4py import MPI
comm =MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

def Marked_auto_correlation_function_box(C1,m1,Lbox):
    #m0_bar = np.mean(m0)
    m1_bar = np.mean(m1)
    
    NR=10.0
    #R0 = Lbox*np.random.random(size=(int(NR)*len(C0),3))
    R1 = Lbox*np.random.random(size=(int(NR)*len(C1),3))

#pairs
    DD_below_par = npairs_xy_z(sample1=C1[:,(0,1,2)],sample2=C1[:,(0,1,2)],rp_bins=sigma,pi_bins=pi,period=None,num_threads=28)
    DD = np.diff(np.diff(DD_below_par,axis=1),axis=0)/2.0

    RR_below_par = npairs_xy_z(sample1=R1[:,(0,1,2)],sample2=R1[:,(0,1,2)],rp_bins=sigma,pi_bins=pi,period=None,num_threads=28)
    RR = np.diff(np.diff(RR_below_par,axis=1),axis=0)/(2.0*(NR*NR))

    WW_below_par =  marked_npairs_xy_z(sample1=C1[:,(0,1,2)],sample2=C1[:,(0,1,2)],rp_bins=sigma,pi_bins=pi,period=None,weights1=m1,weights2=m1,weight_func_id=1,num_threads=28)
    WW = np.diff(np.diff(WW_below_par,axis=1),axis=0)/2.0

    xi_w = (WW)/(m1_bar*m1_bar*RR).astype(float) - 1
    wp_w = 2*np.sum(xi_w,axis=1)*dpi 
    xi_nw = (DD)/RR.astype(float) - 1
    wp_nw = 2*np.sum(xi_nw,axis=1)*dpi
    MCF = (1 + wp_w/s_l)/(1 + wp_nw/s_l)
    
    return MCF

M = sys.argv[1]
Lbox = int(sys.argv[2])
#print("mock parameters: Lbox=%d for %s model."%(Lbox,M))

npi=80
nsi=10
sigma = np.logspace(np.log10(0.5),np.log10(50.),nsi+1)
pi = np.linspace(0,100.,npi+1)
dpi = np.diff(pi)[0]
s_l = np.log10(sigma[:-1]) + np.diff(np.log10(sigma))[0]/2.
s_l = 10**s_l
#print("reading 2pcf function parameters...")
    
#l_galaxies = '/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/Galaxy_'+M+'_z0.3_L'+str(Lbox)+'_HOD_'+theta+'_pos_HaloM200c_weight.dat'
#l_clusters = '/cosma7/data/dp004/dc-armi2/HOD_mocks/cluster_catalogues/Cluster_'+M+'_z0.3_L'+str(Lbox)+'_M200c_pos_mainhaloes_logMcut.dat'
#C0 = np.loadtxt(l_clusters,usecols=(0,1,2,3))
#if rank == 0:
#    print("reading data...")
if M == 'GR':
    theta = '13.102_14.077_13.117_0.150_1.011'
elif M == 'F5':
    theta = '13.161_13.982_13.580_0.111_0.991'
if Lbox == 768: 
    s = 20
elif Lbox == 1536: 
    s = 40
l_galaxy_marks = '/cosma7/data/dp004/dc-armi2/HOD_mocks/marks/V2D_Galaxy_'+M+'_z0.3_L'+str(Lbox)+'_HOD_'+theta+'_pos_HaloM200c_weight.dat.'+str(s)+'_slices_ssize38.4.vol'
C1 = np.loadtxt(l_galaxy_marks,usecols=(0,1,2,3))
#m0 = np.ones_like(C0[:,3])#**0.1
m1 = C1[:,3]**-0.5
#    print("data read at rank 0. Distributing...")
#else:
#    C1 = None
#    m1 = None    
    
#C1 = comm.bcast(C1,root=0)
#m1 = comm.bcast(m1,root=0)

if rank == 0:
    print("Jackknife resampling...")
    
Nsub = 4
cell_size = Lbox / Nsub
JK_ID = (C1[:,(0,1,2)] /cell_size).astype(int)
jkid_l = JK_ID[:,0]*Nsub*Nsub + JK_ID[:,1]*Nsub + JK_ID[:,2]

JK_run = np.zeros((Nsub**3,nsi))
ID_run = np.arange(0,Nsub**3,dtype=int).reshape((Nsub**3,1))

size_per_job = int(Nsub**3/size)
JK_chunk = np.zeros((size_per_job,nsi),dtype=float)
ID_chunk = np.zeros((size_per_job,1),dtype=int)

comm.Scatter(JK_run,JK_chunk,root=0)
comm.Scatter(ID_run,ID_chunk,root=0)
for l_id,IDi in enumerate(ID_chunk):
    print("rank %d calculating run number %d ..."%(rank,l_id))
    C1_chunk = C1[jkid_l != IDi]
    m1_chunk = m1[jkid_l != IDi]
    MCF_JK = Marked_auto_correlation_function_box(C1_chunk,m1_chunk,Lbox)
    JK_chunk[l_id] = MCF_JK
    JK_run[IDi] = MCF_JK

comm.Barrier()

if rank == 0:
    print("all ranks done...")
    print('reducing data from jobs.')
    JK_total = np.zeros_like(JK_run)
    
else:
    JK_total = None
    
comm.Reduce([JK_run,MPI.DOUBLE], [JK_total,MPI.DOUBLE],op=MPI.SUM,root=0)

if rank == 0:
    MCF_mean = np.mean(JK_total,axis=0)
    MCF_std = np.sqrt(Nsub**3 - 1)*np.std(JK_total,axis=0)
    print("file saved by rank %d"%(rank))
    S = np.array([s_l,MCF_mean,MCF_std]).T
    np.savetxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/markedclustering/resampling/M_rp_galaxies_'+M+'_z0.3_L'+str(Lbox)+'_rp_0.5_50_10rpbins_JK64errors.dat',S)
    np.savetxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/markedclustering/resampling/list_JK64_M_rp_galaxies_'+M+'_z0.3_L'+str(Lbox)+'_rp_0.5_50_10rpbins.dat',JK_total)
    print("end of program.")
#D1R2_below_par = npairs_xy_z(sample1=C0[:,(0,1,2)],sample2=R0[:,(0,1,2)],rp_bins=sigma,pi_bins=pi,period=None,num_threads=16)
#D1R2 = np.diff(np.diff(D1R2_below_par,axis=1),axis=0)/NR

#D2R1_below_par = npairs_xy_z(sample1=G0[:,(0,1,2)],sample2=R1[:,(0,1,2)],rp_bins=sigma,pi_bins=pi,period=None,num_threads=16)
#D2R1 = np.diff(np.diff(D2R1_below_par,axis=1),axis=0)/NR

#W1R2_below_par =  marked_npairs_xy_z(sample1=C0[:,(0,1,2)],sample2=R0[:,(0,1,2)],rp_bins=sigma,pi_bins=pi,period=None,weights1=m1,weights2=np.ones(len(R0)),weight_func_id=1,num_threads=16)
#W1R2 = np.diff(np.diff(W1R2_below_par,axis=1),axis=0)/(NR)

#W2R1_below_par =  marked_npairs_xy_z(sample1=G0[:,(0,1,2)],sample2=R1[:,(0,1,2)],rp_bins=sigma,pi_bins=pi,period=None,weights1=m0,weights2=np.ones(len(R1)),weight_func_id=1,num_threads=16)
#W2R1 = np.diff(np.diff(W2R1_below_par,axis=1),axis=0)/(NR)


#LS estimator
#xi_w = (W1W2 - W1R2*m0_bar - W2R1*m1_bar + (m1_bar*m0_bar*R1R2))/(m1_bar*m0_bar*R1R2).astype(float)
#wp_w = 2*np.sum(xi_w,axis=1)*dpi
#xi_nw = xi_nw = (D1D2 - D1R2 - D2R1 + R1R2)/R1R2.astype(float)
#wp_nw = 2*np.sum(xi_nw,axis=1)*dpi

#MCF = (1 + wp_w)/(1 + wp_nw)