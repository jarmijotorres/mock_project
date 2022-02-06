from halotools.mock_observables import npairs_xy_z
from halotools.mock_observables import marked_npairs_xy_z
import numpy as np
import sys
from mpi4py import MPI
comm =MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

theta_sel = np.array([[13.1014074 , 14.05650187, 13.14808593,  0.16010931,  1.01105679],
       [13.09970929, 13.80368504, 13.15077829,  0.16039191,  1.01148704],
       [13.09924124, 13.9793665 , 13.15012907,  0.11753749,  1.00963437],
       [13.09913975, 13.9782508 , 13.3203971 ,  0.16084605,  1.01067659],
       [13.10100118, 13.98072348, 13.15061422,  0.30843286,  1.01004056],
       [13.03974548, 13.98103923, 13.14968951,  0.16010025,  1.01232983],
       [13.09956005, 13.97959469, 13.15035389,  0.22275436,  1.00868381],
       [13.09981537, 13.83745709, 13.1483648 ,  0.15955911,  1.00859748],
       [13.11537517, 13.98080958, 13.14969547,  0.16049704,  1.01003793],
       [13.09901306, 13.97795757, 13.14993893,  0.15908774,  1.12755062],
       [13.0982001 , 13.9823092 , 13.15150962,  0.16056192,  0.9831706 ],
       [13.1007504 , 14.02627264, 13.15143247,  0.15948044,  1.00966563],
       [13.10087412, 13.9808693 , 13.15090725,  0.15938028,  1.01271168],
       [13.10023989, 13.98138193, 13.15028858,  0.18116897,  1.00994535],
       [13.11142288, 13.98005271, 13.15089528,  0.16027899,  1.01032894],
       [13.09767347, 13.97966545, 13.1488371 ,  0.2583546 ,  1.01070151],
       [13.09963582, 13.98124121, 13.150848  ,  0.08154743,  1.01019793],
       [13.1008794 , 13.97848173, 13.12295887,  0.15976516,  1.01125107],
       [13.09916097, 13.98077373, 13.05172548,  0.16060692,  1.01194327]])

npi=100
nsi=10
pimax=100
sigma = np.logspace(np.log10(0.5),np.log10(50.),nsi+1)
pi = np.logspace(np.log(0.5),np.log(pimax),npi+1,base=np.e)
pb = 10**(np.log10(pi[:-1]) + np.diff(np.log10(pi))[0]/2.)
dlogpi = np.diff(np.log(pi))
dlpi = np.mean(dlogpi)
rp = 10**(np.log10(sigma[:-1]) + np.diff(np.log10(sigma))[0]/2.)

def Marked_auto_correlation_function_box(C1,m1):
    
    m1_bar = np.mean(m1)

    DD_below_par = npairs_xy_z(sample1=C1[:,(0,1,2)],sample2=C1[:,(0,1,2)],rp_bins=sigma,pi_bins=pi,period=None,num_threads=16)
    DD = np.diff(np.diff(DD_below_par,axis=1),axis=0)/2.0

    WW_below_par =  marked_npairs_xy_z(sample1=C1[:,(0,1,2)],sample2=C1[:,(0,1,2)],rp_bins=sigma,pi_bins=pi,period=None,weights1=m1,weights2=m1,weight_func_id=1,num_threads=16)
    WW = np.diff(np.diff(WW_below_par,axis=1),axis=0)/2.0
    
    Lbox = 768
    NR=10.0
    #R0 = Lbox*np.random.random(size=(int(NR)*len(C0),3))
    R1 = Lbox*np.random.random(size=(int(NR)*len(C1),3))
    RR_below_par = npairs_xy_z(sample1=R1[:,(0,1,2)],sample2=R1[:,(0,1,2)],rp_bins=sigma,pi_bins=pi,period=None,num_threads=16)
    RR = np.diff(np.diff(RR_below_par,axis=1),axis=0)/(2.0*(NR*NR))

    xi_w = (WW)/(m1_bar*m1_bar*RR).astype(float) - 1
    wp_w = 2*np.sum(xi_w*pb,axis=1)*dlpi
    
    xi_nw = (DD)/RR.astype(float) - 1
    wp_nw = 2*np.sum(xi_nw*pb,axis=1)*dlpi
    
    MCF = (1 + wp_w/rp)/(1 + wp_nw/rp)
    
    return MCF

#l = sys.argv[1]
#theta = sys.argv[2]
theta = theta_sel[rank]
#cat_name = l.split('/')[-1]
l = '/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/V2D_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.20_slices_ssize38.4.vol'%(tuple(theta))
C1 = np.loadtxt(l)
m1 = C1[:,3]**-0.5

MCF_sim = Marked_auto_correlation_function_box(C1,m1)

S = np.array([rp,MCF_sim]).T
np.savetxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/markedclustering/HOD_family/MCF_theta_%.5lf_%.5lf_%.5lf_%.5lf_%.5lf.dat'%tuple(theta),S)