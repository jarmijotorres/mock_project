from halotools.mock_observables import npairs_xy_z
from halotools.mock_observables import marked_npairs_xy_z
import numpy as np

C1 = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/marks/V2D_Galaxy_GR_z0.3_L768_HOD_13.10_13.98_13.32_0.15_1.01_pos_M200c_weight.dat.20_slices_ssize38.4.vol')

npi=100
nsi=13
pimax=100
sigma = np.logspace(np.log10(0.5),np.log10(50.),nsi+1)
#pi = np.logspace(np.log(0.1),np.log(pimax),npi+1,base=np.e)
pi = np.linspace(0,pimax,npi+1)
dlogpi = np.diff(np.log(pi))
rp = 10**(np.log10(sigma[:-1]) + np.diff(np.log10(sigma))[0]/2.)
#pb = 10**(np.log10(pi[:-1]) + np.diff(np.log10(pi))[0]/2.)
pb = 0.5*(pi[:-1]+pi[1:])


def MCF_log(C1):
    Lbox=768
    ls = Lbox/20.
    n1 = len(C1)/float(Lbox)**3
    V_mean = 1 / (n1*ls)
    m1 = (C1[:,-1]/V_mean)**-0.5
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

    xi_nw = (DD)/(RR).astype(float) - 1
    wp_nw = np.zeros(nsi)#2*np.sum(xi_w,axis=1)*pb*dlogpi 
    for i in range(nsi):
        wp_nw[i] = np.sum(xi_nw[i,:]*pb*dlogpi)
    xi_w = (WW)/(m1_bar*m1_bar*RR).astype(float) - 1
    wp_w = np.zeros(nsi)#2*np.sum(xi_w,axis=1)*pb*dlogpi 
    for i in range(nsi):
        wp_w[i] = np.sum(xi_w[i,:]*pb*dlogpi)

    MCF = (1 + wp_w/rp)/(1 + wp_nw/rp)
    return MCF

def MCF_lin(C1):
    Lbox=768
    ls = Lbox/20.
    n1 = len(C1)/float(Lbox)**3
    V_mean = 1 / (n1*ls)
    m1 = (C1[:,-1]/V_mean)**-0.5
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
    MCF = (1 + wp_w/rp)/(1 + wp_nw/rp)
    
    return MCF