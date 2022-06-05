import numpy as np
from glob import glob
import h5py,sys
from scipy.stats import binned_statistic_2d,chi2
from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt
import matplotlib as mpl
import corner


M=sys.argv[1]#'GR'
Lbox = 768
#haloes_table = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/Haloes_MG-Gadget_'+M+'_z0.3_L'+str(Lbox)+'_ID_M200c_R200c_pos_vel_Nsh_FirstSh_SubHaloList_SubHaloMass_SubHaloVel_logMmin_11.2.0.hdf5','r')

#haloes = np.array(haloes_table['MainHaloes'])
#subhaloes = np.array(haloes_table['SubHaloes'])

An = 0.5
Awp = 0.5

samples = []
lklhd_samples = []
for i in range(2,40):
    file_chain = '/cosma7/data/dp004/dc-armi2/mcmc_runs/'+M+'_An'+str(An)+'_Awp'+str(Awp)+'/chains/MCMCpost_chains_HOD_'+M+'_L768_500it_28walkers_'+str(An)+'An_'+str(Awp)+'Awp_target_LOWZ_z0.24_0.36_err_1sigma_fullcov_sim_subhaloes_batch_%d.npy'%i

    file_likelihood = '/cosma7/data/dp004/dc-armi2/mcmc_runs/'+M+'_An'+str(An)+'_Awp'+str(Awp)+'/likelihoods/MCMClklhd_chains_HOD_'+M+'_L768_500it_28walkers_'+str(An)+'An_'+str(Awp)+'Awp_target_LOWZ_z0.24_0.36_err_1sigma_fullcov_sim_subhaloes_batch_%d.npy'%i
    
    chains = np.load(file_chain)
    loglikelihood = np.load(file_likelihood)
    samples.append(chains.reshape(chains.shape[0]*chains.shape[1],chains.shape[2]))
    lklhd_samples.append(loglikelihood.T.flatten())
samples = np.concatenate(samples)
lklhd_samples = np.concatenate(lklhd_samples)

max_likelihood = lklhd_samples.max()
chi_2_samples = -2*(lklhd_samples)
chi_min = -2*max_likelihood
Delta_chi_square = chi_2_samples - chi_min


#=============== plot the chi^2 distribution for the run ============#
df=6.0
xc = np.linspace(chi2.ppf(0.001, df),
                chi2.ppf(0.999, df), 100)
mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

f,ax = plt.subplots(1,1,figsize=(6.5,6))
ax.hist(Delta_chi_square,bins=40,range=(0,20),histtype='step',color='r',linewidth=2.0,density=True)
ax.plot(xc, chi2.pdf(xc, df),
       'k-', lw=5, alpha=0.6, label='chi2 pdf')
ax.set_xlim(0,15)
ax.set_ylim(0,0.4)
ax.set_xlabel(r'$\Delta \chi^2$')
ax.set_ylabel(r'PDF')
plt.tight_layout()
#plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/February2022/MCMC_Delta_chi_square_sample_2_5.pdf',bbox_inches='tight')
plt.show()

prior_range = np.array([[12.8,13.5],
             [12.8,14.8],
             [12.8,14.0],
             [0.0,0.6],
             [0.0,1.6]])
labs = [r'$\log M_{min}$',r'$\log M_{1}$',r'$\log M_0$',r'$\sigma$',r'$\alpha$']



#===================== cornerplot plot ===================#
ndim=5
dens_kw = {'plot_density':False,'plot_datapoints':False,'data_kwargs':{'color':'khaki','marker':'o','ms':1.0,'alpha':1.0}}

fig = corner.corner(
    samples,
    labels=labs,
    bins=13,
    #fig=fig,
    #color='k',
    quantiles=None,
    range = prior_range,
    plot_contours=False,
    show_titles=False, 
    title_kwargs={"fontsize": 16},
    figsize=(7,6),
    hist_kwargs={'color':'k'},
    smooth=0.7,
    **dens_kw)

axes = np.array(fig.axes).reshape((ndim, ndim))

for p1 in range(ndim):
    for p2 in range(p1):
        ax = axes[p1, p2]
        x = samples[:,p2]
        y = samples[:,p1]
        z,xi,yi,Ni = binned_statistic_2d(x,y,Delta_chi_square,statistic='min',bins=20,range=[[prior_range[p2][0],prior_range[p2][1]],[prior_range[p1][0],prior_range[p1][1]]])
        z[np.isnan(z)] = 20.0
        z_filter = gaussian_filter(z,sigma=0.7)
        xb = 0.5*(xi[1:] + xi[:-1])
        yb = 0.5*(yi[1:] + yi[:-1])
        CS = ax.contour(xb,yb,z_filter.T,levels=[0,2.6,4.9],cmap='jet')
#        ax.clabel(Cinline=True, fontsize=10)

plt.tight_layout()
plt.subplots_adjust(hspace=0.01,wspace=0.01)
#plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/February2022/MCMCpost_contourf_samples_2_5.pdf',bbox_inches='tight')