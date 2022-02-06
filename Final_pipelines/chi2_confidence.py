import numpy as np
from astroML.stats import binned_statistic_2d
from scipy.stats import binned_statistic_2d
from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter

chains = np.load('/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/chains/MCMCpost_chains_HOD_GR_L768_1600it_112walkers_0.5An_0.5Awp_target_LOWZ_z0.24_0.36_err_sigma_sim_fullcov_subhaloes.npy')
loglikelihood = np.load('/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/likelihoods/MCMClklhd_chains_HOD_GR_L768_1600it_56walkers_0.5An_0.5Awp_target_LOWZ_z0.24_0.36_err_sigma_fullcov_sim_subhaloes.npy')

nt = []
for i in range(chains.shape[0]):
    if len(np.unique(chains[i],axis=0)) > 1:
        nt.append(i)

lklhd_samples = loglikelihood.T[nt].flatten()
samples = chains[nt].reshape(chains[nt].shape[0]*chains[nt].shape[1],chains[nt].shape[2])

prior_range = np.array([np.min(samples,axis=0),np.max(samples,axis=0)]).T

max_likelihood = lklhd_samples.max()
chi_2_samples = -2*(lklhd_samples)
chi_min = -2*max_likelihood
Delta_chi_square = chi_2_samples - chi_min

p1=0
p2=1
x = samples[:,p1]
y = samples[:,p2]

z,xi,yi = binned_statistic_2d(x,y,Delta_chi_square,statistic='median',bins=50,range=[[prior_range[p1][0],prior_range[p1][1]],[prior_range[p2][0],prior_range[p2][1]]])

xb = 0.5*(xi[1:] + xi[:-1])
yb = 0.5*(yi[1:] + yi[:-1])

z[np.isinf(z)] = 100
z[np.isnan(z)] = 100

sigma = 0.3 # this depends on how noisy your data is, play with it!
z_filter = gaussian_filter(z, sigma)


theta_1s = samples[(Delta_chi_square<5.9)&(samples[:,4]>0.8)]
theta_1s_unique = np.unique(theta_1s,axis=0)

Y_model = []
for t in theta_1s_unique:
    ymodel = model(t)
    Y_model.append(ymodel)

p1=0
p2=1
f,ax = plt.subplots(1,1,figsize=(6.5,6))
for i in range(chains.shape[0]):
    ax.plot(chains[i][:,p1],chains[i][:,p2],'.',c=ci[i],ms=1)
ax.set_xlabel(labs[p1])
ax.set_ylabel(labs[p2])
ax.set_xlim(prior_range[p1][0],prior_range[p1][1])
ax.set_ylim(prior_range[p2][0],prior_range[p2][1])
plt.tight_layout()
#plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/January2022/parameter_space_03.png',bbox_inches='tight')
plt.show()

f,ax = plt.subplots(1,1,figsize=(6.5,6))
#ax.plot(samples[:,p1],samples[:,p2],'.',c=ci[i],ms=1)
ax.contourf(xb,yb,z.T,levels=[0,5.9,9.2,11.3],cmap='jet_r')
ax.set_xlabel(r'$\log M_{min}$')    
ax.set_ylabel(r'$\log M_1$')
plt.tight_layout()
#plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/January2022/parameter_space_03.png',bbox_inches='tight')
plt.show()

p1=0
p2=1
f,ax = plt.subplots(1,1,figsize=(6.5,6))

ax.plot(samples[:,p1],samples[:,p2],'.',c=ci[i],ms=1)
ax.set_xlabel(labs[p1])
ax.set_ylabel(labs[p2])
ax.set_xlim(13.0,13.5)
ax.set_ylim(13.6,14.4)
plt.tight_layout()
#plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/January2022/parameter_space_03.png',bbox_inches='tight')
plt.show()


f,ax = plt.subplots(1,1,figsize=(6.5,6))
for i in range(len(Y_model)):
    ax.plot(rp,(Y_model[i][1][:,0]-Y_data[1][:,0])/Y_data[1][:,0],'-')
ax.errorbar(rp,np.zeros_like(rp),yerr=3*Y_data[1][:,1]/Y_data[1][:,0],fmt='ko')
ax.set_xscale('log')
ax.set_ylim(-1,1)
plt.tight_layout()
#plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/January2022/parameter_space_03.png',bbox_inches='tight')
plt.show()