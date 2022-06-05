import numpy as np
from glob import glob

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


M = 'GR'
An= 0.5
Awp= 0.5
name_files = '/cosma7/data/dp004/dc-armi2/mcmc_runs/test/ns/MCMCpost_chains_HOD_'+M+'_L768_500it_28walkers_'+str(An)+'An_'+str(Awp)+'Awp_target_LOWZ_z0.24_0.36_err_1sigma_fullcov_sim_subhaloes_cycle_'
name_files2 = '/cosma7/data/dp004/dc-armi2/mcmc_runs/test/wps/MCMCpost_chains_HOD_'+M+'_L768_500it_28walkers_'+str(An)+'An_'+str(Awp)+'Awp_target_LOWZ_z0.24_0.36_err_1sigma_fullcov_sim_subhaloes_cycle_'
name_files3 = '/cosma7/data/dp004/dc-armi2/mcmc_runs/test/chains/test/MCMCpost_chains_HOD_'+M+'_L768_500it_28walkers_'+str(An)+'An_'+str(Awp)+'Awp_target_LOWZ_z0.24_0.36_err_1sigma_fullcov_sim_subhaloes_cycle_'

list_ngal = []
list_wp = []
list_chain = []
ngal_samples = []
wp_samples = []
chain_samples = []
for i in range(20):
    l = name_files + str(i) + 'test_ngal.npy'
    m = name_files2 + str(i) + 'test_wp.npy'
    n = name_files3 + str(i) + '.npy'
    list_ngal.append(l)
    list_wp.append(m)
    list_chain.append(n)
    ngal_chain = np.load(l)
    wp_chain = np.load(m)
    chain = np.load(n)
    ngal_samples.append(ngal_chain)
    wp_samples.append(wp_chain)
    chain_samples.append(chain)
ns = np.concatenate(ngal_samples,axis=1)
wps = np.concatenate(wp_samples,axis=1)
chains = np.concatenate(chain_samples,axis=1)

mc_step = np.arange(0,500*len(list_chain))
f,ax = plt.subplots(1,1,figsize=(8.0,6.0))
wi=0
wf=28
for i in range(wi,wf):
    ax.plot(mc_step,ns[i],'-')
#ax.hlines(2.9e-4,0,10000,linewidth=2.0,zorder=10,linestyle='--')
ax.set_xlabel(r'MC step')
ax.set_ylabel(r'$n_{gal}$')
#ax.set_ylim(400,800)
ax.set_ylim(1e-4,5e-4)
ax.set_xlim(-100,1e4)
#plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/March2022/MCMC_ngal_MCsteps_An'+str(An)+'_Awp'+str(Awp)+'_'+M+'_'+str(wi)+'_'+str(wf-1)+'.pdf',bbox_inches='tight')
plt.tight_layout()

wp_mean2 = np.mean(np.mean(wps,axis=1),axis=0)

wi=0
wf=10
f,ax = plt.subplots(1,1,figsize=(8.0,6.0))
for i in range(wi,wf):
    ax.plot(mc_step,np.sum(wps[i],axis=1) - np.sum(wp_mean2,axis=0),'-')
#ax.hlines(2.9e-4,0,10000,linewidth=2.0,zorder=10,linestyle='--')
ax.set_xlabel(r'MC step')
ax.set_ylabel(r'$\sum (f - \mu)$')
ax.set_ylim(-300,300)
#ax.set_ylim(1e-4,5e-4)
ax.set_xlim(-100,1e4)
plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/March2022/MCMC_wp_MCsteps_An'+str(An)+'_Awp'+str(Awp)+'_'+M+'_'+str(wi)+'_'+str(wf-1)+'.pdf',bbox_inches='tight')
plt.tight_layout()

for p1 in range(5):
    f,ax = plt.subplots(1,1,figsize=(8.0,6.0))
    wi=0
    wf=10
    for i in range(wi,wf):
        ax.plot(mc_step,chains[i][:,p1],'-')
    #ax.hlines(2.9e-4,0,10000,linewidth=2.0,zorder=10,linestyle='--')
    ax.set_xlabel(r'MC step')
    ax.set_ylabel(labs[p1])
    #ax.set_ylim(400,800)
    ax.set_xlim(-100,1e4)
    ax.set_ylim(PR[p1][0],PR[p1][1])
    plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/March2022/MCMC_'+labs[p1]+'_MCsteps_An'+str(An)+'_Awp'+str(Awp)+'_'+M+'_'+str(wi)+'_'+str(wf-1)+'.pdf',bbox_inches='tight')
    plt.tight_layout()




#===== autocorrelation time ======#


M = 'GR'
An= 0.5
Awp= 0.5
list_chain = glob('/cosma7/data/dp004/dc-armi2/mcmc_runs/GR_An0.5_Awp0.5/chains/MCMCpost_chains_HOD_'+M+'_L768_500it_28walkers_'+str(An)+'An_'+str(Awp)+'Awp_target_LOWZ_z0.24_0.36_err_1sigma_fullcov_sim_subhaloes_cycle_*.npy')
chains = []
for i in range(43):
    sample1 = np.load(list_chain[i])
    chains.append(sample1)
sample_chain = np.concatenate(chains,axis=1)

N = np.exp(np.linspace(np.log(100), np.log(4e4), 15)).astype(int)

new = np.empty(len(N))

sample_chain = np.concatenate(chains,axis=1)
for j in range(5):
    chain0 =  sample_chain[:,:,j]
    for i, n in enumerate(N):
        new[i] = autocorr_new(chain0[:, :n])
    plt.loglog(N, new, "o-")        
ylim = plt.gca().get_ylim()
#plt.plot(N, N / 30.0, "--k", label=r"$\tau = N/50$")
plt.ylim(ylim)
plt.xlim(100,2e4)
plt.xlabel("number of samples, $N$")
plt.ylabel(r"$\tau$ estimates")
plt.legend(fontsize=14);