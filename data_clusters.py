import numpy as np
from astropy.io import fits
from astropy.cosmology import WMAP7 as cosmo
from scipy.interpolate import interp1d

c2 = fits.open('/cosma5/data/dp004/dc-armi2/CODEX/codex_phz_spe_combined.fits')
#codex_sdss = c1[1].data
codex_legacy_Lindholm = c2[1].data
#Cluter catalogue and randoms:
codex = codex_legacy_Lindholm[codex_legacy_Lindholm['z'] > 0]
r1 = np.loadtxt('/cosma5/data/dp004/dc-armi2/CODEX/random_codex_fixedrichness.cat.gz')

#Richness-redshift dependence
zrange = np.arange(0,0.7,0.01)
l_5 = 22*(zrange/0.15)**0.8#richness cut for the 95% best richness
#l5_z = interp1d(zrange,l_5,kind='linear')
z_grid = [0.2,0.4]
c=10
#fz_grid = [l5_z(0.4)+c,l5_z(0.4)]
richness_cut = interp1d(zrange,l_5,kind='linear',fill_value='extrapolate')

codex_clean = codex[((codex['LAMBDA_CHISQ_OPT'])>richness_cut(codex['z']))]
#codex_clean = codex[((codex['LAMBDA_CHISQ_OPT'])>l5_z(codex['z']))]
random_clean = r1[(r1[:,7]>richness_cut(r1[:,2]))]
#random_clean = r1[(r1[:,7]>l5_z(0.4))]

codex_subsample = codex_clean[(codex_clean['z']>0.2)&(codex_clean['z']<0.4)]
random_subsample = random_clean[(random_clean[:,2]>0.2)&(random_clean[:,2]<0.4)]

#numberdensity and JK error
zbins = np.linspace(0.2,0.4,21)#np.array([0.1,0.2,0.3,0.4,0.5,0.6])
Om_r = 10337.2 * (np.pi/180.)**2. #full sdss area ngc+sgc
dV_dz = Om_r/3 * (dcom(zbins[1:])**3 - dcom(zbins[:-1])**3)#com. volume
n_cluster = len(codex_subsample)/np.sum(dV_dz)

survey_limits = np.loadtxt('/cosma5/data/dp004/dc-armi2/SDSS/BOSS/DR12/subsamples/resampling/R25/limits_subvolumes.txt')

C = np.zeros(len(survey_limits))
for i in range(len(C)):
    patch = survey_limits[i]
    cond1 = (codex_subsample['RA_OPT']>patch[0])&((codex_subsample['RA_OPT']<patch[1]))&(codex_subsample['DEC_OPT']>patch[2])&(codex_subsample['DEC_OPT']<patch[3])
    C[i] = len(codex_subsample[~cond1])

n_std = np.sqrt(len(C))*np.std(C) / np.sum(dV_dz)    
np.savetxt('/cosma7/data/dp004/dc-armi2/Jackknife_runs/JK25_HighRichnessClusterNumberDensity_z0.2_0.4.txt',np.array([n_cluster,n_std]).T,newline=" ")

cs = (codex_subsample['RA_OPT']>100)&(codex_subsample['RA_OPT']<300) #north-south separation
rs = (random_subsample[:,0]>100)&(random_subsample[:,0]<300)

codex_subsample_ngc = codex_subsample[cs]
codex_subsample_sgc = codex_subsample[~cs]
random_subsample_ngc = random_subsample[rs]
random_subsample_sgc = random_subsample[~rs]

#Creating cute input catalogues: RA DEC z weight
codex_subsample_cute = np.array([codex_subsample['RA_OPT'],codex_subsample['DEC_OPT'],codex_subsample['z'],np.ones_like(codex_subsample['z']),codex_subsample['LAMBDA_CHISQ_OPT']]).T

codex_subsample_ngc_cute = np.array([codex_subsample_ngc['RA_OPT'],codex_subsample_ngc['DEC_OPT'],codex_subsample_ngc['z'],np.ones_like(codex_subsample_ngc['z']),codex_subsample_ngc['LAMBDA_CHISQ_OPT']]).T

codex_subsample_sgc_cute = np.array([codex_subsample_sgc['RA_OPT'],codex_subsample_sgc['DEC_OPT'],codex_subsample_sgc['z'],np.ones_like(codex_subsample_sgc['z']),codex_subsample_sgc['LAMBDA_CHISQ_OPT']]).T

random_subsample_cute = np.array([random_subsample[:,0],random_subsample[:,1],random_subsample[:,2],np.ones_like(random_subsample[:,2]),random_subsample[:,7]]).T

random_subsample_ngc_cute = np.array([random_subsample_ngc[:,0],random_subsample_ngc[:,1],random_subsample_ngc[:,2],np.ones_like(random_subsample_ngc[:,2]),random_subsample_ngc[:,7]]).T

random_subsample_sgc_cute = np.array([random_subsample_sgc[:,0],random_subsample_sgc[:,1],random_subsample_sgc[:,2],np.ones_like(random_subsample_sgc[:,2]),random_subsample_sgc[:,7]]).T

np.savetxt('/cosma5/data/dp004/dc-armi2/CODEX/subsamples/cluster_CODEX-LEGACY_z0.2_0.4.dat',codex_subsample_cute)
np.savetxt('/cosma5/data/dp004/dc-armi2/CODEX/subsamples/cluster_CODEX-LEGACY_North_z0.2_0.4.dat',codex_subsample_ngc_cute)
np.savetxt('/cosma5/data/dp004/dc-armi2/CODEX/subsamples/cluster_CODEX-LEGACY_South_z0.2_0.4.dat',codex_subsample_sgc_cute)
np.savetxt('/cosma5/data/dp004/dc-armi2/CODEX/subsamples/random_CODEX-LEGACY_z0.2_0.4.dat',random_subsample_cute)
np.savetxt('/cosma5/data/dp004/dc-armi2/CODEX/subsamples/random_CODEX-LEGACY_North_z0.2_0.4.dat',random_subsample_ngc_cute)
np.savetxt('/cosma5/data/dp004/dc-armi2/CODEX/subsamples/random_CODEX-LEGACY_South_z0.2_0.4.dat',random_subsample_sgc_cute)





#codex_clean = codex[((codex['LAMBDA_CHISQ_OPT'])>l5_z(0.4))]
#random_clean = r1[(r1[:,7]>24)]

