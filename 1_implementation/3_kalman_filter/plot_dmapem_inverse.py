"""

"""


from scipy import linalg
import mne
from mne.datasets import sample
from dmap_em_tf import dynamic_map_em
import tensorflow as tf
#from .dmap_em import transition_matrix
from mne.viz import plot_sparse_source_estimates



print(__doc__)

data_path = sample.data_path()
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'
subjects_dir = data_path + '/subjects'

# Read noise covariance matrix
cov = mne.read_cov(cov_fname)
# Handling average file
condition = 'Left Auditory'
evoked = mne.read_evokeds(ave_fname, condition=condition, baseline=(None, 0))
#evoked.crop(tmin=0, tmax=0.003)
evoked.crop(tmin=0, tmax=0.008)
# Handling forward solution
fwd = mne.read_forward_solution(fwd_fname)

ylim = dict(eeg=[-10, 10], grad=[-400, 400], mag=[-600, 600])
evoked.plot(ylim=ylim, proj=True)

###############################################################################
# Run solver
SNR = 5
lam = 1. / SNR**2   # Regularization related to power SNR
b = 3   # Parameter for inv gamma hyper prior, to make it non (little) inform
phi = 0.8   # Temporal autocorrelation of lag 1
maxit = 1
mem_type = 'ram'

#F_hemis = transition_matrix(fwd['src'], alpha=0.5, dist_weight=False)

# F = linalg.block_diag(F_hemis[0].todense(), F_hemis[1].todense())

F = None

# Compute dmapem inverse solution
stc, nus, cost = dynamic_map_em(fwd, evoked, cov, phi=phi, F=F, lam=lam,
                                nu=None, C=None, b=b, save_nu_iter=True,
                                tol=1e-7, maxit=maxit, mem_type=mem_type,
                                prefix=None, delete_cov=True, verbose=None)


###############################################################################
# View in 2D and 3D ("glass" brain like 3D plot)
#plot_sparse_source_estimates()

#mne.viz.plot_source_estimates(fwd['src'], stc, bgcolor=(1, 1, 1),
#                             fig_name="dmapem (cond %s)" % condition,
#                             opacity=0.1)


mne.viz.plot_sparse_source_estimates(fwd['src'],stc)


# brain = stc.plot('sample', 'inflated', 'both',//
#                      subjects_dir=subjects_dir,
#                      clim=dict(kind='value', lims=(0.25, 0.4, 0.65)))
# brain.show_view('lateral')
#
# brain = stc.plot(surface='inflated', hemi='rh', subjects_dir=subjects_dir)