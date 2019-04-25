"""Implemenation of the dynamic maximum a posteriori expectation
maximization algorithm for source localization.

"""
# Authors:  Camilo Lamus <camilo@neurostat.mit.edu>
#
# License: BDS (3-Clause)
from datetime import datetime
from os import remove

import numpy as np
from scipy import sparse, linalg

from sklearn.preprocessing import normalize
from mne.minimum_norm.inverse import _prepare_forward
#, _check_reference
#from mne.minimum_norm.inverse import prepare_forward, check_reference
from mne.source_estimate import SourceEstimate
from mne.source_space import SourceSpaces
from mne.utils import logger, verbose, ProgressBar
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

X_holder = tf.placeholder('float')
Y_holder = tf.placeholder('float')
Z_holder = tf.placeholder('float')
rh_holder = tf.placeholder('float')
alg_mat_holder = tf.placeholder('float')
mat_multiple = tf.tensordot(X_holder, Y_holder, axes=1)
mat_multiple_3 = tf.tensordot(Z_holder, mat_multiple, axes=1)

mat_solve_alg = tf.matrix_solve_ls(alg_mat_holder, rh_holder)
dot_product = tf.tensordot(X_holder, Y_holder, axes=1)

def _vertno_to_nuse_indexing(s):
    r"""Compute index between 0 and nuse-1 corresponding to vertno in
    source space.

    """
    nuse = s['nuse']
    vertno = s['vertno']
    index_map = dict(zip(vertno, range(nuse)))
    return index_map


def _check_num_hemispheres(left_right):
    r"""Checks that source space was given as a list of lenght less or
    equal than 2.

    """
    num_hemis = len(left_right)
    if not isinstance(left_right, list) or num_hemis > 2:
        message = 'Input should be list of size less than 3 (at most 2 hemis)'
        raise ValueError(message)
    return num_hemis


def _tris_to_nuse_indexing(src):
    r"""Transform triangulation index values corresponding to the
    source space surface to indices that would represent the columns of
    the forward solution if it were computed for each hemisphere
    separately.

    """
    num_hemis = _check_num_hemispheres(src)
    tris_nuse_idx = [None] * num_hemis

    for hemi, s in enumerate(src):
        tri = np.copy(s['use_tris'])

        # Create map between two sets of indices
        idx_map = _vertno_to_nuse_indexing(s)

        # Loop through triangulation array to swap to new indices
        for idx in np.nditer(tri, op_flags=['readwrite']):
            idx[...] = idx_map[idx.tolist()]

        tris_nuse_idx[hemi] = tri
    return tris_nuse_idx


def _adjacency_from_tris(tris, dist_weight=False, src=None):
    r"""Computes adjacency matrix from triangulation. Returns a list
    scipy sparse dok matrix, one per hemisphere.

    """
    num_hemis = _check_num_hemispheres(tris)
    if dist_weight:
        if src is None:
            msg = ('Distance weighting requires a source space')
            raise ValueError(msg)

    adjs = [None] * num_hemis
    pairs = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)]
    for hemi, tri in enumerate(tris):
        n_vert = tri.max() + 1
        adjs[hemi] = sparse.dok_matrix((n_vert, n_vert), dtype=np.float)

        # Check that the distance info is in source spaces
        if dist_weight:
            if 'dist' not in src[hemi].keys():
                msg = ('Distance information not in source space. Please add '
                       'source space distance.')
                raise ValueError(msg)
            elif src[hemi]['dist'] is None:
                raise ValueError(msg)
            # Extract distance for vertices in source space
            vertno = src[hemi]['vertno']
            dist = src[hemi]['dist'][np.ix_(vertno, vertno)]

        # Loop through edges of triangulation and compute weights
        # between nearest neighbors
        for row in tri:
            for pair in pairs:
                i, j = row[pair[0]], row[pair[1]]
                weight = 1 if not dist_weight else 1. / dist[i, j]
                adjs[hemi][i, j] = weight

    for hemi in range(num_hemis):
        adjs[hemi].tocsr()
    return adjs


def transition_matrix(src, alpha=0.5, dist_weight=False):
    r"""Computes the transition matrix for the source dynamics.

    The transition matrix :math: `F` is computed as a convex combination
    of the identity matrix :math: `I` and the row-normalized adjacency
    matrix of nearest neighbors is the source space triangulation:

    .. math:: F = \alpha I + (1-\apha)D^{-1}A,

    where :math:`D` is a diagonal matrix with the row sum of the
    weighted adjacency :math:`A`, and :math: `0 < \alpha < 1`.

    Parameters
    ----------
    src : SourceSpaces
        The source space used to computed the weighted adjacency.
    alpha : float, optional
        The parameter that determines the convex combination. It need to
        be a value between 0 and 1, not inclusive (Default is 0.5).
    dist_weight : boolean, optional
        Determines whether or not to use the distance information to
        compute the weights of the adjacency matrix. When `dist_weight`
        is True, the weights in :math: `A` for nearest neighbors are
        equal to the inverse of the distance between sources. When
        False, the weights between neares neighbors are equal to 1. In
        either case, the weights between sources that are not nearest
        neighbors is 0. (Default is False).

    Returns
    -------
    list (of scipy.sparse.csr.csr_matrix)
        A list of size equal to the number of hemispheres, with the
        transition matrix :math: `F` for each hemisphere as a sparse csr
        matrix.

    See Also
    --------
    equilibrium_prior_cov

    References
    ----------
    .. [1] Lamus, C., et al. (2012). A spatiotemporal dynamic solution
       to the MEG inverse problem: An Empirical Bayes approach. arXiv.
       http://arxiv.org/pdf/1511.05056v3.pdf

    """
    # Check inputs
    if not isinstance(src, SourceSpaces):
        raise TypeError('src needs to be instace of SourceSpaces. Got '
                        'instead %s.' % type(src))
    if not 0 < alpha < 1:
        raise ValueError('The value alpha for the convex combination '
                         'has to be greater than 0 and less than 1. '
                         'Got alpha = %d.' % alpha)
    if not isinstance(dist_weight, bool):
        raise TypeError('dist_weight need to be boolean. Got instead '
                        '%b.' % type(dist_weight))

    # Compute adjacency
    tris = _tris_to_nuse_indexing(src)
    adjs = _adjacency_from_tris(tris, dist_weight=dist_weight, src=src)
    # Compute F as the convex combination
    num_hemis = _check_num_hemispheres(adjs)
    F = [None] * num_hemis
    for hemi, adj in enumerate(adjs):
        n_vert = adj.shape[0]
        F[hemi] = alpha * sparse.identity(n_vert, dtype=np.float, format='csr')
        DinvA = normalize(adj, norm='l1', axis=1, copy=True)
        F[hemi] += (1 - alpha) * DinvA

    return F


def equilibrium_prior_cov(F, phi=0.8, sigma2w=1):
    r"""Computes equilibrium prior covariance of the source dynamics.

    This equilibrium covariance is computed in each hemisphere
    separately, which is valid since the transition matrix :math:`F` do
    not share connections between hemispheres, making the prior
    covariance block diagonal. The equilibrium is given as the solution
    to the Lyapunov equation:

    .. math:: C = \phi^2 F C F^T + (1 - \phi^2)sigma^2_{\omega} I.

    Parameters
    ----------
    F : list (of scipy.sparse.csr.csr_matrix)
        The transition matrix, one per hemisphere.
    phi : float, optional
        The history parameter `phi` determines the strengh of the model
        temporal autocorrelation. This value needs to be between -1 and
        1, not inclusive. (Default is 0.8)
    sigma2w : float, optional
        State input variance. Determines the power of the sources.
        (Default is 1).

    Returns
    -------
    list (of 2d numpy arrays)
        A list of size equal to the number of hemispheres, with the
        equilibrium source covariance :math: `C` for each hemisphere.

    See Also
    --------
    transition_matrix

    References
    ----------
    .. [1] Lamus, C., et al. (2012). A spatiotemporal dynamic solution
       to the MEG inverse problem: An Empirical Bayes approach. arXiv.
       http://arxiv.org/pdf/1511.05056v3.pdf

    """
    # Check inputs
    msg = 'F need to be a list of scipy.sparse.csr matrices.'
    if not isinstance(F, list):
        raise TypeError(msg + ' Instead got %s' % type(F))
    for F_hemi in F:
        if not isinstance(F_hemi, sparse.csr.csr_matrix):
            raise TypeError(msg + 'Instead got list of %s' % type(F_hemi))
    if np.abs(phi) >= 1:
        raise ValueError('The absolute value of phi needs to be less than 1')

    # Solve Lyapunov equation in each hemisphere
    num_hemis = _check_num_hemispheres(F)
    C = [None] * num_hemis
    for hemi in range(num_hemis):
        n_vert = F[hemi].shape[0]
        A = phi * F[hemi].todense()
        Q = (1 - phi**2) * sigma2w * np.eye(n_vert)
        C[hemi] = linalg.solve_discrete_lyapunov(A, Q, method='bilinear')

    return C


def _from_mne_to_equations(fwd, evoked, cov, F, lam, nu, C, mem_type, prefix):
    r"""Helper to get mne-python objects to the equations in the model

    """
    gain_info, gain, cov, W, _ = _prepare_forward(fwd, evoked.info, cov)

    #_check_reference(evoked)
    all_ch_names = evoked.ch_names
    sel = [all_ch_names.index(name) for name in gain_info['ch_names']]
    Y = np.dot(W, evoked.data[sel])
    X = np.dot(W, gain)
    _, t = Y.shape
    n, p = X.shape
    tr_Sigma = (linalg.norm(X, ord='fro')**2) / n

    if F is None:
        F = np.eye(p, dtype=np.float)
    if nu is None:
        nu = np.ones(p, dtype=np.float)
    if C is None:
        C = np.diag(nu / (lam * tr_Sigma))
    if mem_type == 'memmap' and prefix is None:
            prefix = datetime.now().strftime('%Y-%m-%d') + '_memmap'

    return X, Y, t, n, p, F, tr_Sigma, nu, C, prefix


def _kalman_filter(X, Y, phi, F, lam, nu, tr_Sigma, C, show_progress=False):
    r"""Implements kalman filter without saving filter and predicted
    covariances.D

    """
    sess = tf.Session(config=config)

    #sess.run(mat_multiple,feed_dict={X_holder=a, Y_holder=b})
    n, p = X.shape
    t = Y.shape[1]
    Beta_filtT = np.zeros((t, p), dtype=np.float)

    # Initialization
    beta_filt = np.zeros(p, dtype=np.float)
    V_filt = C

    # Precompute auxiliary variables
    phiFT = phi * F.T
    sigma2_input = (1 - phi**2) / (lam * tr_Sigma) * nu
    diag_idx_p = np.diag_indices(p, ndim=2)
    diag_idx_n = np.diag_indices(n, ndim=2)

    logger.info('Kalman Filter begins.')
    if show_progress:
        progress = ProgressBar(t - 1, spinner=True)
    for row, y in enumerate(Y.T):
        # Kalman prediction
        if show_progress:
            progress.update(row)
        #beta_pred = np.dot(beta_filt, phiFT)
        beta_pred = sess.run(mat_multiple, feed_dict={X_holder: beta_filt, Y_holder: phiFT})
        #V_pred = np.dot(phiFT.T, np.dot(V_filt, phiFT))
        V_pred = sess.run(mat_multiple_3, feed_dict={Z_holder: phiFT.T, X_holder: V_filt, Y_holder: phiFT})
        V_pred[diag_idx_p] += sigma2_input

        # Kalman innovations
        e_pred = y - np.dot(beta_pred, X.T)  # Innovations
        #XV_pred = np.dot(X, V_pred)
        XV_pred = sess.run(mat_multiple, feed_dict={X_holder: X, Y_holder: V_pred})
        #R = np.dot(XV_pred, X.T)  # Covariance of innovations
        R = sess.run(mat_multiple, feed_dict={X_holder: XV_pred, Y_holder: X.T})
        R[diag_idx_n] += 1

        # Kalman correction
        #G = linalg.solve(R, XV_pred, sym_pos=True).T  # Kalman gain
        G = sess.run(mat_solve_alg, feed_dict={alg_mat_holder: R, rh_holder: XV_pred})

        #beta_filt = beta_pred + np.dot(e_pred, G.T)
        beta_filt = beta_pred + sess.run(mat_multiple, feed_dict={X_holder: e_pred, Y_holder: G.T})
        #V_filt = V_pred - np.dot(G, XV_pred)
        V_filt = V_pred - sess.run(mat_multiple, feed_dict={X_holder: G, Y_holder: XV_pred}) #  np.dot(G, XV_pred)
        Beta_filtT[row] = beta_filt
    logger.info('Kalman filter is done.')

    return Beta_filtT.T


def _kalman_filter_cov(X, Y, phi, F, lam, nu, tr_Sigma, C, mem_type, prefix,
                       compute_deviance=False, show_progress=False):
    r"""Implements Kalman filter and stores filter and prediction
    covariance for all time points using numpy's memmap or in memory.

    """
    n, p = X.shape
    t = Y.shape[1]
    Beta_predT = np.zeros((t, p), dtype=np.float)
    Beta_filtT = np.zeros((t, p), dtype=np.float)
    sess = tf.Session(config=config)



    # Create numpy array or memmaps for predicted and filter covariance
    if mem_type == 'memmap':
        V_pred = np.memmap(prefix + '-V_pred.dat', dtype='float64', mode='w+',
                           shape=(t, p, p))
        V_filt = np.memmap(prefix + '-V_filt.dat', dtype='float64', mode='w+',
                           shape=(t, p, p))
    else:
        V_pred = np.zeros((t, p, p), dtype=np.float)
        V_filt = np.zeros((t, p, p), dtype=np.float)

    # Initialization
    beta_filt_ini = np.zeros(p, dtype=np.float)
    V_filt_ini = C

    # Precompute auxiliary variables
    phiFT = phi * F.T
    sigma2_input = (1 - phi**2) / (lam * tr_Sigma) * nu
    diag_idx_p = np.diag_indices(p, ndim=2)
    diag_idx_n = np.diag_indices(n, ndim=2)

    if compute_deviance:
        deviance = n * t * np.log(2 * np.pi)
    else:
        deviance = None

    logger.info('Kalman Filter begins.')
    if show_progress:
        progress = ProgressBar(t - 1, spinner=True)
    for row, y in enumerate(Y.T):
        # Kalman prediction
        if show_progress:
            progress.update(row)
        if row == 0:
            phiFT = np.float32(phiFT)
            beta_filt_ini = np.float32(beta_filt_ini)
            V_filt_ini = np.float32(V_filt_ini)
            #Beta_predT[row] = np.dot(beta_filt_ini, phiFT).astype('float32')
            Beta_predT[row] = sess.run(dot_product, feed_dict={X_holder: beta_filt_ini, Y_holder: phiFT})
            #V_pred[row] = np.dot(phiFT.T, np.dot(V_filt_ini, phiFT)).astype('float32')
            V_pred[row] = sess.run(mat_multiple_3, feed_dict={Z_holder: phiFT.T, X_holder: V_filt_ini, Y_holder: phiFT})
        else:
            #Beta_predT[row] = np.dot(Beta_filtT[row - 1], phiFT)
            Beta_predT[row] = sess.run(mat_multiple, feed_dict={X_holder: Beta_filtT[row-1], Y_holder: phiFT})
            #V_pred[row] = np.dot(phiFT.T, np.dot(V_filt[row - 1], phiFT))
            V_pred[row] = sess.run(mat_multiple_3, feed_dict={Z_holder: phiFT.T, X_holder: V_filt[row-1], Y_holder: phiFT})
        V_pred[row][diag_idx_p] += sigma2_input

        # Kalman innovations
        #e_pred = y - np.dot(Beta_predT[row], X.T)  # Innovations
        e_pred = y - sess.run(mat_multiple, feed_dict={X_holder: Beta_predT[row], Y_holder: X.T})
        #XV_pred = np.dot(X, V_pred[row])
        XV_pred = sess.run(mat_multiple, feed_dict={X_holder: X, Y_holder: V_pred[row]})
        #R = np.dot(XV_pred, X.T)  # Covariance of innovations
        R = sess.run(mat_multiple, feed_dict={X_holder: XV_pred, Y_holder: X.T})
        R[diag_idx_n] += 1
        #sess.close()

        if compute_deviance:
            deviance += np.linalg.slogdet(R)[1]
            #sess.run()
            tmp = linalg.solve(R, e_pred, sym_pos=True)
            #tmp = sess.run(mat_solve_alg, feed_dict={alg_mat_holder: R, rh_holder: e_pred})
            #deviance += np.dot(e_pred, tmp)
            deviance += sess.run(mat_multiple, feed_dict={X_holder: e_pred, Y_holder: tmp})

        # Kalman correction
        #G = linalg.solve(R, XV_pred, sym_pos=True).T  # Kalman gain
        G = sess.run(mat_solve_alg, feed_dict={alg_mat_holder: R, rh_holder: XV_pred}).T
        #Beta_filtT[row] = Beta_predT[row] + np.dot(e_pred, G.T)
        Beta_filtT[row] = Beta_predT[row] + sess.run(mat_multiple, feed_dict={X_holder: e_pred, Y_holder: G.T})
        #V_filt[row] = V_pred[row] - np.dot(G, XV_pred)
        V_filt[row] = V_pred[row] - sess.run(mat_multiple, feed_dict={X_holder: G, Y_holder: XV_pred})
    logger.info('Kalman filter is done.')
    sess.close()

    # Flush memory to disk
    if mem_type == 'memmap':
        V_pred.flush()
        V_filt.flush()

    return Beta_filtT.T, Beta_predT.T, V_filt, V_pred, deviance


def _delete_posterior_covariance(delete_cov, prefix):
    r"""Helper to delete covariances"""
    sufixes = ['-V_pred.dat', '-V_filt.dat']
    for sufix in sufixes:
        remove(prefix + sufix)


@verbose
def kalman_filter(fwd, evoked, cov, phi=0.8, F=None, lam=0.04, nu=None, C=None,
                  mem_type='nocov', prefix=None, delete_cov=False,
                  show_progress=False, verbose=None):
    r"""The Kalman filter for source localization.

    Compute the Kalman filter estimate of the sources recursively. It
    allows for this computation without storing prediction and filter
    covariance matrices [1].

    Parameters
    ----------
    fwd : instance of Forward
        The forward model. Need to be in fixed orientation.
    evoked : instance of Evoked
        Evoked data.
    cov : instance of Covariance
        Observation noise covariance.
    phi : float, optional
        The history parameter `phi` determines the strengh of the model
        temporal autocorrelation. This value needs to be between -1 and
        1, not inclusive. (Default is 0.8).
    F : numpy.array 2d, optional
        The transition matrix. If None, it is set to the identity
        matrix.
    lam : float, optional
        SNR related parameter. It can be set to the inverse of the power
        signal-to-noise ratio in the data. (Defaults to 0.04, which
        gives an amplitude SNR of 5).
    nu : 1d numpy.array, optional
        Source input variance. (Defaults to numpy.ones(n_sources)).
    C : 2d numpy.array, optional
        Initial state source covariance matrix. (Defaults to multiple
        of the identity matrix).
    mem_type : {'nocov', 'memmap', 'ram'}, optional
        Determines whether the posterior source covariance matrices for
        the different time points are stored. If `mem_type` is 'nocov',
        the covariances are not stored. When `mem_type` is 'memmap', the
        covariances are stores as a memory map binary in disk. When
        it is set to `ram`, they are stores in memory. (Defaults to
        'nocov').
    prefix : string, optional
        Prefix string to save posterior source covariance matrices as
        binary file on disk when `mem_type` is 'memmap'.
    delete_cov : bool, optional
        Whether to erase posterior source covariance binary files from
        disk when `mem_type` is 'memmap'. (Default is False)
    show_progress : bool, optional
        Whether to show progress bar in computation. (Default is True)
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc : SourceEstimate
        Source time courses estimated from evoked data.
    nu : 1d numpy.array of list of 1d numpy.array
        Source input variance estimate. When `save_nu_iter` is True,
        return the list with the estimates in all iterations.
    cost : list (of float)
        The cost function evaluated through the EM iterations that the
        algorithm is maximizing.

    See Also
    --------
    transition_matrix, equilibrium_prior_cov, dynamic_map_em

    References
    ----------
    .. [1] Lamus, C., et al. (2012). A spatiotemporal dynamic solution
       to the MEG inverse problem: An Empirical Bayes approach. arXiv.
       http://arxiv.org/pdf/1511.05056v3.pdf

    """
    #sess = tf.Session(config=config)
    accepted_mem_type = ('nocov', 'memmap', 'ram')
    if mem_type not in accepted_mem_type:
        msg = ('Invalid mem_type ({mem_type}). Accepted values '
               'are "%s"' % '" or "'.join(accepted_mem_type + ('None',)))
        raise ValueError(msg)

    X, Y, t, n, p, F, tr_Sigma, nu, C, prefix = _from_mne_to_equations(
        fwd, evoked, cov, F, lam, nu, C, mem_type, prefix)

    if mem_type in ('memmap', 'ram'):
        if mem_type == 'memmap' and prefix is None:
            prefix = datetime.now().strftime('%Y-%m-%d') + '_memmap_kalman'
        Beta, _, _, _, _ = _kalman_filter_cov(X, Y, phi, F, lam, nu, tr_Sigma,
                                              C, mem_type, prefix,
                                              compute_lik=False,
                                              show_progress=show_progress)
    else:
        Beta = _kalman_filter(X, Y, phi, F, lam, nu, tr_Sigma, C,
                              show_progress=show_progress)

    if delete_cov and mem_type == 'memmap':
        _delete_posterior_covariance(delete_cov, prefix)

    lh_vertno = fwd['src'][0]['vertno']
    rh_vertno = fwd['src'][1]['vertno']
    tmin = evoked.times[0]
    tstep = 1.0 / evoked.info['sfreq']
    stc = SourceEstimate(Beta, vertices=[lh_vertno, rh_vertno], tmin=tmin,
                         tstep=tstep)
    return stc


def _backwards_smoother(Beta_filt, Beta_pred, V_filt, V_pred, phi, F, C,
                        mem_type, prefix, compute_suffi=False,
                        show_progress=False):
    r"""Implements backwards smoother recursions

    """
    sess = tf.Session(config=config)

    p, t = Beta_pred.shape
    Beta_smoothedT = np.zeros((t, p), dtype=np.float32)
    Beta_predT = np.float32(Beta_pred.T)
    Beta_filtT = np.float32(Beta_filt.T)
    #if True: # just used for debug.
    #    #Beta_smoothedT = Beta_filtT
    #    A = np.zeros((p, p), dtype=np.float)
    #    return Beta_smoothedT.T, A


    # Get numpy array memmaps for predicted and filter covariance
    if mem_type == 'memmap':
        V_pred = np.memmap(prefix + '-V_pred.dat', dtype='float64', mode='r',
                           shape=(t, p, p))
        V_filt = np.memmap(prefix + '-V_filt.dat', dtype='float64', mode='r',
                           shape=(t, p, p))
    if compute_suffi:
        A13 = np.zeros((p, p), dtype=np.float)
        A2 = np.zeros((p, p), dtype=np.float)
        V_smoothed = np.zeros((p, p), dtype=np.float)
        V_smoothed += V_filt[t - 1]     # To force V_smoothed not to be memmap
    else:
        A = None

    # Precompute auxiliary variables
    phiF = phi * F
    Beta_smoothedT[t - 1] = Beta_filtT[t - 1].copy()

    logger.info('Backwards Smoother begins.')
    if show_progress:
        progress = ProgressBar(t - 2, spinner=True)
    for idx in range(t - 2, -1, -1):
        if show_progress:
            progress.update(idx)
        idx_plus = idx + 1
        #phiFV_filt = np.dot(phiF, V_filt[idx])
        phiFV_filt = sess.run(mat_multiple, feed_dict={X_holder:phiF, Y_holder: V_filt[idx]})
        #JT = linalg.solve(V_pred[idx_plus], phiFV_filt, sym_pos=True)
        JT = sess.run(mat_solve_alg, feed_dict={mat_solve_alg: V_pred[idx_plus], rh_holder: phiFV_filt})

        delta_smoothed_pred = Beta_smoothedT[idx_plus] - Beta_predT[idx_plus]
        #Beta_smoothedT[idx] = Beta_filtT[idx] + np.dot(delta_smoothed_pred, JT)
        Beta_smoothedT[idx] = Beta_filtT[idx] + sess.run(mat_multiple,feed_dict={X_holder: delta_smoothed_pred, Y_holder: JT})

        if compute_suffi:
            #V_lag_smoothed = np.dot(V_smoothed, JT)
            V_lag_smoothed = sess.run(mat_multiple, feed_dict={X_holder: V_smoothed, Y_holder: JT})
            A2 += V_lag_smoothed
            A2 += np.outer(Beta_smoothedT[idx_plus], Beta_smoothedT[idx])
            delta_V_sm_pre = V_smoothed - V_pred[idx_plus]
            #V_smoothed = V_filt[idx] + np.dot(JT.T, np.dot(delta_V_sm_pre, JT))
            V_smoothed = V_filt[idx] + sess.run(mat_multiple_3, feed_dict={Z_holder: JT.T, X_holder: delta_V_sm_pre, Y_holder: JT})
            A13 += V_smoothed
            A13 += np.outer(Beta_smoothedT[idx], Beta_smoothedT[idx])

    if compute_suffi:
        # Compute smoothed beta of initial condition
        # phiFV_filt = np.dot(phiF, C)
        phiFV_filt = sess.run(mat_multiple,feed_dict={X_holder: phiF, Y_holder:C})
        # JT = linalg.solve(V_pred[0], phiFV_filt, sym_pos=True) # that's the only stuff so slow now.
        JT = sess.run(mat_solve_alg, feed_dict={alg_mat_holder: V_pred[0], rh_holder: phiFV_filt})
        delta_smoothed_pred = Beta_smoothedT[0] - Beta_predT[0]
        # beta_smoothed_ini = np.dot(delta_smoothed_pred, JT)
        beta_smoothed_ini= sess.run(mat_multiple, feed_dict={X_holder: delta_smoothed_pred, Y_holder: JT})

        # Add correlation of beta of last last time point to A1
        A1 = A13 + V_filt[t - 1]
        A1 += np.outer(Beta_smoothedT[t - 1], Beta_smoothedT[t - 1])

        # Add cross-correlation of beta between initial cond and first data pt
        #V_lag_smoothed = np.dot(V_smoothed, JT)
        V_lag_smoothed = sess.run(mat_multiple, feed_dict={X_holder: V_smoothed, Y_holder: JT})
        A2 += V_lag_smoothed
        A2 += np.outer(Beta_smoothedT[0], beta_smoothed_ini)

        # Compute smoothed covariance of inital condition
        delta_V_sm_pre = V_smoothed - V_pred[0]
        #V_smoothed = C + np.dot(JT.T, np.dot(delta_V_sm_pre, JT))
        V_smoothed = C + sess.run(mat_multiple_3, feed_dict={Z_holder: JT.T, X_holder: delta_V_sm_pre, Y_holder: JT})
        # Add correlation of beta of initial condition to A3
        A3 = A13 + V_smoothed
        A3 += np.outer(beta_smoothed_ini, beta_smoothed_ini)

        #A = A1 - np.dot(A2, phiF.T) - np.dot(phiF, A2.T)
        A = A1 - sess.run(mat_multiple, feed_dict={X_holder: A2, Y_holder: phiF.T}) - sess.run(
            mat_multiple, feed_dict={X_holder: phiF, Y_holder: A2.T})
        #A += np.dot(phiF, np.dot(A3, phiF.T))
        A += sess.run(mat_multiple_3, feed_dict={Z_holder: phiF, X_holder: A3, Y_holder: phiF.T})

    logger.info('Backwards Smoother ends.')

    return Beta_smoothedT.T, A



@verbose
def dynamic_map_em(fwd, evoked, cov, phi=0.8, F=None, lam=0.04, nu=None,
                   C=None, b=3, save_nu_iter=False, tol=1e-5, maxit=20,
                   mem_type='ram', prefix=None, delete_cov=False,
                   show_progress=True, verbose=None):
    r"""The Dynamic Maximum a Posteriori Expectation-Maximization
    algorithm for source localization algorithm.

    Compute the Empirical Bayes sources estimates using the Kalman
    Smoother (fixed interval smoother), where the source input variance
    is estiamted via EM [1].

    Parameters
    ----------
    fwd : instance of Forward
        The forward model. Need to be in fixed orientation.
    evoked : instance of Evoked
        Evoked data.
    cov : instance of Covariance
        Observation noise covariance.
    phi : float, optional
        The history parameter `phi` determines the strengh of the model
        temporal autocorrelation. This value needs to be between -1 and
        1, not inclusive. (Default is 0.8).
    F : numpy.array 2d, optional
        The transition matrix. If None, it is set to the identity
        matrix.
    lam : float, optional
        SNR related parameter. It can be set to the inverse of the power
        signal-to-noise ratio in the data. (Defaults to 0.04, which
        gives an amplitude SNR of 5).
    nu : 1d numpy.array, optional
        Source input variance. (Defaults to numpy.ones(n_sources)).
    C : 2d numpy.array, optional
        Initial state source covariance matrix. (Defaults to multiple
        of the identity matrix).
    b : float, optional
        Parameter for inverse gamma prior on the source input variance
        `nu`. (Defaults to 3, which makes a flat prior).
    save_nu_iter : bool, optional
        Whether to return the source inpute variance vector (`nu`) from
        all EM iterations. (Defaults to False).
    tol : float, optional
        Tolerance parameter. The EM iterations stop when the improvement
        between interations is less that `tol` time the deviance of the
        null model. (Default is 1e-5)
    maxit : int, optional
        Maximum number of iterations. (Defaults to 20)
    mem_type : {'memmap', 'ram'}, optional
        Determines if posterior source covariance matrices for the
        different time points are stores as a numpy memory map or in
        RAM. (Defaults to 'memmap').
    prefix : string, optional
        Prefix string to save posterior source covariance matrices as
        binary file on disk when `mem_type` is 'memmap'.
    delete_cov : bool, optional
        Whether to erase posterior source covariance binary files from
        disk when `mem_type` is 'memmap'. (Default is False)
    show_progress : bool, optional
        Whether to show progress bar in computation. (Default is True)
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc : SourceEstimate
        Source time courses estimated from evoked data.
    nus : 1d numpy.array of list of 1d numpy.array
        Source input variance estimate. When `save_nu_iter` is True,
        return the list with the estimates in all iterations.
    cost : list (of float)
        The cost function evaluated through the EM iterations that the
        algorithm is maximizing.

    See Also
    --------
    transition_matrix, equilibrium_prior_cov, kalman_filter

    References
    ----------
    .. [1] Lamus, C., et al. (2012). A spatiotemporal dynamic solution
       to the MEG inverse problem: An Empirical Bayes approach. arXiv.
       http://arxiv.org/pdf/1511.05056v3.pdf

    """
    accepted_mem_type = ('memmap', 'ram')
    if mem_type not in accepted_mem_type:
        msg = ('Invalid mem_type ({mem_type}). Accepted values '
               'are "%s"' % '" or "'.join(accepted_mem_type + ('None',)))
        raise ValueError(msg)

    X, Y, t, n, p, F, tr_Sigma, nu, C, prefix = _from_mne_to_equations(
        fwd, evoked, cov, F, lam, nu, C, mem_type, prefix)

    # Compute a base cost (deviance of null model), which is used in the
    # convergence check. This cost must be high in relation future iters
    deviance_null = n * t * np.log(2 * np.pi) + linalg.norm(Y, ord='fro')**2
    two_neg_log_prior = 2 * b * np.sum(np.log(nu) + 1. / nu)
    cost = [deviance_null + two_neg_log_prior]
    nus = [nu]
    it_num = 0
    converged = False
    logger.info('dMAP-EM begins.')
    while not converged:
        it_num += 1
        logger.info('EM iteration ' + str(it_num))
        # E-step
        # Temporarily track changes in nu
        if it_num > 1:
            delta_nu = linalg.norm(nus[-1] - nus[-2], ord=np.inf)
            print (delta_nu)
        Beta_filt, Beta_pred, V_filt, V_pred, deviance = _kalman_filter_cov(
            X, Y, phi, F, lam, nus[-1], tr_Sigma, C, mem_type, prefix,
            compute_deviance=True, show_progress=show_progress)
        Beta_smoothed, A = _backwards_smoother(
            Beta_filt, Beta_pred, V_filt, V_pred, phi, F, C, mem_type, prefix,
            compute_suffi=True, show_progress=show_progress)
        # Evaluate cost (deviance - 2 * log_prior)
        two_neg_log_prior = 2 * b * np.sum(np.log(nus[-1]) + 1. / nus[-1])
        cost.append(deviance + two_neg_log_prior)
        # Check for convergence
        delta_cost = np.abs(cost[-1] - cost[-2])
        print (delta_cost)
        print (delta_cost / deviance_null)
        if it_num >= maxit or delta_cost < tol * deviance_null:
            converged = True
        else:
            # M-step
            a = np.diag(A)
            nu_new = (lam * tr_Sigma / (1 - phi**2) * a + 2 * b) / (t + 2 * b) # inverse gamma. will need to be changed, add laplacian, Jeffreys, log_sum,
            # observation covariance update. added by Feng.
            nus.append(nu_new)
        #converged = 1 # add by Feng for debug

    if not save_nu_iter:
        nus = nus[-1]
    # Remove cost[0], which is not a real cost but just a number for padding
    cost.pop(0)

    if delete_cov and mem_type == 'memmap':
        _delete_posterior_covariance(delete_cov, prefix)

    lh_vertno = fwd['src'][0]['vertno']
    rh_vertno = fwd['src'][1]['vertno']
    tmin = evoked.times[0]
    tstep = 1.0 / evoked.info['sfreq']
    #Beta_smoothed_norm = norm_3_to_1(Beta_smoothed)
    Beta_smoothed_norm = np.asarray([np.mean(Beta_smoothed[i:(i + 1) * 3, :], axis=0) for i in range(np.floor(Beta_smoothed.shape[0] / 3).astype(int))])

    stc = SourceEstimate(Beta_smoothed_norm, vertices=[lh_vertno, rh_vertno],
                         tmin=tmin, tstep=tstep)
    return stc, nus, cost
@verbose
def sparse_dynamic_map_em(fwd, evoked, cov, phi=0.8, F=None, lam=0.04, nu=None,
                   C=None, b=3, save_nu_iter=False, tol=1e-5, maxit=20,
                   mem_type='ram', prefix=None, delete_cov=False,
                   show_progress=True, verbose=None):




    return  stc, nus, cost

