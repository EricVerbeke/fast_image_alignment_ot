import numpy as np

import ftk  
import pywt 
import ot 

### my lib

import utils
import transforms


def rotational_distances(U, V):
    ### fast compute ||U - T_l V||^2_F for all shifts of V
    ### EV: need to add normalizing constant
    
    U_norm = np.linalg.norm(U)**2
    V_norm = np.linalg.norm(V)**2

    C = np.zeros(V.shape).astype(V.dtype)
    C[:, 0] = V[:, 0]
    C[:, 1:] = np.flip(V[:, 1:], axis=1)

    Uj_hat = np.fft.fft(U, axis=1)
    Cj_hat = np.fft.fft(C, axis=1)

    UV_l = np.fft.ifft(Uj_hat * Cj_hat, axis=1).real
    UV_l_sum = np.sum(UV_l, axis=0)

    dists = U_norm + V_norm - 2*UV_l_sum
    
    return dists


def signed_rotational_distances(Up, Vp, Un, Vn):
    
    dists_pos = rotational_distances(Up, Vp)
    dists_neg = rotational_distances(Un, Vn)
    
    dists = (dists_pos + dists_neg) / 2
        
    return dists


def slow_sliced_rotational_distances(U, V, n_theta):
    
    dists = np.zeros(n_theta)

    for l in range(n_theta):

        d_w2 = np.mean((U - utils.translate(V, 0, l))**2, axis=0)
        d_sw2 = np.sqrt(np.mean(d_w2))

        dists[l] = d_sw2
        
    return dists


def slow_signed_sliced_rotational_distances(Up, Vp, Un, Vn, n_theta):
            
    d_sw2_p = slow_sliced_wasserstein_rotation(Up, Vp, n_theta)
    d_sw2_n = slow_sliced_wasserstein_rotation(Un, Vn, n_theta)

    dists = (d_sw2_p + d_sw2_n) / 2

    return dists


def l2_distance(U, V):
    
    return np.linalg.norm(U - V)**2


def sliced_distance(U, V):
    
    d_w2 = np.mean((U - V)**2, axis=0)
    d_sw2 = np.sqrt(np.mean(d_w2))
    
    return d_sw2


def signed_sliced_distance(Up, Vp, Un, Vn):
    
    d_sw2_p = sliced_distance(Up, Vp)
    d_sw2_n = sliced_distance(Un, Vn)
    
    d_sw2 = (d_sw2_p + d_sw2_n) / 2
    
    return d_sw2


def max_sliced_distance(U, V):
    
    d = np.mean((U - V)**2, axis=0)
    d_max = np.sqrt(np.amax(d))
    
    return d_max


def rotational_max_sliced_wasserstein(U, V, n_theta):
        
    return np.array([max_sliced_distance(U, utils.translate(V, 0, l)) for l in range(n_theta)])


def reference_rotational_max_sliced_wasserstein(U, V, n_theta, N):

    return {idx: rotational_max_sliced_wasserstein(U, V[idx], n_theta) for idx in range(N)}

            
def reference_rotational_distances(reference, images):
    
    N = images.shape[0]
    dists_dict = {}
    
    for i in range(N):
        dists_dict[i] = rotational_distances(reference, images[i])
        
    return dists_dict
    
    
def reference_signed_rotational_distances(ref_pos, images_pos, ref_neg, images_neg):
    
    N = images_pos.shape[0]
    dists_dict = {}
    
    for i in range(N):
        dists_dict[i] = signed_rotational_distances(ref_pos, images_pos[i], ref_neg, images_neg[i])
        
    return dists_dict

   
def pairwise_rotational_distances(images, n_points):
    
    N = images.shape[0]
    dists_dict = {}
    
    for i in range(N):
        for j in range(i+1, N):
            dists_dict[(i, j)] = rotational_distances(images[i], images[j], n_points)
            
    return dists_dict


def pairwise_signed_rotational_distances(images_pos, images_neg):
    
    N = images_pos.shape[0]
    dists_dict = {}
    
    for i in range(N):
        for j in range(i+1, N):
            dists_dict[(i, j)] = signed_rotational_distances(images_pos[i], images_pos[j], images_neg[i], images_neg[j])
            
    return dists_dict


def real_space_rotational_distances(image1, image2, angles):
    
    dists = []
    
    for a in angles:
        image2_rot = utils.rotate(image2, -a)  # rotate clockwise as hack for now
        dists.append(np.linalg.norm(image1 - image2_rot)**2)  # squared-norm to match fast conv
        
    return np.array(dists)


def real_space_rotational_distance_pairwise(images, angles):
    
    N = images.shape[0]
    dists_dict = {}
    
    for i in range(N):
        for j in range(i+1, N):
            dists_dict[(i, j)] = real_space_rotational_distance(images[i], images[j], angles)
            
    return dists_dict


def fast_translational_distances(image1, image2):
    ### does not zero pad (i.e. circular translation allowed)
    ### vectorize this to multiple images
    
    image1_ft = np.fft.fft2(image1)
    image2_ft = np.fft.fft2(image2)

    corr = np.fft.ifft2(np.conj(image1_ft) * image2_ft)
    ty, tx = np.unravel_index(np.argmax(corr), corr.shape)
    
    return ty, tx


#################################################
#####            OTHER DISTANCES            #####
#####            POT / FTK / WEMD           #####
#################################################


def compute_transport_matrix(image, metric='sqeuclidean'):
    """
    when using emd2 for Wasserstein distance, 
    metric='sqeuclidean' -> W2 and metric='euclidean' -> W1
    https://pythonot.github.io/quickstart.html
    """
    
    ny, nx = image.shape
    xs = np.arange(nx)
    xgrid, ygrid = np.meshgrid(xs, xs)
    points = np.array(list(zip(ygrid.flatten(), xgrid.flatten())))  # order matches flatten convention
    M = ot.dist(points, points, metric=metric)  # returns the 2-norm squared, change metric='euclidean' for w1    
    
    return M


def wasserstein_distance(image1, image2, M, numItermax=100000):
        
    image1_flat = image1.flatten()
    image1_flat = np.where(image1_flat < 0, 0, image1_flat)
    image1_flat = image1_flat / np.sum(image1_flat)
    
    image2_flat = image2.flatten()
    image2_flat = np.where(image2_flat < 0, 0, image2_flat)
    image2_flat = image2_flat / np.sum(image2_flat)
    
    dist = ot.emd2(image1_flat, image2_flat, M, numItermax)  # EV: add numItermax?

    return dist


def rotational_wasserstein_distances(image1, image2, angles, M):
        
    dists = []
    image1_flat = image1.flatten()

    for a in angles:
        image2_rot = utils.rotate(image2, -a)
        image2_rot = np.where(image2_rot<0, 0, image2_rot)
        image2_rot = image2_rot / np.sum(image2_rot)  # normalize images back to one
        image2_flat = image2_rot.flatten()

        dists.append(ot.emd2(image1_flat, image2_flat, M))  # EV: add numItermax?

    return np.array(dists)


def rotational_sinkhorn_distances(image1, image2, angles, M, reg=10, numItermax=3):
    ### EV: can probably combine this with Wasserstein
        
    dists = []
    image1 = np.where(image1<1e-5, 1e-5, image1)  # sinkhorn seems to break with zeros
    image1 = image1 / np.sum(image1)
    image1_flat = image1.flatten()

    for a in angles:
        image2_rot = utils.rotate(image2, -a)
        image2_rot = np.where(image2_rot<1e-5, 1e-5, image2_rot)  # catch negative values from interpolation
        image2_rot = image2_rot / np.sum(image2_rot)  # normalize images back to one
        image2_flat = image2_rot.flatten()

        dists.append(ot.sinkhorn2(image1_flat, image2_flat, M, reg=reg, numItermax=numItermax, method='sinkhorn'))

    return np.array(dists)


def ftk_precompute(reference, images, n_psi, B):
    
    L = reference.shape[0]
    n_images = images.shape[0]
    
    T = 2
    N = L
    ftk_eps = 1e-2
    
    delta_range = 1*B
    oversampling = 1  # what should this parameter be???
    
    rmax = N / 2*np.pi
    ngridr = 3*N
    
    # the following four lines of code perform the necessary precompution for the fourier grids
    pf_grid = ftk.make_tensor_grid(rmax, ngridr, n_psi) # grid for polar fourier transform
    tr_grid = ftk.make_adaptive_grid(delta_range, T/N, oversampling) # grid for shift
    n_bessel = int(np.ceil(delta_range))
    plan = ftk.ftk_plan(tr_grid, pf_grid, n_bessel, ftk_eps)
    
    # the following code computes the polar fourier transforms of the images 
    Shat = ftk.cartesian_to_pft(images, T, pf_grid)
    Mhat = ftk.cartesian_to_pft(reference.reshape((1,L,L)), T, pf_grid)
    
    prods_ftk, tm_ftk = ftk.ftk_execute(plan, Mhat, Shat)
    
    ftk_parameters = [n_images, prods_ftk, pf_grid, n_psi, tr_grid]
    
    return ftk_parameters
    

def ftk_align(reference, images, n_psi, B):
    
    ftk_parameters = ftk_precompute(reference, images, n_psi, B)
    n_images = ftk_parameters[0]
    prods_ftk = ftk_parameters[1]
    pf_grid = ftk_parameters[2]
    n_psi = ftk_parameters[3]
    tr_grid = ftk_parameters[4]
    
    corr=np.zeros(n_images)  # the maximum correlation found (theoretical maximum should be 1)
    angle_rec=np.zeros(n_images)  # the in-plane rotation angle recovered
    shift_rec=np.zeros((n_images,2))  # the shift recovered
    
    for idx in range(n_images):      
        index=np.unravel_index(prods_ftk[0,idx,:,:].argmax(), prods_ftk[0,idx,:,:].shape)
        corr[idx]=prods_ftk[0,idx,:,:].max()
        angle_rec[idx]=pf_grid['all_psi'][:n_psi][index[0]]*180/np.pi 
        shift_rec[idx]=tr_grid['trans'][index[1]]
    
    return corr, angle_rec, shift_rec


def embed(arr, wavelet, level):
    """
    *** This code is borrowed from: https://github.com/RuiyiYang/BOTalign/blob/main/wemd.py#L11
    This function computes an embedding of Numpy arrays such that the L1 distance
    between the resulting embeddings is approximately equal to the Earthmover distance of the arrays.

    Input:
        arr - numpy array.
        level - Decomposition level of the wavelets. Larger levels yield more coefficients and more accurate results.
        wavelet - either the name of a wavelet supported by PyWavelets (e.g. 'coif3', 'sym3') or a pywt.Wavelet object.
                  See https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#built-in-wavelets-wavelist
    Output:
        One-dimensional numpy array containing weighted details coefficients.
        Approximate Earthmover's distances are then given by the l_1 distances of the results, e.g.
            wemd(arr1, arr2) := numpy.linalg.norm(embed(arr1, wavelet, level)-embed(arr2, wavelet, level), ord=1)
    """

    arrdwt = pywt.wavedecn(arr/arr.sum(), wavelet, mode='zero', level=level)

    dimension = len(arr.shape)
    assert dimension in [2,3]

    n_levels = len(arrdwt[1:])

    weighted_coefs = [arrdwt[0].flatten()*2**(n_levels)]
    for (j, details_level_j) in enumerate(arrdwt[1:]):
        for coefs in details_level_j.values():
            multiplier = 2**((n_levels-1-j)*(1+(dimension/2.0)))
            weighted_coefs.append(coefs.flatten()*multiplier)

    return np.concatenate(weighted_coefs)


def wemd_rotational_distances(image1, image2, angles, wavelet='sym3', level=3):
        
    dists = []
    
    w1 = embed(image1, wavelet, level)
    
    for a in angles:
        image2_rot = utils.rotate(image2, -a)
        w2 = embed(image2_rot, wavelet, level)
        dists.append(np.linalg.norm(w1 - w2, ord=1))
        
    return np.array(dists)

