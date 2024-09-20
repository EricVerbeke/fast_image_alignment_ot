import numpy as np

import ftk  # PUT FTK HERE FOR NOW
import pywt  # PUT WEMD HERE FOR NOW
import ot  # PUT OT HERE FOR NOW

### my lib

import utils
import transforms

def circulant_matvec(v, c):

    return np.fft.ifft(np.fft.fft(v) * np.fft.fft(c)).real


def fast_shift_computation(vi, vj):
    ### compute the norm squared difference of the shifts of vj with vi in O(n log n).
    ### i.e., the circular convolution for each row vector

    c = np.zeros(vj.size).astype(vj.dtype)  # inherit complex for complex L2
    c[0] = vj[0]
    c[1:] = np.flip(vj[1:])
    dist = np.linalg.norm(vi)**2 + np.linalg.norm(vj)**2 - 2*circulant_matvec(vi, np.conj(c))

    return dist


def rotational_distances(image1, image2, n_points):
    
    dist_matrix = np.array([fast_shift_computation(image1[j, :], image2[j, :]) for j in range(n_points)])
    dists = np.sum(dist_matrix, axis=0)
    
    return dists


def split_rotational_distances(image1_pos, image2_pos, image1_neg, image2_neg, n_points, scale):
    
    dists_pos = rotational_distances(image1_pos, image2_pos, n_points)
    dists_neg = rotational_distances(image1_neg, image2_neg, n_points)
    
    if scale:
        dists_pos = dists_pos / np.amax(dists_pos)  # EV: How should these be weighted / averaged ???
        dists_neg = dists_neg / np.amax(dists_neg)
    
    dists = (dists_pos + dists_neg) / 2
    
    return dists


### not exactly sure how to set up these funcitons
### should signed be separate function? should include all scores or just min?
### change formatting from dict to matrix? add function to compute mins
def reference_rotational_distances(reference, images, n_points):
    
    N = images.shape[0]
    dists_dict = {}
    
    for i in range(N):
        dists_dict[i] = rotational_distances(reference, images[i], n_points)
        
    return dists_dict
    
    
def split_reference_rotational_distances(ref_pos, images_pos, ref_neg, images_neg, n_points, scale):
    
    N = images_pos.shape[0]
    dists_dict = {}
    
    for i in range(N):
        dists_dict[i] = split_rotational_distances(ref_pos, images_pos[i], 
                                                   ref_neg, images_neg[i], 
                                                   n_points,
                                                   scale)
        
    return dists_dict
        
        
def reference_signed_rotational_distance(refs_t_pos, refs_t_neg, imgs_t_pos, imgs_t_neg, N, n_points, n_theta, metric='Wasserstein'):
    ### Computes the rotational signed sliced distance: d(mu^+ + nu^- , mu^- + nu^+)
    ### refs_t and imgs_t should be the Radon transform of the images
    ### EV: could change to take just RT as input
    
    dists_dict = {idx: np.zeros(n_theta) for idx in range(N)}
    
    for idx in range(N):
        
        for t in range(n_theta):
        
            P = refs_t_pos[0] + abs(utils.translate(imgs_t_neg[idx], 0, t))
            Q = utils.translate(imgs_t_pos[idx], 0, t) + abs(refs_t_neg[0])

            P = transforms.pdf_to_cdf(P)
            Q = transforms.pdf_to_cdf(Q)

            if metric == 'Wasserstein':
                P = transforms.cdf_to_icdf(P, n_points, n_theta)
                Q = transforms.cdf_to_icdf(Q, n_points, n_theta)

            dists_dict[idx][t] = np.linalg.norm(P - Q)
        
    return dists_dict


def reference_rotational_max_sliced_wasserstein(refs_t, imgs_t, N, n_theta):
    ### Input should be inverse cdf transform without ramp filter 
    
    dists_dict = {idx: np.zeros(n_theta) for idx in range(N)}
    
    P = refs_t[0]
    
    for idx in range(N):
        
        Q = imgs_t[idx]
    
        for t in range(n_theta):
        
            Qi = utils.translate(Q, 0, t)

            dist = np.linalg.norm(P - Qi, axis=0)
            dist_max = np.amax(dist)

            dists_dict[idx][t] = dist_max
        
    return dists_dict

   
def pairwise_rotational_distances(images, n_points):
    
    N = images.shape[0]
    dists_dict = {}
    
    for i in range(N):
        for j in range(i+1, N):
            dists_dict[(i, j)] = rotational_distances(images[i], images[j], n_points)
            
    return dists_dict


# def split_pairwise_rotational_distances(images_pos, images_neg, n_points, scale):
    
#     N = images_pos.shape[0]
#     dists_dict = {}
    
#     for i in range(N):
#         for j in range(i+1, N):
#             dists_dict[(i, j)] = split_rotational_distances(images_pos[i], images_pos[j],
#                                                             images_neg[i], images_neg[j],
#                                                             n_points,
#                                                             scale)
            
    return dists_dict


def real_space_rotational_distance(image1, image2, angles):
    ### Need to make this compatible with other distance functions
    ### to be used in the alignment class as distance 'type'
    ### this is the "brute force" rotation approach
    
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


def distance_matrix_from_dict(dists_dict, n_images):
    
    dist_mat = np.zeros((n_images, n_images))
    
    for key, dist in dists_dict.items():
        
        idx1, idx2 = key
        dist_min = np.amin(dist)
        dist_mat[idx1, idx2] = dist_min
        dist_mat[idx2, idx1] = dist_min
        
    return dist_mat


def fast_translation_distance(image1, image2):
    ### note assumptions about this method e.g. circular convolution
    ### vectorize this to multiple images
    
    image1_ft = np.fft.fft2(image1)
    image2_ft = np.fft.fft2(image2)

    corr = np.fft.ifft2(np.conj(image1_ft) * image2_ft)
    ty, tx = np.unravel_index(np.argmax(corr), corr.shape)
    
    return ty, tx


##### PUTTING FTK / WEMD / POT HERE FOR NOW #####


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


def wemd_rotational_distance(image1, image2, angles, wavelet='sym3', level=3):
        
    dists = []
    
    w1 = embed(image1, wavelet, level)
    
    for a in angles:
        image2_rot = utils.rotate(image2, -a)
        w2 = embed(image2_rot, wavelet, level)
        dists.append(np.linalg.norm(w1 - w2, ord=1))
        
    return np.array(dists)


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


def rotational_wasserstein_distance(image1, image2, angles, M):
        
    dists = []
    image1_flat = image1.flatten()

    for a in angles:
        image2_rot = utils.rotate(image2, -a)
        image2_rot = np.where(image2_rot<0, 0, image2_rot)
        image2_rot = image2_rot / np.sum(image2_rot)  # normalize images back to one
        image2_flat = image2_rot.flatten()

        dists.append(ot.emd2(image1_flat, image2_flat, M))  # EV: add numItermax?

    return np.array(dists)


def rotational_sinkhorn_wasserstein_distance(image1, image2, angles, M, reg=10, numItermax=3):
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