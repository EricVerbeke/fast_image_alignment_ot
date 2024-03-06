import utils
import numpy as np

import ftk  # PUT FTK HERE FOR NOW
import pywt  # PUT WEMD HERE FOR NOW
import ot  # PUT OT HERE FOR NOW



def circulant_matvec(v, c):

    return np.fft.ifft(np.fft.fft(v) * np.fft.fft(c)).real


def fast_shift_computation(vi, vj):
    ### compute the norm squared difference of the shifts of vi and vj in O(n log n).
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


def signed_rotational_distances(image1_pos, image2_pos, image1_neg, image2_neg, n_points):
    
    dists_pos = rotational_distances(image1_pos, image2_pos, n_points)
    dists_neg = rotational_distances(image1_neg, image2_neg, n_points)
    
    dists = dists_pos + dists_neg
    
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
    
    
def signed_reference_rotational_distances(ref_pos, images_pos, ref_neg, images_neg, n_points):
    
    N = images_pos.shape[0]
    dists_dict = {}
    
    for i in range(N):
        dists_dict[i] = signed_rotational_distances(ref_pos, images_pos[i], ref_neg, images_neg[i], n_points)
        
    return dists_dict
        
        
def pairwise_rotational_distances(images, n_points):
    
    N = images.shape[0]
    dists_dict = {}
    
    for i in range(N):
        for j in range(i+1, N):
            dists_dict[(i, j)] = rotational_distances(images[i], images[j], n_points)
            
    return dists_dict


def signed_pairwise_rotational_distances(images_pos, images_neg, n_points):
    
    N = images_pos.shape[0]
    dists_dict = {}
    
    for i in range(N):
        for j in range(i+1, N):
            dists_dict[(i, j)] = signed_rotational_distances(images_pos[i], images_pos[j], images_neg[i], images_neg[j], n_points)
            
    return dists_dict


def real_space_rotational_distance(image1, image2, angles):
    ### Need to make this compatible with other distance functions
    ### to be used in the alignment class as distance 'type'
    ### rename this as brute force
    
    dists = []
    
    for a in angles:
        image2_rot = utils.rotate(image2, -a)  # rotate clockwise as hack for now
        dists.append(np.linalg.norm(image1 - image2_rot)**2)  # squared-norm to match fast conv
        
    return np.array(dists)


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


def shift_align_images(image1, image2):
    ### change this function name
    
    ty, tx = fast_translation_distance(image1, image2)
    
    image_2_align = utils.translate(image2, -ty, -tx)
    
    return image_2_align





##### PUTTING FTK / WEMD / POT HERE FOR NOW #####

def ftk_precompute(reference, images, B):
    
    L = reference.shape[0]
    n_images = images.shape[0]
    
    T = 2
    N = L
    ftk_eps = 1e-2
    
    delta_range = 1*B
    oversampling = 1
    
    rmax = N / 2*np.pi
    ngridr = 3*N
    n_psi = 360
    
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
    

def ftk_align(reference, images, B):
    
    ftk_parameters = ftk_precompute(reference, images, B)
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


def wemd_rotational_distance(image1, image2, angles, wavelet='coif3', level=1):
    
    ### *** How to set the wavelet and level???
    
    dists = []
    
    w1 = embed(image1, wavelet, level)
    
    for a in angles:
        image2_rot = utils.rotate(image2, -a)
        w2 = embed(image2_rot, wavelet, level)
        dists.append(np.linalg.norm(w1 - w2, ord=1))
        
    return np.array(dists)


def rotational_wasserstein_distance(image1, image2, angles, M):
        
    dists = []
    image1_flat = image1.flatten()

    for a in angles:
        image2_rot = utils.rotate(image2, -a)
        image2_rot = np.where(image2_rot<0, 0, image2_rot)
        image2_rot = image2_rot / np.sum(image2_rot)
        image2_flat = image2_rot.flatten()

        dists.append(ot.emd2(image1_flat, image2_flat, M))

    return np.array(dists)    


def compute_transport_matrix(image):
    
    ny, nx = image.shape
    xs = np.arange(nx)
    xgrid, ygrid = np.meshgrid(xs, xs)
    points = np.array(list(zip(ygrid.flatten(), xgrid.flatten())))  # order matches flatten convention
    M = ot.dist(points, points)  # returns the 2-norm squared, change metric='euclidean' for w1    
    
    return M



# ### separate the precompute part of FTK code to speed up
# def ftk_align(templates, img, B):
#     '''
#     inputs:
#         templates: an ntemp x L x L matrix, where ntemp = the number of images to be aligned and L is the size of the images
#         img: an L x L matrix, representing a single image
#         B: a number, representing the window size for shift. The shifts to be searched over lie in [-B,B]^2.   
    
    
#     returns: (see more at the end of this function)
#         corr: maximum correlation found
#         angle_rec: optimal in-plane rotation angle
#         shift_rec: optimal shift
#     '''

  
    
#     L=img.shape[0]; ntemp=templates.shape[0]
    
#     T=2; N=L; ftk_eps=1e-2; 
#     # These parameters are used by the authors and I didn't play much around with their choices  
    
#     delta_range=1*B; oversampling=1 
#     # delta_range = maximum size of shifts in each dimension 
#     # oversampling can only be an integer, which controls the mesh size of discretization
#     # oversampling does not change the window size
#     # oversampling=1 is good enough and oversampling=2 gives a finer discretization
    
#     rmax=N/2*np.pi; ngridr=3*N; n_psi=360 
#     # ngridr = the number of grids in the radial direction; n_psi = the number of grids in theta direction
#     # my experience is that ngridr=N; n_psi=90 is good enough  
    
    
#     # the following four lines of code perform the necessary precompution for the fourier grids
#     pf_grid = ftk.make_tensor_grid(rmax, ngridr, n_psi) # grid for polar fourier transform
#     tr_grid = ftk.make_adaptive_grid(delta_range, T/N, oversampling) # grid for shift
#     n_bessel = int(np.ceil(delta_range))
#     plan = ftk.ftk_plan(tr_grid, pf_grid, n_bessel, ftk_eps)
    
    
#     # the following code computes the polar fourier transforms of the images 
#     Shat = ftk.cartesian_to_pft(templates, T, pf_grid)
#     Mhat = ftk.cartesian_to_pft(img.reshape((1,L,L)), T, pf_grid)
    
#     # everything up to here can be treated as precomputation and can be separated from the rest 
#     # of the code
    
    

#     # the following code computes all the inner products between 
#     # templates and shifted&rotated versions of img 
#     # (or shifted&rotated versions of templates and img)
#     # prods_ftk is a 1 x ntemp x n_psi x n_trans matrix, where n_trans is the number of translation grid
#     prods_ftk, tm_ftk = ftk.ftk_execute(plan, Mhat, Shat)
    
    
#     # the following code finds for each image in templates the optimal rotation and shift.
#     # this is done by finding first the index of largest inner product for each image
#     # and then recover the rotation and shift from pf_grid and tr_grid
#     # PS: the shift recovery is probably wrong but I never figured out the correct way

#     corr=np.zeros(ntemp); angle_rec=np.zeros(ntemp); shift_rec=np.zeros((ntemp,2))
#     # corr = the maximum correlation found (theoretical maximum should be 1)
#     # angle_rec = the in-plane rotation angle recovered
#     # shift_rec = the shift recovered
#     for i in range(ntemp):      
#         index=np.unravel_index(prods_ftk[0,i,:,:].argmax(), prods_ftk[0,i,:,:].shape)
#         corr[i]=prods_ftk[0,i,:,:].max()
#         angle_rec[i]=pf_grid['all_psi'][:n_psi][index[0]]*180/np.pi; 
#         shift_rec[i]=tr_grid['trans'][index[1]]
    
#     return corr, angle_rec, shift_rec