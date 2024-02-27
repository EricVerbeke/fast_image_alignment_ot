###

# Authors: 

#==============#
# Notes / Todo #
#==============#
# - change to have consistent notation
# - check about correct scalings when using FTs (especially fast fourier l2)
#   - scalings could also be from normalizations of pdf/cdf/icdf e.g. if not strictly positive and normalized could be div error
# - effect of odd/even size images
# - change to work with nonsquare images (supported on disc)

# - make final fast version without redundancies/checks
# - make function that just returns the rotationally aligned image (how to 'unrotate')

# - should drop_negatives be on image, pdf or both?


#==============#
# observations #
#==============#
# - SW2+ramp+drop zeros is defacto best (why is this?)
# - L2 and SW2+ramp+cdf seem to be approximately equal (why is this?) (for cryo but not for cameraman?)
# - L2 with ramp filter only seems to work with mainly low freq images
# - L2(no ramp), SW2(ramp) SW2(cdf + ramp) start to look similar with low freq images
# - sometimes L2 is off by 1degree even with no shift, seems wrong


#================#
# main questions #
#================#
# - order of operations for rotate and translate in iterative approach?  (algo: CoM -> R -> T)
# - interpolation with iterative approach? (e.g. changes image values if updating)
# - ramp filter stuff? 
#    - e.g. results or proof that W2 and/or L2 works better/worse for high-freq images
# - how well does the SW2 approximate the W2 (result must exist already) how do you even compute W2?


#===================#
# potential outline #
#===================#
# - Problem: fast rotation alignment, W2 robust to shift and deformation
# - Result 1) advantage of W2 over L2, simple proof with two deltas (high frequency)
# - Result 2) bounding of W2 with shifts (and/or rotations)
#             - for cryo-EM show that metric also bounded by rotation angle of volume (Moscovich paper)
# - Result 3) computational complexity of algorithm
# - Result 4) shift magnitude plot with real image and cryo image
# - Restul 5) subjective example with mnist dat

# - optional 1) prove something about low vs high frequency images
# - optional 2) prove equivalence of SW2 and W2 

# - proof that CDF ~ L2
# - proof that W2 + ramp + positives is good

# - discussion 1) L2 probably better for full alignment

###


import mrcfile
import numpy as np
import finufft
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


def log_abs(array):
    return np.log(1 + np.abs(array))


def ft2(array):
    return np.fft.fftshift(np.fft.fft2(array))


def ift2(array):
    return np.fft.ifft2(np.fft.ifftshift(array)).real


def ftn(array):
    return np.fft.fftshift(np.fft.fftn(array))


def iftn(array):
    return np.fft.ifftn(np.fft.ifftshift(array)).real


def rad2degree(radian):
    return radian * 180/np.pi


def rotate(array, degrees, reshape=False):
    return ndi.rotate(array, degrees, reshape=reshape)


def translate(array, shift):
    """Translate a 2D image by pixels, shift is a list for [y, x] translation"""
    return np.roll(array, shift, axis=(0,1))


def get_random_rotations(n_images):
    return np.random.uniform(low=0, high=2*np.pi, size=n_images)


def get_random_translations(n_images, max_x_shift=50, max_y_shift=50):
    # set max shift in integer pixles for now, need better way to do this
    return np.array([(np.random.randint(low=-max_y_shift, high=max_y_shift + 1),
                      np.random.randint(low=-max_x_shift, high=max_x_shift + 1)) for _ in range(n_images)])

def apply_rotations(image, rotations):
    return np.array([rotate(image, r) for r in rotations])


def apply_translations(image, translations):
    return np.array([translate(image, t) for t in translations])


def apply_rotations_then_tranaslations(image, rotations, translations):
    # make this also work for just a single image
    
    assert len(rotations) == len(translations), "number rotations and translations not equal"
    
    images_rt = []
    
    N = len(rotations)
    
    for i in range(N):
        image_r = rotate(image, rotations[i])
        image_rt = translate(image_r, translations[i])
        images_rt.append(image_rt)
        
    return np.array(images_rt)


def open_mrc(mrc_file, return_voxel=False):
    with mrcfile.open(mrc_file) as mrc:
        x = mrc.data
        voxel = mrc.voxel_size.x
        mrc.close()
        if return_voxel:
            x = [x, voxel]
    
    return x


def radial_distance_grid(shape):
    """Compute grid of radial distances"""
    
    center = [n//2 for n in shape]
    idx = [slice(-center[i], l-center[i]) for i, l in enumerate(shape)] 
    coords = np.ogrid[idx] # zero-centered grid index
    square_coords = [c**2 for c in coords] # square grid for distance (x^2 + y^2 + z^2 = r^2)
    
    radial_dists = square_coords[0] # initialize to broadcast distance grid by dimension
    for dimension in range(1, len(shape)):
        radial_dists = radial_dists + square_coords[dimension]
        
    return np.round(np.sqrt(radial_dists))


def sphere_mask(r_dists, radius=False):
    """Returns sphere mask as boolean"""
    
    if not radius:
        center = [n//2 for n in r_dists.shape]
        radius = np.amin(center)
        
    mask = r_dists <= radius
    
    return mask


def b_factor_function(shape, voxel_size, B):
    """B factor equation as function of spatial frequency"""
    
    N = shape[0]
    
    spatial_frequency = np.fft.fftshift(np.fft.fftfreq(N, voxel_size))

    sf_grid = np.meshgrid(*[spatial_frequency**2 for dimension in range(len(shape))])

    square_sf_grid = sf_grid[0] # initialize to broadcast by dimension
    for dimension in range(1, len(shape)):
        square_sf_grid = square_sf_grid + sf_grid[dimension]
    
    G = np.exp(- square_sf_grid * (B/4))
    
    return G


def apply_b_factor(v, voxel, B_signal):
    """return array after applying B-factor decay, input is real array"""
    
    G = b_factor_function(v.shape, voxel, B_signal)
    V = ftn(v)
    Vb = G * V
    vb = iftn(Vb)
    
    return vb


def fourier_downsample(array, factor=1, rescale=False):
    """Downsample array by cropping its Fourier transform (factor 2 would give 100pix -> 50pix)"""
    
    assert factor >= 1, "scale factor must be greater than 1"
    
    shape = array.shape
    center = [d//2 for d in shape]
    new_shape = [int(d / factor) for d in shape]
    
    F = ftn(array)
    idx = tuple([slice(center[i] - new_shape[i]//2, center[i] + new_shape[i]//2) for i in range(len(shape))])
    F = F[idx] 
    
    if rescale:
        F = F * (np.product(new_shape) / np.product(shape))
    
    f_downsample = iftn(F)
    
    return f_downsample


def fourier_upsample(array, factor=1, rescale=False):
    """Upsample array by zero-padding its Fourier transform (factor 2 would give 100pix -> 200pix)"""
    
    assert factor >= 1, "scale factor must be greater than 1"
    
    shape = array.shape
    
    F = ftn(array)
    p = int((shape[0] * (factor-1)) / 2)
    F_upsample = np.pad(F, p)
    
    if rescale:
        F_upsample = F_upsample * (np.product(F_upsample.shape) / np.product(F.shape))
    
    f_upsample = iftn(F_upsample)
    
    return f_upsample


def compute_spherically_averaged_power_spectrum(array, rmax):
    
    shape = array.shape

    F = ftn(array)

    rdists = radial_distance_grid(shape)
    index = np.unique(rdists)[:rmax]
    spherically_averaged_power_spectrum = ndi.mean(abs(F)**2, rdists, index)
    
    return spherically_averaged_power_spectrum


def low_pass_filter(array, voxel_size, resolution):
    """Low pass filter array to specified resolution"""

    n = array.shape[0]

    assert resolution >= ((n - 2) / (2*n*voxel_size))**-1, "specified resolution greater than Nyquist"
    
    freq = get_radial_spatial_frequencies(array, voxel_size)  
    res = np.array([1/f if f > 0 else 0 for f in freq])
    radius = np.where(res <= resolution)[0][1]

    r_dists = radial_distance_grid(array.shape)
    lpf_mask = sphere_mask(r_dists, radius)
    
    F = ftn(array)
    F_lpf = F * lpf_mask
    f_lpf = iftn(F_lpf)
    
    return f_lpf


def get_radial_spatial_frequencies(array, voxel_size):
        
    r = np.amax(array.shape)  
    r_freq = np.fft.fftfreq(r, voxel_size)[:r//2]
    
    return r_freq


def get_cdf_from_pdf(pdf):
    ### move normalization to icdf because cdf dist doesn't need cdf [0, 1]
    ### don't want to add div by cdf[-1] incase input is not strictly positive
    
    return np.cumsum(pdf)


def get_uniform_inverse_cdf_from_pdf(pdf, grid, values):
    ### important to normalize for inverse function bc grid and values [0, 1]. can be adaptive?
    
    cdf = get_cdf_from_pdf(pdf)
    
    cdf = cdf / cdf[-1]  
    
    inv_cdf = np.interp(grid, cdf, values)
    
    return inv_cdf


def convert_radon_transform_to_inverse_cdf(array, drop_negatives=True):
    ### fast_radon_transform can contain some negatives so need to replace with zero (even w/out ramp filter)
    ### better labels for ny,nx,grid,values
    ### probably should remove drop_negative flag since I think it's necessary
    
    ny, nx = array.shape
        
    if drop_negatives:
        array = np.where(array<0, 0, array)  # need this because otherwise cdf isn't monotonic increase [0, 1]
        
    grid = np.linspace(0, 1, ny) # uniform grid with n_points (the signal length)
    values = np.arange(ny)
    
    projection_inv_cdfs = [get_uniform_inverse_cdf_from_pdf(array[:, i], grid, values) for i in range(nx)]
    
    return np.array(projection_inv_cdfs).T


def convert_radon_transform_to_cdf(array, drop_negatives=False):
    
    ny, nx = array.shape
    
    if drop_negatives:
        array = np.where(array<0, 0, array)     
        projection_cdfs = [get_cdf_from_pdf(array[:, i]) for i in range(nx)]
        projection_cdfs = [cdf/cdf[-1] for cdf in projection_cdfs]  # normalize to 1 (not necessary)
        
    else:
        projection_cdfs = [get_cdf_from_pdf(array[:, i]) for i in range(nx)]
    
    return np.array(projection_cdfs).T


def normalize_image_to_unit_mass(image, drop_negatives=False):
    
    if drop_negatives:
        image = np.where(image<0, 0, image)
    
    return image / np.sum(image)


def get_sigma_for_snr(x, snr):
    """return standard deviation of WGN for desired snr given real array x"""
    
    N = x.size
    signal = np.sum(x**2)
    noise = np.sqrt(signal / (snr * N))
    
    return noise


def radon_transform_from_real_rotation(array, angles):
    
    ### should include a mask after every rotation ?
    
    projections = np.zeros((array.shape[1], len(angles)))
    
    for i, a in enumerate(angles):
        img_r = rotate(array, a)
        p = np.sum(img_r, axis=0)
        projections[:, i] = p
        
    return projections


def fast_radon_transform(images, angles, apply_ramp=False, nufft_type=2, n_trans=1, eps=1e-8):
    """compute fast radon transform for images using nufft"""
    
    ### for now assume that input is a pair of stacked images to be aligned
    ### need to have check for odd/even shape images
    
    nz, ny, nx = images.shape
    
    images = images.astype(np.complex128)
    
    rads = angles / 180 * np.pi
    n_lines = len(rads)

    y_idx = np.arange(-ny / 2, ny / 2) / ny * 2

    x_theta = y_idx[:, np.newaxis] * np.sin(rads)[np.newaxis, :]
    x_theta = np.pi * x_theta.flatten()
    
    y_theta = y_idx[:, np.newaxis] * np.cos(rads)[np.newaxis, :]
    y_theta = np.pi * y_theta.flatten()
    
    plan = finufft.Plan(nufft_type, (nx, ny), n_trans, eps)
    plan.setpts(x_theta, y_theta)
    
    freqs = np.abs(np.pi * y_idx)
    
    img_rts = []
    
    for img in images:
        s = plan.execute(img)
        s = s.reshape(ny, n_lines)
        
        if apply_ramp:
            s *= freqs[:, np.newaxis]
        
        p = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(s, axes=0), axis=0), axes=0).real  # check scaling after this
        img_rts.append(p)
        
    return np.array(img_rts)


def circulant_matvec(x, c):
    return np.fft.ifft(np.fft.fft(x) * np.fft.fft(c)).real


def fast_shift_computation(y1, y2):
    ### Implement a O(n log n) method to compute the norm squared difference of the shifts of y1 and y2.
    ### we want to compute the circular convolution for each row matrix
    ### including conj and dtype for L2 even though W2 uses real space 
    ### change notation to match paper
    
    c = np.zeros(y2.size).astype(y2.dtype)  # can inherit complex for complex L2
    c[0] = y2[0]
    c[1:] = np.flip(y2[1:])

    return np.linalg.norm(y1)**2 + np.linalg.norm(y2)**2 - 2 * circulant_matvec(y1, np.conj(c))


def sliced_wasserstein_distance(img_1, img_2, angles, apply_ramp=True, inv_cdf=True, drop_negatives=True, all_scores=False):
    ### this is the version for computing SW2 outside of main alignment 
    ### should probably move image normalization to main alignment function
    ### check which things don't need to be recomputed for doing multiple times w/ translation e.g. angles, grid, etc.
    ### have set default angles?
    ### changes to have a different drop_negative for image normalization or have in different functions
    
    ny, nx = img_1.shape
    
    img_1 = normalize_image_to_unit_mass(img_1, drop_negatives)  # normalize so each line projection (pdf) sum = 1
    img_2 = normalize_image_to_unit_mass(img_2, drop_negatives)
    
    imgs = np.stack((img_1, img_2))
    
    img_radon_transforms = fast_radon_transform(imgs, angles, apply_ramp)
    
    if inv_cdf:
        cdf_1 = convert_radon_transform_to_inverse_cdf(img_radon_transforms[0], drop_negatives)
        cdf_2 = convert_radon_transform_to_inverse_cdf(img_radon_transforms[1], drop_negatives)
    
    else:
        cdf_1 = convert_radon_transform_to_cdf(img_radon_transforms[0], drop_negatives)
        cdf_2 = convert_radon_transform_to_cdf(img_radon_transforms[1], drop_negatives)
    
    dist_matrix = np.array([fast_shift_computation(cdf_1[j, :], cdf_2[j, :]) for j in range(nx)])  
    dists = np.sum(dist_matrix, axis=0)
    
    d_min_idx = np.argmin(dists)
    d_min = dists[d_min_idx]
    a_min = angles[d_min_idx]
    
    if all_scores:
        return dists
    else:
        return d_min, a_min


def fast_rotation_l2_distance(img_1, img_2, angles, apply_ramp=True, all_scores=False):
    
    nufft_type=2; n_trans=1; eps=1e-8  # could set as function inputs
    
    ny, nx = img_1.shape

    rads = angles / 180 * np.pi
    n_lines = len(rads)

    y_idx = np.arange(-ny / 2, ny / 2) / ny * 2

    x_theta = y_idx[:, np.newaxis] * np.sin(rads)[np.newaxis, :]
    x_theta = np.pi * x_theta.flatten()

    y_theta = y_idx[:, np.newaxis] * np.cos(rads)[np.newaxis, :]
    y_theta = np.pi * y_theta.flatten()

    plan = finufft.Plan(nufft_type, (nx, ny), n_trans, eps)
    plan.setpts(x_theta, y_theta)
    
    freqs = np.abs(np.pi * y_idx)

    img_1_f = plan.execute(img_1.astype(np.complex128))
    img_1_f = img_1_f.reshape(ny, n_lines) 

    img_2_f = plan.execute(img_2.astype(np.complex128))
    img_2_f = img_2_f.reshape(ny, n_lines)
    
    if apply_ramp:
        img_1_f *= freqs[:, np.newaxis]
        img_2_f *= freqs[:, np.newaxis]

    dist_matrix = np.array([fast_shift_computation(img_1_f[j, :], img_2_f[j, :]) for j in range(nx)]) 
    dists = np.sum(dist_matrix, axis=0)
    
    d_min_idx = np.argmin(dists)
    d_min = dists[d_min_idx]
    a_min = angles[d_min_idx]

    if all_scores:
        return dists
    else:
        return d_min, a_min
    
    
def real_space_rotational_distance(img_1, img_2, angles, all_scores=False):
    
    dists = []
    
    for a in angles:
        img_2_rot = rotate(img_2, -a)  # rotate clockwise as hack for now
        dists.append(np.linalg.norm(img_1 - img_2_rot))
        
    d_min_idx = np.argmin(dists)
    d_min = dists[d_min_idx]
    a_min = angles[d_min_idx]
    
    if all_scores:
        return dists
    else:
        return d_min, a_min


def fast_translation_alignment(img1, img2):
    """aligns image 2 to image 1 by maximum correlation"""
    ### want min distance if trying to compare to L2
    ### add option to log scores
    
    Y1 = ft2(img1)
    Y2 = ft2(img2)

    c = ift2(np.conj(Y1) * Y2).real
    ty, tx = np.unravel_index(np.argmax(c), c.shape)

    img2_aligned = translate(img2, -np.array([ty, tx]))
    
    return img2_aligned


def align_two_images(img_1, img_2, n_shifts, angles, apply_ramp=True, inv_cdf=True):
    ### this returns min angle rotation over n_shifts for now
    ### remove redundant calculations if doing large batch alignment
    
    shifts = np.arange(-n_shifts, n_shifts+1)
    
    alignment_distances = []  # need to make a nice structure or class for this later

    for sy in shifts:
        for sx in shifts:
            img_2_shift = translate(img_2, [sy, sx])

            d_min, a_min = sliced_wasserstein_distance(img_1, img_2_shift, angles, apply_ramp=apply_ramp, inv_cdf=inv_cdf)

            alignment_distances.append((d_min, a_min, sy, sx))
            
    return alignment_distances


def sw2_iterative_rotation_translation_alignment(img_1, img_2, angles, n_iter=3, eps=1e-8, keep_log=True, 
                                                 apply_ramp=True, inv_cdf=True, center=True):

    ### Remake this function for final version! fix logging
    
    log = {i: [] for i in range(n_iter)}  
    
    if center:
        img_1 = shift_to_center_of_mass(img_1)
        img_2 = shift_to_center_of_mass(img_2)
    
    for iteration in range(n_iter):
        
        d_min, a_min = sliced_wasserstein_distance(img_1, img_2, angles, apply_ramp=apply_ramp, inv_cdf=inv_cdf)
        img_2 = rotate(img_2, -a_min)
        
        img_2 = fast_translation_alignment(img_1, img_2)
        
        if keep_log:
            log[iteration] = img_2
        
        z = np.linalg.norm(img_1 - img_2)
        
        if z < eps:
            print('converge at iteration ', iteration)
            break
    
    return img_2


def l2_iterative_rotation_translation_alignment(img_1, img_2, angles, n_iter=3, eps=1e-8, keep_log=True, 
                                                apply_ramp=True, center=True):
    
    ### Remake this function for final version! fix logging
    
    log = {i: [] for i in range(n_iter)}  
    
    if center:
        img_1 = shift_to_center_of_mass(img_1)
        img_2 = shift_to_center_of_mass(img_2)
    
    for iteration in range(n_iter):
        
        d_min, a_min = fast_rotation_l2_distance(img_1, img_2, angles, apply_ramp=apply_ramp)
        img_2 = rotate(img_2, -a_min)
        
        img_2 = fast_translation_alignment(img_1, img_2)
        
        if keep_log:
            log[iteration] = img_2
        
        z = np.linalg.norm(img_1 - img_2)
        
        if z < eps:
            print('converge at iteration ', iteration)
            break
    
    return img_2


def shift_to_center_of_mass(img):
    ### just use scipy for now, but need a fast CoM if aligning many images
    ### CoM with noise?
    
    ny, nx = img.shape
    oy, ox = ny//2, nx//2
    
    cy, cx = ndi.center_of_mass(img)
    
    return translate(img, -np.array(np.round([cy-oy, cx-ox])).astype('int'))


######################
# Plotting Utilities #
######################

def two_plot(image_1, image_2, size=5, color='viridis'):
    fig, axs = plt.subplots(1, 2, figsize=(size, size))
    axs[0].imshow(image_1, cmap=color)
    axs[0].axis('off')
    axs[1].imshow(image_2, cmap=color)
    axs[1].axis('off')
    fig.tight_layout()
    plt.show()


#############################################################################
#############################################################################
#############################################################################

# def fast_radon_transform(array, angles, use_ramp=False):

#     angles = np.array(angles).flatten()
#     img_size = array.shape[1]
#     rads = angles / 180 * np.pi
#     y_idx = np.polynomial.chebyshev.chebpts1(img_size)  # use Cheb pts ?
#     # y_idx = np.arange(-img_size / 2, img_size / 2) / img_size * 2
#     x_theta = y_idx[:, np.newaxis] * np.sin(rads)[np.newaxis, :]
#     y_theta = y_idx[:, np.newaxis] * np.cos(rads)[np.newaxis, :]

#     pts = np.pi * np.vstack(
#         [
#             x_theta.flatten(),
#             y_theta.flatten(),
#         ]
#     )
#     pts = pts.astype(np.float32)

#     lines_f = nufft(array, pts).reshape((img_size, -1))

#     if img_size % 2 == 0:
#         lines_f[0, :] = 0

#     # if use_ramp:
#     #     freqs = np.abs(np.pi * y_idx)
#     #     lines_f *= freqs[:, np.newaxis]

#     # projections = np.real(xp.asnumpy(fft.centered_ifft(xp.asarray(lines_f), axis=0)))  # EV: this is slow on my machine
#     projections = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(lines_f, axes=0), axis=0), axes=0).real

#     return projections



##################
### Extra / Unused
##################

# def unit_rescale(vec):
#     """rescale vector values between 0 and 1"""
#     amin = np.amin(vec)
#     amax = np.amax(vec)
    
#     return (vec - amin) / (amax - amin)


# def unit_rescale_matrix_columns(A):
#     """rescale matrix columns between 0 and 1"""
    
#     shape = A.shape
    
#     A_rescale = np.zeros(shape)
    
#     for j in range(shape[1]):
#         A_rescale[:, j] = rescale(A[:, j])
        
#     return A_rescale

#############################################################################
#############################################################################
#############################################################################