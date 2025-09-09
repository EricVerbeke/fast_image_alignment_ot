import finufft
import numpy as np


class Transform:
    
    def __init__(self, images, angles=None, n_points=None, apply_ramp=True):
        """
        Apply various transforms to images:
        - non-uniform polar Fourier transform
        - Radon transform
        - cumulative distribution transform
        - signed cumulative distribution transform
        - inverse cumulative distribution transform
        - signed inverse cumulative distribution transform
        
        Inputs:
        - images as array
        - angles for Radon transform (in degrees)
        - n_points (number of equispaced points sampled in NUFFT)
        
        Output:
        - (regular) (N x ny x nx) array
        - (signed) (N x ny x nx , N x ny x nx) array

        TODO:
        - add an option to increase the sampling on the xgrid (x) for ICDF
        - add option to change grid sampling for ICDF (i.e. Chebyshev nodes)
        """
        
        self.images = images.astype(np.complex128) # required type for nufft
        self.angles = angles
        self.n_points = n_points
        self.apply_ramp = apply_ramp
        self.shape = images.shape
        
        if len(self.shape) == 2:
            self.N, self.ny, self.nx = 1, self.shape[0], self.shape[1]
        elif len(self.shape) == 3:
            self.N, self.ny, self.nx = self.shape[0], self.shape[1], self.shape[2]   

        if self.angles is None:
            self.angles = np.linspace(0, 360, self.ny, endpoint=False)  # scale with size of image
        if self.n_points is None:
            self.n_points = self.ny
            
        self.n_theta = self.angles.size
    
    
    def set_radial_2d_nufft_plan(self, nufft_type=2, eps=1e-8):
                
        rads = self.angles / 180 * np.pi
        
        ### EV: shift points to reflect a true grid center?
        if self.n_points % 2 == 0:
            y_idx = np.linspace(-1, 1, self.n_points, endpoint=False)  # EV: default for even
            # y_idx = np.linspace(-(1 - (1/self.n_points)), 1 - (1/self.n_points), self.n_points, endpoint=True)
            # y_idx = np.linspace(-1, 1, self.n_points, endpoint=True)
        else:
            y_idx = np.linspace(-(1 - (1/self.n_points)), 1 - (1/self.n_points), self.n_points, endpoint=True) # EV: default for odd
            # y_idx = np.linspace(-1, 1, self.n_points, endpoint=True)
            # y_idx = np.linspace(-1, 1, self.n_points, endpoint=False) 

        x_theta = y_idx[:, np.newaxis] * np.sin(rads)[np.newaxis, :]
        x_theta = np.pi * x_theta.flatten()

        y_theta = y_idx[:, np.newaxis] * np.cos(rads)[np.newaxis, :]
        y_theta = np.pi * y_theta.flatten()

        plan = finufft.Plan(nufft_type, (self.nx, self.ny), self.N, eps)
        plan.setpts(x_theta, y_theta)
        
        self.ramp = np.abs(y_idx) # EV: can try variations of the ramp filter
        self.n_lines = len(rads)
        
        return plan
    

    def polar_nufft(self):
        """output ordered in regular grid with size (n_images x n_points x n_angles)"""
        
        plan = self.set_radial_2d_nufft_plan()
        images_ft = plan.execute(self.images).reshape(self.N, self.n_points, self.n_lines)
        
        if self.apply_ramp:
            images_ft *= self.ramp[:, np.newaxis]
        
        return images_ft
    
    
    def radon_transform(self):
        
        images_ft = self.polar_nufft()
        images_rt = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(images_ft, axes=1), axis=1), axes=1).real
        
        return images_rt
    
    
    def get_cdf(self, images_rt):
        
        return np.cumsum(images_rt, axis=1)
    
    
    def cdf_transform(self, rescale=True):
        
        images_rt = self.radon_transform() 
        images_cdf = self.get_cdf(images_rt)
        
        if rescale:
            images_cdf = self.rescale_cdf(images_cdf)
        
        return images_cdf
    
    
    def signed_cdf_transform(self, w=10, B=100, voxel_size=1, smooth=False, rescale=True):
                
        images_rt = self.radon_transform()
        images_rt_pos, images_rt_neg = hahn_decomposition(images_rt)
                
        images_cdf_pos = self.get_cdf(images_rt_pos)
        images_cdf_neg = self.get_cdf(images_rt_neg)
        
        if rescale:
            images_cdf_pos = self.rescale_cdf(images_cdf_pos)
            images_cdf_neg = self.rescale_cdf(images_cdf_neg)
        
        # if smooth:
        #     images_cdf_pos = smooth_cdf(images_cdf_pos, w=w, B=B, voxel_size=voxel_size)
        #     images_cdf_neg = smooth_cdf(images_cdf_neg, w=w, B=B, voxel_size=voxel_size)
             
        return images_cdf_pos, images_cdf_neg
    
    
    def rescale_cdf(self, images_cdf):
                
        cdf_scale = images_cdf[:, -1, :]  # last value of each cdf
        cdf_scale = cdf_scale.reshape(self.N, 1, self.n_theta)
        images_cdf = images_cdf / cdf_scale
        
        return images_cdf       
    
    
    def get_uniform_inverse_cdf(self, images_cdf):
        
        images_icdf = np.zeros(images_cdf.shape)
        
        x = np.linspace(0, 1, self.n_points, endpoint=True)  # EV: use this in interp if including different amount of sampling (xp)
        xp = np.linspace(0, 1, self.n_points, endpoint=True)  # EV: make uniform grid from [0, 1] for interpolation
                
        for n, cdfs in enumerate(images_cdf):
            
            for idx in range(self.n_lines):
                
                icdf = np.interp(x, cdfs[:, idx], xp)
                                
                ### EV: HACK TO ENFORCE BOUNDS (not sure why this helps)
                icdf[0] = 0
                icdf[-1] = 1
                ###
                
                images_icdf[n, :, idx] = icdf
                
        return images_icdf
   
    
    def inverse_cdf_transform(self, rescale=True):
        
        images_cdf = self.cdf_transform(rescale=rescale)
        images_icdf = self.get_uniform_inverse_cdf(images_cdf)
        
        return images_icdf
    
    
    def signed_inverse_cdf_transform(self, rescale=True):
        
        images_cdf_pos, images_cdf_neg = self.signed_cdf_transform(rescale=rescale)
        
        images_icdf_pos = self.get_uniform_inverse_cdf(images_cdf_pos)
        images_icdf_neg = self.get_uniform_inverse_cdf(images_cdf_neg)
        
        return images_icdf_pos, images_icdf_neg
    
    
    def apply_forward_transform(self):
        pass
    

### Extra functions


def ft2(array):
    return np.fft.fftshift(np.fft.fft2(array))


def ift2(array):
    return np.fft.ifft2(np.fft.ifftshift(array)).real


def ftn(array):
    return np.fft.fftshift(np.fft.fftn(array))


def iftn(array):
    return np.fft.ifftn(np.fft.ifftshift(array)).real


def hahn_decomposition(f):
    """takes signed array f -> (f_+, f_-), both of size f"""

    f_p = np.where(f >= 0, f, 0)
    f_n = np.where(f <= 0, f, 0)

    return f_p, f_n



########



# def pdf_to_cdf(A, scale=True):
#     ### A is a matrix with columns as pdf (i.e. from Radon transform)
    
#     A_cdf = np.cumsum(A, axis=0)
    
#     if scale:
#         A_norm = A_cdf[-1, :]
#         A_cdf = A_cdf / A_norm
    
#     return A_cdf


# def cdf_to_icdf(A, n_points, n_theta):
#     ### A is a matrix with columns as cdf
#     ### this can be sped up for signed SW
    
#     start_values = np.zeros(n_theta)
#     end_values = np.ones(n_theta) 
    
#     # start_values = A[0, :]
#     # end_values = A[-1, :]  # EV: breaks if cdf is not monotonic increase

#     xgrid = np.linspace(start_values, end_values, n_points, endpoint=True)
#     yvals = np.arange(n_points)

#     A_icdf = np.zeros((n_points, n_theta))

#     for idx in range(n_theta):
#         icdf = np.interp(xgrid[:, idx], A[:, idx], yvals)
        
#         ### EV: Enforce bounds to ignore interpolation 
#         icdf[0] = 0
#         icdf[-1] = n_points
#         ###
        
#         A_icdf[:, idx] = icdf / n_points  # add normalization to [0, 1] instead of number of pixels
        
#     return A_icdf
    

# def smooth_cdf(cdfs, w=10, B=200, voxel_size=1):
#     ### add option to renormalize each column
#     ### change this to apply to whole image stack
    
#     cdfs_blur = np.zeros(cdfs.shape)
    
#     N = cdfs[0, :, 0].size + 2*w
#     spatial_frequency =  np.fft.fftshift(np.fft.fftfreq(N, voxel_size))
#     grid = np.meshgrid(spatial_frequency**2)[0]
#     G = np.exp(- grid * (B/4))
    
#     for idx, cdf in enumerate(cdfs):
        
#         cdf_pad = np.pad(cdf, pad_width=((w, w), (0, 0)), mode='constant', constant_values=(0, 1))
#         F = np.fft.fftshift(np.fft.fft(cdf_pad, axis=0), axes=0)
#         F_conv = F * G[:, np.newaxis]
#         cdf_blur = np.fft.ifft(np.fft.ifftshift(F_conv, axes=0), axis=0).real
#         cdf_blur = cdf_blur[w:-w, :]
#         cdfs_blur[idx] = cdf_blur
        
#     return cdfs_blur

    
##### add these functions for testing
    
# def radon_transform_from_skimage(array, angles):
#     from skimage.transform import radon
#     return radon(array, angles, preserve_range=True)[:, ::-1]


# def radon_transform_from_real_rotation(array, angles):
        
#     projections = np.zeros((array.shape[1], len(angles)))
    
#     for i, a in enumerate(angles):
#         img_r = rotate(array, a)
#         p = np.sum(img_r, axis=0)
#         projections[:, i] = p
        
#     return projections