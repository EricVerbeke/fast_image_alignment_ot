import finufft
import numpy as np


class Transform:
    
    def __init__(self, images, transform='SCDF', angles=None, n_points=None, apply_ramp=True):
        """
        Assumes input is image stack, apply various polar image transforms:
        - RT = Radon transform
        - PFT = polar Fourier transform
        - CDF = cumulative distribution function
        - SCDF = signed cumulative distribution function
        - ICDF = inverse cumulative distribution function
        - ISCDF = inverse signed cumulative distribution function
        
        - angles for Radon transform (in degrees)
        - n_points (number of radial grid points)
        
        ### TODO:
        ### - make transforms apply to single image stacks
        ### - address comments in functions (e.g. add oversample in icdf)
        ### - fix grids
        """
        
        self.images = images.astype(np.complex128) # required type for nufft
        self.apply_ramp = apply_ramp
        self.angles = angles
        self.n_points = n_points
        
        shape = self.images.shape
        self.N = shape[0]
        self.ny = shape[1]
        self.nx = shape[2]

        # If angles and n_points not set, scale according to image size
        if self.angles is None:
            self.angles = np.linspace(0, 360, self.ny, endpoint=False)
        if self.n_points is None:
            self.n_points = self.ny
            
        self.n_theta = self.angles.size
    
    
    def set_radial_2d_nufft_plan(self, nufft_type=2, eps=1e-8):
        
        n_trans = self.N  
        
        rads = self.angles / 180 * np.pi
        
        ### EV: need to check that grid is generated correctly
        if self.n_points % 2 == 0:
            y_idx = np.linspace(-1, 1, self.n_points, endpoint=False)  # EV: shift points right to reflect true grid center?
            # y_idx = np.linspace(-(1 - (1/self.n_points)), 1 - (1/self.n_points), self.n_points, endpoint=True)
        else:
            y_idx = np.linspace(-(1 - (1/self.n_points)), 1 - (1/self.n_points), self.n_points, endpoint=True)

        x_theta = y_idx[:, np.newaxis] * np.sin(rads)[np.newaxis, :]
        x_theta = np.pi * x_theta.flatten()

        y_theta = y_idx[:, np.newaxis] * np.cos(rads)[np.newaxis, :]
        y_theta = np.pi * y_theta.flatten()

        plan = finufft.Plan(nufft_type, (self.nx, self.ny), n_trans, eps)
        plan.setpts(x_theta, y_theta)
        
        self.ramp = np.abs(y_idx)
        # self.ramp = np.hamming(y_idx.size)
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

        ### pick to have function inputs here or in class parameters
        
        images_rt = self.radon_transform()
        
        ### *** EV: STILL TESTING THIS! 
        ### *** do I need to add zeros to both side
        ### *** how to manage update to size?
        # # if add_zeros: do this
        # images_rt = np.pad(images_rt, pad_width=((0, 0), (1, 1), (0, 0)))  # enfore rt starts at 0
        # self.n_points = self.n_points + 2
        ###
        
        images_cdf = self.get_cdf(images_rt)
        
        if rescale:
            images_cdf = self.rescale_cdf(images_cdf)
        
        return images_cdf
    
    
    def signed_cdf_transform(self, rescale=True):
        
        ### assertion doesn't catch if individual images in stack are all positive, move assertion to the decomp?
        
        images_rt = self.radon_transform()
        
        ### *** EV: STILL TESTING THIS! 
        ### *** do I need to add zeros to both side
        ### *** how to manage update to size?
        # # if add_zeros: do this
        # images_rt = np.pad(images_rt, pad_width=((0, 0), (1, 0), (0, 0)))  # enfore rt starts at 0
        # self.n_points = self.n_points + 1
        ###
        
        images_rt_pos, images_rt_neg = self.jordan_decomposition(images_rt)
        
        # assert np.any(images_rt_neg) == True, "all images are positive, use cdf_transform"
        
        images_cdf_pos = self.get_cdf(images_rt_pos)
        images_cdf_neg = self.get_cdf(images_rt_neg)
        
        if rescale:
            images_cdf_pos = self.rescale_cdf(images_cdf_pos)
            images_cdf_neg = self.rescale_cdf(images_cdf_neg)
        
        return images_cdf_pos, images_cdf_neg
    
    
    def rescale_cdf(self, images_cdf):
        
        ### add additonal normalization step for signed measures? e.g. P(X < x_max) = 1
        ### fix division for small values
        ### this rescale scheme converts t_negative to positive values, does this matter?
        ### some cdfs don't necessarily start at 0, e.g. camerman w/ ramp filter, (large mass early in pdf?)
        
        cdf_scale = images_cdf[:, -1, :]  # last value of each cdf
        # cdf_scale = np.where(cdf_scale<1e-8, 1e-8, cdf_scale)  # catch small values
        cdf_scale = cdf_scale.reshape(self.N, 1, self.n_theta)
        images_cdf = images_cdf / cdf_scale
        
        return images_cdf       
    
    
    def jordan_decomposition(self, t):
        """takes signed array t -> (t_+, t_-), both of size t"""

        ### make the t_n positive after decomp?
        ### need to normalize after?
        ### expensive to compute for large array stacks

        t_p = np.where(t >= 0, t, 0)
        t_n = np.where(t <= 0, t, 0)

        return t_p, t_n
    
    
    def get_uniform_inverse_cdf(self, images_cdf):
        ### this is a slow version, need to implement fast maybe using ndi.map_coords to vectorize interpolate
        ### add an option to increase the sampling on the xgrid
        
        images_icdf = np.zeros(images_cdf.shape)
        yvals = np.arange(self.n_points)
        
        for n, cdfs in enumerate(images_cdf):
            # start_values = cdfs[0, :]  ### EV: MAKE THIS ONE THE DEFAULT
            # end_values = cdfs[-1, :]  ### these should always be ones
            
            # ### EV: can pull next three lines out of loop if using this version
            start_values = np.zeros(self.n_lines)
            end_values = np.ones(self.n_lines) 
            # ###
            
            # ### EV: can pull next three lines out of loop if using this version
            # start_values = np.zeros(self.n_lines) + 0.01  ### *** EV: arbitrary threshould to avoid bound
            # end_values = np.ones(self.n_lines) - 0.01
            # ###
            
            ### EV: try prepend zero and append ones ?
            
            xgrid = np.linspace(start_values, end_values, self.n_points, endpoint=True)
            
            for idx in range(self.n_lines):
                icdf = np.interp(xgrid[:, idx], cdfs[:, idx], yvals)
                # icdf = np.interp(xgrid[:, idx], cdfs[:, idx], yvals, period=self.n_points)  ### *** EV: add period to interp??
                
                # ### EV: TEST THIS HACK TO ENFORCE BOUNDS
                icdf[0] = 0
                icdf[-1] = self.n_points
                # ###
                
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


def pdf_to_cdf(A):
    ### A is a matrix with columns as pdf (i.e. from Radon transform)
    
    A_cdf = np.cumsum(A, axis=0)
    A_norm = A_cdf[-1, :]
    A_cdf = A_cdf / A_norm
    
    return A_cdf


def cdf_to_icdf(A, n_points, n_theta):
    ### A is a matrix with columns as cdf
    ### this can be sped up for signed SW
    
    start_values = A[0, :]
    end_values = A[-1, :]

    xgrid = np.linspace(start_values, end_values, n_points, endpoint=True)
    yvals = np.arange(n_points)

    A_icdf = np.zeros((n_points, n_theta))

    for idx in range(n_theta):
        icdf = np.interp(xgrid[:, idx], A[:, idx], yvals)
        
        ### EV: Enforce bounds to ignore interpolation 
        icdf[0] = 0
        icdf[-1] = n_points
        ###
        
        A_icdf[:, idx] = icdf   
        
    return A_icdf
    
    
    
##### add/test these functions later
    
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