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
        ### - organize input parameters
        ### - make transforms apply to single image stacks
        ### - comments in functions (e.g. add oversample in icdf)
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
        
        n_trans = self.N  ### set this somewhere else?
        
        rads = self.angles / 180 * np.pi
        
        y_idx = np.arange(-self.n_points / 2, self.n_points / 2) / self.n_points * 2

        x_theta = y_idx[:, np.newaxis] * np.sin(rads)[np.newaxis, :]
        x_theta = np.pi * x_theta.flatten()

        y_theta = y_idx[:, np.newaxis] * np.cos(rads)[np.newaxis, :]
        y_theta = np.pi * y_theta.flatten()

        plan = finufft.Plan(nufft_type, (self.nx, self.ny), n_trans, eps)
        plan.setpts(x_theta, y_theta)
        
        self.freqs = np.abs(np.pi * y_idx)
        self.n_lines = len(rads)
        
        return plan
    

    def polar_nufft(self):
        """output ordered in regular grid with size (n_images x n_points x n_angles)"""
        
        plan = self.set_radial_2d_nufft_plan()
        images_ft = plan.execute(self.images).reshape(self.N, self.n_points, self.n_lines)
        
        if self.apply_ramp:
            images_ft *= self.freqs[:, np.newaxis]
        
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
        images_cdf = self.get_cdf(images_rt)
        
        if rescale:
            images_cdf = self.rescale_cdf(images_cdf)
        
        return images_cdf
    
    
    def signed_cdf_transform(self, rescale=True):
        
        ### assertion doesn't catch if individual images in stack are all positive, move assertion to the decomp?
        
        images_rt = self.radon_transform()
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
        ### this is a slow version, need to implement fast using ndi.map_coords to vectorize interpolate
        ### need to add an option to increase the sampling on the xgrid
        ### make work with single image
        
        images_icdf = np.zeros(images_cdf.shape)
        yvals = np.arange(self.n_points)
        
        for n, cdfs in enumerate(images_cdf):
            start_values = cdfs[0, :]
            # start_values = np.zeros(cdfs[0, :].shape)  # to match the orignal
            end_values = np.ones(start_values.shape)
            xgrid = np.linspace(start_values, end_values, self.n_points)
            
            for idx in range(self.n_lines):
                icdf = np.interp(xgrid[:, idx], cdfs[:, idx], yvals)
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