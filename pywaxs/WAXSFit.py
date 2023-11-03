import sys, os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import lmfit
from lmfit import Model, Parameters
from lmfit.lineshapes import gaussian2d, lorentzian
from scipy.stats import multivariate_normal
from scipy.ndimage import rotate
from scipy.ndimage import affine_transform
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import gmean
import hdbscan
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN

# Fixing the fractal noise function to match shape dimensions
def fractal_noise(shape, exponent):
    freqs = np.fft.fftfreq(shape[0]), np.fft.fftfreq(shape[1])
    freq_grid = np.meshgrid(freqs[0], freqs[1], indexing="ij")
    radius = np.sqrt(freq_grid[0]**2 + freq_grid[1]**2)
    amplitude = np.power(1.0 + radius**2, exponent / 2.0)
    phase = np.random.rand(*shape) * 2.0 * np.pi
    noise = np.fft.ifft2(amplitude * np.exp(1j * phase)).real
    return noise / noise.std()

def generate_random_2D_gaussian(shape=(50, 50), coords=None, sigma_1=None, sigma_2=None, angle = None):
    shape_x = np.random.randint(50, 101)
    shape_y = np.random.randint(50, 101)
    shape = (shape_x, shape_y)
    
    if coords is None:
        coords = {'q_xy': np.linspace(-1, 1, shape[1]), 'q_z': np.linspace(-1, 1, shape[0])}
    
    x, y = np.meshgrid(coords['q_xy'], coords['q_z'])
    pos = np.dstack((x, y))
    
    # If sigma_1 and sigma_2 are not provided, set them to random values
    if sigma_1 is None:
        sigma_1 = np.random.rand()
    if sigma_2 is None:
        sigma_2 = np.random.rand()
    
    # Constraints to keep Gaussian within frame
    buffer = 3.5 * max(sigma_1, sigma_2)
    min_center = min(coords['q_xy']) + buffer, min(coords['q_z']) + buffer
    max_center = max(coords['q_xy']) - buffer, max(coords['q_z']) - buffer
    
    # Generate random mean and covariance matrix
    mean = [np.random.uniform(min_center[0], max_center[0]), np.random.uniform(min_center[1], max_center[1])]
    cov_matrix = [[sigma_1, 0], [0, sigma_2]]
    
    # Generate 2D Gaussian
    z = multivariate_normal(mean, cov_matrix).pdf(pos)
    
    # Add random background noise
    noise_level = np.random.uniform(0.01, 0.05)
    z += np.random.normal(0, noise_level, z.shape)
    
    # Generate random rotation matrix
    # angle = np.random.uniform(0, np.pi)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    
    # Apply rotation to Gaussian only
    mean_rot = np.dot(rotation_matrix, mean)
    cov_rot = np.dot(np.dot(rotation_matrix, cov_matrix), rotation_matrix.T)
    z_rotated = multivariate_normal(mean_rot, cov_rot).pdf(pos)
    
    # Add random background noise again to rotated Gaussian
    z_rotated += np.random.normal(0, noise_level, z_rotated.shape)
    
    # Add linear background
    max_intensity = np.max(z_rotated)
    background_intensity = np.random.uniform(0.02 * max_intensity, 0.15 * max_intensity)
    axis_choice = np.random.choice(['x', 'y'])
    
    if axis_choice == 'x':
        background = background_intensity * (coords['q_xy'] - coords['q_xy'].min()) / (coords['q_xy'].max() - coords['q_xy'].min())
        z_rotated += background[None, :]
    else:
        background = background_intensity * (coords['q_z'] - coords['q_z'].min()) / (coords['q_z'].max() - coords['q_z'].min())
        z_rotated += background[:, None]
    
    # Create xarray DataArray and Dataset
    # data_array = xr.DataArray(z_rotated, coords=[('q_z', coords['q_z']), ('q_xy', coords['q_xy'])], name='intensity')
    
    # Create xarray DataArray and Dataset
    data_array = xr.DataArray(z, coords=[('q_z', coords['q_z']), ('q_xy', coords['q_xy'])], name='intensity')
    data_array.attrs['sigma_1'] = sigma_1
    data_array.attrs['sigma_2'] = sigma_2
    
    # Add attributes
    data_array.attrs['noise_level'] = noise_level
    data_array.attrs['background_intensity'] = background_intensity
    data_array.attrs['background_axis'] = axis_choice
    data_array.attrs['sigma_1'] = sigma_1
    data_array.attrs['sigma_2'] = sigma_2
    data_array.attrs['rotation_angle'] = angle
    data_array.attrs['min_center'] = min_center
    data_array.attrs['max_center'] = max_center
    data_array.attrs['position'] = pos

    dataset = xr.Dataset({'intensity': data_array})
    dataset['intensity'].attrs = data_array.attrs

    return dataset

def generate_polar_gaussian(shape=None, coords=None, fwhm_qr=0.2, fwhm_chi=10.0, 
                            center_qr=1.0, center_chi=45.0, add_noise=False, add_background=False):
    if shape is None:
        shape_x = np.random.randint(50, 101)
        shape_y = np.random.randint(50, 101)
        # shape_x = np.random.randint(20, 51)
        # shape_y = np.random.randint(20, 51)
        shape = (shape_x, shape_y)
    
    if coords is None:
        coords = {'q_xy': np.linspace(0, 2, shape[1]), 'q_z': np.linspace(0, 2, shape[0])}
    
    x, y = np.meshgrid(coords['q_xy'], coords['q_z'])
    q_r = np.sqrt(x**2 + y**2)
    chi = np.degrees(np.arctan2(y, x))
    
    # Convert FWHM to sigma
    sigma_qr = fwhm_qr / np.sqrt(8 * np.log(2))
    sigma_chi = fwhm_chi / np.sqrt(8 * np.log(2))
    
    # Create 2D Gaussian in polar coordinates
    z = np.exp(-((q_r - center_qr)**2 / (2 * sigma_qr ** 2) + (chi - center_chi)**2 / (2 * sigma_chi ** 2)))

    # Apply random Poissonian noise
    if add_noise:
        noise = np.random.poisson(np.max(z), shape)  # Poisson noise based on max intensity
        scaled_noise = 0.05 * noise / np.max(noise)  # Scale so that max is 5% of signal max
        z += scaled_noise

    # Add 'halo'-like quadratic background
    if add_background:
        qr_min = np.random.uniform(0.75, 1.25)
        qr_max = np.random.uniform(1.75, 2.25)
        mask = (q_r >= qr_min) & (q_r <= qr_max)
        max_intensity = np.max(z)
        background_intensity = np.random.uniform(0.02 * max_intensity, 0.1 * max_intensity)
        background = background_intensity * ((q_r - center_qr)**2)
        z += mask * background
    
    # Create xarray DataArray and Dataset
    data_array = xr.DataArray(z, coords=[('q_z', coords['q_z']), ('q_xy', coords['q_xy'])], name='intensity')
    
    # Add attributes
    data_array.attrs['fwhm_qr'] = fwhm_qr
    data_array.attrs['fwhm_chi'] = fwhm_chi
    data_array.attrs['center_qr'] = center_qr
    data_array.attrs['center_chi'] = center_chi

    return xr.Dataset({'intensity': data_array})

def generate_combined_gaussian(shape=(100, 100), coords=None, 
                               fwhm_qr=0.2, fwhm_chi=10.0, center_qr=1.0, center_chi=45.0,
                               fwhm_qxy=0.1, fwhm_qz=0.1,
                               cartesian_weight=0.5, add_noise=False, add_background=False):
    
    # Validate and set Cartesian Gaussian weight
    if not 0 <= cartesian_weight <= 1:
        print("Invalid cartesian_weight value. Using default value of 0.5.")
        cartesian_weight = 0.5
    polar_weight = 1 - cartesian_weight
    
    if coords is None:
        coords = {'q_xy': np.linspace(0, 2, shape[1]), 'q_z': np.linspace(0, 2, shape[0])}
    
    # Convert center_qr and center_chi to center_qxy and center_qz
    center_chi_rad = np.radians(center_chi)
    center_qxy = center_qr * np.cos(center_chi_rad)
    center_qz = center_qr * np.sin(center_chi_rad)

    # Generate meshgrid
    x, y = np.meshgrid(coords['q_xy'], coords['q_z'])
    
    # Calculate qr and chi (polar coordinates) based on x and y
    q_r = np.sqrt(x**2 + y**2)
    chi = np.degrees(np.arctan2(y, x))
    
    # Convert FWHM to sigma for the polar Gaussian
    sigma_qr = fwhm_qr / np.sqrt(8 * np.log(2))
    sigma_chi = fwhm_chi / np.sqrt(8 * np.log(2))
    
    # Create 2D Gaussian in polar coordinates
    polar_gaussian = np.exp(-((q_r - center_qr)**2 / (2 * sigma_qr ** 2) + 
                              (chi - center_chi)**2 / (2 * sigma_chi ** 2)))
    
    # Convert FWHM to sigma for the Cartesian Gaussian
    sigma_qxy = fwhm_qxy / np.sqrt(8 * np.log(2))
    sigma_qz = fwhm_qz / np.sqrt(8 * np.log(2))
    
    # Create and normalize 2D Gaussians
    polar_gaussian = np.exp(-((q_r - center_qr)**2 / (2 * sigma_qr ** 2) + 
                              (chi - center_chi)**2 / (2 * sigma_chi ** 2)))
    polar_gaussian /= np.max(polar_gaussian)

    cartesian_gaussian = np.exp(-((x - center_qxy)**2 / (2 * sigma_qxy ** 2) + 
                                  (y - center_qz)**2 / (2 * sigma_qz ** 2)))
    cartesian_gaussian /= np.max(cartesian_gaussian)

    # Weighted sum of the two normalized Gaussians
    combined_gaussian = cartesian_weight * cartesian_gaussian + polar_weight * polar_gaussian
    combined_gaussian /= np.max(combined_gaussian)
    
    # Apply random Poissonian noise
    if add_noise:
        noise = np.random.poisson(np.max(combined_gaussian), shape)  # Poisson noise based on max intensity
        scaled_noise = 0.02 * noise / np.max(noise)  # Scale so that max is 5% of signal max
        combined_gaussian += scaled_noise
    
    # Add background
    if add_background:
        background_intensity = np.random.uniform(0.02 * np.max(combined_gaussian), 
                                                 0.1 * np.max(combined_gaussian))
        background = background_intensity * ((q_r - center_qr)**2)
        combined_gaussian += background
    
    # Create xarray DataArray and Dataset
    data_array = xr.DataArray(combined_gaussian, 
                              coords=[('q_z', coords['q_z']), ('q_xy', coords['q_xy'])], 
                              name='intensity')

    # data_array = xr.DataArray(z, coords=[('q_z', coords['q_z']), ('q_xy', coords['q_xy'])], name='intensity')

    
    # Add attributes
    data_array.attrs['fwhm_qr'] = fwhm_qr
    data_array.attrs['fwhm_chi'] = fwhm_chi
    data_array.attrs['center_qr'] = center_qr
    data_array.attrs['center_chi'] = center_chi
    data_array.attrs['fwhm_qxy'] = fwhm_qxy
    data_array.attrs['fwhm_qz'] = fwhm_qz
    data_array.attrs['center_qxy'] = center_qxy
    data_array.attrs['center_qz'] = center_qz

    return xr.Dataset({'intensity': data_array})

class WAXSFit:
    '''WAXSFit: Fit a 2D Gaussian of a selected ROI dataarray that is passed to the WAXSFit class instance. Return an ROI Fit dataarray, along with corresponding fit attributes.
    Adapted From: https:/x/lmfit.github.io/lmfit-py/examples/example_two_dimensional_peak.html '''

    def __init__(self, data_array: xr.DataArray):
        # Check for NaN values and replace them with zeros
        nan_mask = np.isnan(data_array['intensity'].values)
        self.nan_mask = xr.DataArray(nan_mask, coords=data_array['intensity'].coords, name='nan_mask')
        data_array['intensity'] = xr.where(xr.DataArray(nan_mask, coords=data_array['intensity'].coords), 0, data_array['intensity'])

        self.data_array = data_array
        self.result = None
        self.model = None
        self.combined_fitter = None

    def perform_combined_gaussian_fit(self):
        # Create an instance of CombinedGaussianFitter
        self.combined_fitter = CombinedGaussianFitter(self.data_array)

        # Construct the model
        # self.combined_fitter.construct_model()

        # Run the fitting procedure
        self.combined_fitter.perform_combined_fit()

        # Store the result and model in WAXSFit for further analysis
        self.model = self.combined_fitter.model
        self.result = self.combined_fitter.result

    @staticmethod
    def polar_gaussian_2D(q_xy, q_z, fwhm_qr, fwhm_chi, center_qr, center_chi):
        # Convert to qr and chi
        qr = np.sqrt(q_xy**2 + q_z**2)
        chi = np.degrees(np.pi / 2 - np.arctan2(q_z, q_xy))
        
        # Convert FWHM to sigma
        sigma_qr = fwhm_qr / np.sqrt(8 * np.log(2))
        sigma_chi = fwhm_chi / np.sqrt(8 * np.log(2))
        #     gaussian_polar = I0 * np.exp(-4 * np.log(2) * ((qr - center_qr)**2 / fwhm_qr**2 + (chi - center_chi)**2 / fwhm_chi**2))

        # Gaussian in polar coordinates
        return np.exp(-((qr - center_qr)**2 / (2 * sigma_qr ** 2) + (chi - center_chi)**2 / (2 * sigma_chi ** 2)))

    def construct_azimuth_model(self):
        self.model = Model(self.polar_gaussian_2D, independent_vars=['q_xy', 'q_z'])

    def perform_azimuth_2dgauss_fit(self):
        q_xy_vals = self.data_array.coords['q_xy'].values
        q_z_vals = self.data_array.coords['q_z'].values
        q_xy, q_z = np.meshgrid(q_xy_vals, q_z_vals)
        intensity = self.data_array['intensity'].values

        # Convert to qr and chi
        qr_vals = np.sqrt(q_xy ** 2 + q_z ** 2)
        chi_vals = np.degrees(np.pi / 2 - np.arctan2(q_z, q_xy))

        # Estimate center and FWHM
        center_qr_est = np.average(qr_vals, weights=intensity)
        center_chi_est = np.average(chi_vals, weights=intensity)
        fwhm_qr_est = 2.0 * np.sqrt(np.average((qr_vals - center_qr_est) ** 2, weights=intensity))
        fwhm_chi_est = 2.0 * np.sqrt(np.average((chi_vals - center_chi_est) ** 2, weights=intensity))

        print(f"Initial estimates: center_qr = {center_qr_est}, center_chi = {center_chi_est}, fwhm_qr = {fwhm_qr_est}, fwhm_chi = {fwhm_chi_est}")

        # Check for NaN or Inf in intensity and weights
        if np.any(np.isnan(intensity)) or np.any(np.isinf(intensity)):
            print("Warning: NaN or Inf detected in intensity.")
            
        if np.any(np.isnan(qr_vals)) or np.any(np.isinf(qr_vals)) or np.any(np.isnan(chi_vals)) or np.any(np.isinf(chi_vals)):
            print("Warning: NaN or Inf detected in qr_vals or chi_vals.")
            
        params = self.model.make_params(I0=np.max(intensity), 
                                        center_qr=center_qr_est, 
                                        center_chi=center_chi_est, 
                                        fwhm_qr=fwhm_qr_est, 
                                        fwhm_chi=fwhm_chi_est)

        params['center_qr'].set(min=0)
        params['center_chi'].set(min=0, max=90)
        params['fwhm_qr'].set(min=0)
        params['fwhm_chi'].set(min=0, max=90)

        # Test the model with initial parameters
        test_model_vals = self.model.func(q_xy.ravel(), q_z.ravel(), **params.valuesdict())
        print(f"First few model values with initial parameters: {test_model_vals[:10]}")
        
        error = np.sqrt(intensity + 1)
        self.result = self.model.fit(intensity.ravel(), q_xy=q_xy.ravel(), q_z=q_z.ravel(), params=params, weights=1/np.sqrt(error.ravel()))

    @staticmethod
    def gaussian2d_rotated(x, y, amplitude, centerx, centery, sigmax, sigmay, rotation):
        xp = (x - centerx) * np.cos(rotation) - (y - centery) * np.sin(rotation)
        yp = (x - centerx) * np.sin(rotation) + (y - centery) * np.cos(rotation)
        g = amplitude * np.exp(-((xp/sigmax)**2 + (yp/sigmay)**2) / 2.)
        return g
    
    def construct__linear_2dgauss_model(self):
        self.model = Model(self.gaussian2d_rotated, independent_vars=['x', 'y'])

    def perform_linear_2dgauss_fit(self):
        x_vals = self.data_array.coords['q_xy'].values
        y_vals = self.data_array.coords['q_z'].values
        x, y = np.meshgrid(x_vals, y_vals)
        z = self.data_array['intensity'].values
        # z = self.data_array.values

        params = self.model.make_params(amplitude=np.max(z), centerx=np.mean(x_vals), centery=np.mean(y_vals),
                                        sigmax=np.std(x_vals), sigmay=np.std(y_vals), rotation=0)
        
        params['rotation'].set(value = .1, min=0, max=np.pi)
        params['sigmax'].set(min=0)
        params['sigmay'].set(min=0)

        error = np.sqrt(z + 1)

        self.result = self.model.fit(z.ravel(), x=x.ravel(), y=y.ravel(), params=params, weights=1/np.sqrt(error.ravel()))

    def calculate_residuals(self):
        if self.result is None:
            return None

        x, y = np.meshgrid(self.data_array.coords['q_xy'], self.data_array.coords['q_z'])
        fit_data = self.model.func(x.flatten(), y.flatten(), **self.result.best_values)
        return self.data_array['intensity'].values.flatten() - fit_data

    def plot_fit(self):
        if self.result is None:
            return

        x, y = np.meshgrid(self.data_array.coords['q_xy'], self.data_array.coords['q_z'])
        z = self.data_array['intensity'].values

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Set the same contrast scaling based on the main dataarray
        vmin = np.min(z)
        vmax = np.max(z)

        # Font size settings
        xfontsize = 12
        yfontsize = 12

        # Original Data
        im0 = axs[0].imshow(z, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axs[0].set_title('Original Data')
        axs[0].set_xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)', fontsize=xfontsize)
        axs[0].set_ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)', fontsize=yfontsize)
        fig.colorbar(im0, ax=axs[0], shrink=0.5)

        # Fit Data
        fit_data = self.model.func(x, y, **self.result.best_values).reshape(x.shape)
        im1 = axs[1].imshow(fit_data, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax, interpolation='none')
        axs[1].set_title('Fit')
        axs[1].set_xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)', fontsize=xfontsize)
        axs[1].set_ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)', fontsize=yfontsize)
        fig.colorbar(im1, ax=axs[1], shrink=0.5)

        # Residuals
        residuals = z - fit_data
        im2 = axs[2].imshow(residuals, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axs[2].set_title('Residuals')
        axs[2].set_xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)', fontsize=xfontsize)
        axs[2].set_ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)', fontsize=yfontsize)
        fig.colorbar(im2, ax=axs[2], shrink=0.5)

        plt.show()

    def fit_statistics(self):
        if self.result is None:
            return None

        # Create an empty dictionary to store fit statistics
        fit_stats_dict = {}

        # Chi-square
        fit_stats_dict['chi_square'] = self.result.chisqr

        # Akaike Information Criterion
        fit_stats_dict['aic'] = self.result.aic

        # Bayesian Information Criterion
        fit_stats_dict['bic'] = self.result.bic

        # Parameters and their uncertainties
        for param_name, param in self.result.params.items():
            fit_stats_dict[f"{param_name}_value"] = param.value
            fit_stats_dict[f"{param_name}_stderr"] = param.stderr

        return fit_stats_dict

class CombinedGaussianFitter:
    def __init__(self, data_array: xr.DataArray):
        # Check for NaN values and replace them with zeros
        nan_mask = np.isnan(data_array['intensity'].values)
        self.nan_mask = xr.DataArray(nan_mask, coords=data_array['intensity'].coords, name='nan_mask')
        data_array['intensity'] = xr.where(xr.DataArray(nan_mask, coords=data_array['intensity'].coords), 0, data_array['intensity'])

        self.data_array = data_array
        self.result = None
        self.model = None
        self.residuals = None
        self.percent_residuals = None

    @staticmethod
    def combined_gaussian_2D(q_xy, q_z, fwhm_qr, fwhm_chi, center_qr, center_chi, fwhm_qxy, fwhm_qz, weight):
        # Convert to qr and chi
        qr = np.sqrt(q_xy**2 + q_z**2)
        chi = np.degrees(np.pi / 2 - np.arctan2(q_z, q_xy))

        # Convert FWHM to sigma
        sigma_qr = fwhm_qr / np.sqrt(8 * np.log(2))
        sigma_chi = fwhm_chi / np.sqrt(8 * np.log(2))
        sigma_qxy = fwhm_qxy / np.sqrt(8 * np.log(2))
        sigma_qz = fwhm_qz / np.sqrt(8 * np.log(2))

        # Gaussian in polar and Cartesian coordinates
        polar_gaussian = np.exp(-((qr - center_qr)**2 / (2 * sigma_qr ** 2) + (chi - center_chi)**2 / (2 * sigma_chi ** 2)))
        cartesian_gaussian = np.exp(-((q_xy - center_qr*np.cos(np.radians(center_chi)))**2 / (2 * sigma_qxy ** 2) + 
                                       (q_z - center_qr*np.sin(np.radians(center_chi)))**2 / (2 * sigma_qz ** 2)))

        # Combined Gaussian with weight
        return (1 - weight) * polar_gaussian + weight * cartesian_gaussian

    def construct_model(self):
        self.model = Model(self.combined_gaussian_2D, independent_vars=['q_xy', 'q_z'])

    def initial_peak_identification(self, img_xr, threshold_ratio):
        # Initialize variables
        max_intensity = np.max(img_xr)
        mean_intensity = np.mean(img_xr)
        median_intensity = np.median(img_xr)
        min_intensity = np.min(img_xr)
        noise_level = np.std(img_xr[img_xr < median_intensity])
        snr = (max_intensity - median_intensity) / noise_level
        # threshold_ratio = initial_threshold_ratio
        found_peaks = False

        # Adaptive thresholding and sigma adjustment
        while not found_peaks and threshold_ratio <= 1.0:
            sigma1 = snr * 0.5
            sigma2 = snr * 1.5
            img_smooth1 = gaussian_filter(np.nan_to_num(img_xr), sigma=sigma1)
            img_smooth2 = gaussian_filter(np.nan_to_num(img_xr), sigma=sigma2)

            dog = img_smooth1 - img_smooth2
            threshold = max(threshold_ratio * max_intensity, 0.1 * max_intensity)

            # Identify peak locations
            peaks = np.where(dog >= threshold)

            if peaks[0].size > 0:
                found_peaks = True
                break
            else:
                threshold_ratio += 0.1  # Increment threshold ratio

        # Proceed with clustering if peaks are found
        if found_peaks:
            coords = np.column_stack(peaks)
            clustering = DBSCAN(eps=3, min_samples=2).fit(coords)
            labels = clustering.labels_
            # cluster_means = [geometric_mean(coords[labels == i], axis=0) for i in set(labels) if i != -1]
            
            # Calculate geometric mean of each cluster
            cluster_means = [gmean(coords[labels == i], axis=0) for i in set(labels) if i != -1]

            if cluster_means:
                center_estimate = max(cluster_means, key=lambda x: img_xr[tuple(x.astype(int))])
            else:
                center_estimate = [np.nan, np.nan]
        else:
            center_estimate = [np.nan, np.nan]

        return center_estimate, dog, peaks

    def initial_fit(self):
        q_xy_vals = self.data_array.coords['q_xy'].values
        q_z_vals = self.data_array.coords['q_z'].values
        q_xy, q_z = np.meshgrid(q_xy_vals, q_z_vals)
        intensity = self.data_array['intensity'].values

        # Ensure shapes match
        if q_xy.shape != intensity.shape or q_z.shape != intensity.shape:
            raise ValueError("Shape mismatch between q_xy, q_z, and intensity arrays.")

        # Ensure the model is expecting these variables
        if 'q_xy' not in self.model.independent_vars or 'q_z' not in self.model.independent_vars:
            raise ValueError("Model does not expect q_xy and q_z as independent variables.")

        # Convert to qr and chi
        qr_vals = np.sqrt(q_xy ** 2 + q_z ** 2)
        chi_vals = np.degrees(np.pi / 2 - np.arctan2(q_z, q_xy))

        # Call the peak identification method
        # center_estimate, dog, peaks = self.initial_peak_identification(intensity, threshold_ratio=0.5)
        center_estimate = np.nan

        if not np.isnan(center_estimate).any():
            # Convert the center estimate to qr and chi values
            center_qr_est, center_chi_est = self._convert_to_qr_chi(center_estimate)
        else:
            # Fallback to maximum intensity if no peaks are found
            max_intensity_idx = np.unravel_index(np.argmax(intensity), intensity.shape)
            center_qr_est, center_chi_est = qr_vals[max_intensity_idx], chi_vals[max_intensity_idx]

        fwhm_qr_est = 2.0 * np.sqrt(np.average((qr_vals - center_qr_est) ** 2, weights=intensity))
        fwhm_chi_est = 2.0 * np.sqrt(np.average((chi_vals - center_chi_est) ** 2, weights=intensity))

        # Check for NaN or Inf in intensity and weights
        if np.any(np.isnan(intensity)) or np.any(np.isinf(intensity)):
            print("Warning: NaN or Inf detected in intensity.")

        if np.any(np.isnan(qr_vals)) or np.any(np.isinf(qr_vals)) or np.any(np.isnan(chi_vals)) or np.any(np.isinf(chi_vals)):
            print("Warning: NaN or Inf detected in qr_vals or chi_vals.")

        # Set up initial parameters
        params = self.model.make_params(I0=np.max(intensity), 
                                        center_qr=center_qr_est, 
                                        center_chi=center_chi_est, 
                                        fwhm_qr=fwhm_qr_est, 
                                        fwhm_chi=fwhm_chi_est)

        params['center_qr'].set(min=0)
        params['center_chi'].set(min=0, max=90)
        params['fwhm_qr'].set(min=0)
        params['fwhm_chi'].set(min=0, max=90)
        params.add('fwhm_qxy', value=0.2, min=0, vary=False)
        params.add('fwhm_qz', value=0.2, min=0, vary=False)
        params.add('weight', value=0, min=0, max=1, vary=False)

        error = np.sqrt(intensity + 1)

        # Perform initial fit
        self.result = self.model.fit(intensity.ravel(), q_xy=q_xy.ravel(), q_z=q_z.ravel(), params=params, weights=1/np.sqrt(error.ravel()))

    def intermediate_fit(self):
        # Unlock weight and lock fwhm_qr, fwhm_chi, center_qr, center_chi
        self.result.params['weight'].set(value=0.5, min=0, max=1, vary=True)
        for param in ['fwhm_qr', 'fwhm_chi', 'center_qr', 'center_chi']:
            self.result.params[param].set(vary=False)

        # Perform intermediate fit
        q_xy_vals = self.data_array.coords['q_xy'].values
        q_z_vals = self.data_array.coords['q_z'].values
        q_xy, q_z = np.meshgrid(q_xy_vals, q_z_vals)
        intensity = self.data_array['intensity'].values
        # intensity = self.data_array['intensity'].values.ravel()
        error = np.sqrt(intensity + 1)

        self.result = self.model.fit(intensity.ravel(), q_xy=q_xy.ravel(), q_z=q_z.ravel(), params=self.result.params, weights=1/np.sqrt(error.ravel()))

    def final_fit(self):
        # Unlock all parameters and adjust bounds
        for param in self.result.params:
            self.result.params[param].set(vary=True)
            if param != 'weight':
                self.result.params[param].set(min=self.result.params[param].value * 0.8, 
                                            max=self.result.params[param].value * 1.2)

        # Set FWHM of qxy and qz to qr and lock them
        fwhm_qr_current_value = self.result.params['fwhm_qr'].value
        self.result.params['fwhm_qxy'].set(value=fwhm_qr_current_value, vary=False)
        self.result.params['fwhm_qz'].set(value=fwhm_qr_current_value, vary=False)

        # Perform final fit
        q_xy_vals = self.data_array.coords['q_xy'].values
        q_z_vals = self.data_array.coords['q_z'].values
        q_xy, q_z = np.meshgrid(q_xy_vals, q_z_vals)
        intensity = self.data_array['intensity'].values

        error = np.sqrt(intensity + 1)

        self.result = self.model.fit(intensity.ravel(), q_xy=q_xy.ravel(), q_z=q_z.ravel(), params=self.result.params, weights=1/np.sqrt(error.ravel()))

    def perform_combined_fit(self, fit_method='gaussianpolarcartesian'):
        self.construct_model()

        if fit_method == 'gaussianpolar':
            self.initial_fit()
        elif fit_method == 'gaussianpolarcartesian':
            self.initial_fit()
            self.intermediate_fit()
            self.final_fit()
        elif fit_method == 'gaussiancartesian':
            # Placeholder for gaussiancartesian fitting method
            pass
        else:
            print("Invalid fitting method selected. Defaulting to gaussianpolarcartesian.")
            self.initial_fit()
            self.intermediate_fit()
            self.final_fit()

    def get_fit_result(self):
        return self.result

    def calculate_residuals(self):
        if self.result is None:
            return None

        x, y = np.meshgrid(self.data_array.coords['q_xy'], self.data_array.coords['q_z'])
        fit_data = self.model.func(x.flatten(), y.flatten(), **self.result.best_values)
        self.residuals = self.data_array['intensity'].values.flatten() - fit_data

        # Calculate residuals as percentage of maximum intensity
        max_intensity = np.max(self.data_array['intensity'].values)
        self.percent_residuals = (self.residuals / max_intensity) * 100

        return self.percent_residuals

    def plot_fit(self):
        if self.result is None:
            return

        x, y = np.meshgrid(self.data_array.coords['q_xy'], self.data_array.coords['q_z'])
        z = self.data_array['intensity'].values

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Set the contrast scaling based on percentiles of the main data array
        vmin = np.nanpercentile(z, 10)
        vmax = np.nanpercentile(z, 99)

        # Font size settings
        fontsize = 12

        # Original Data
        im0 = axs[0].imshow(z, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axs[0].set_title('Original Data')
        axs[0].set_xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)', fontsize=fontsize)
        axs[0].set_ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)', fontsize=fontsize)
        fig.colorbar(im0, ax=axs[0], shrink=0.5)

        # Fit Data
        fit_data = self.model.func(x, y, **self.result.best_values).reshape(x.shape)
        im1 = axs[1].imshow(fit_data, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax, interpolation='none')
        axs[1].set_title('Fit')
        axs[1].set_xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)', fontsize=fontsize)
        axs[1].set_ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)', fontsize=fontsize)
        fig.colorbar(im1, ax=axs[1], shrink=0.5)

        # Percent Residuals
        percent_residuals = self.percent_residuals.reshape(x.shape)
        im2 = axs[2].imshow(percent_residuals, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis', vmin=np.nanpercentile(percent_residuals, 10), vmax=np.nanpercentile(percent_residuals, 99))
        axs[2].set_title('Percent Residuals')
        axs[2].set_xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)', fontsize=fontsize)
        axs[2].set_ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)', fontsize=fontsize)
        fig.colorbar(im2, ax=axs[2], shrink=0.5)

    def fit_statistics(self):
        if self.result is None:
            return None

        # Create an empty dictionary to store fit statistics
        fit_stats_dict = {}

        # Chi-square
        fit_stats_dict['chi_square'] = self.result.chisqr

        # Akaike Information Criterion
        fit_stats_dict['aic'] = self.result.aic

        # Bayesian Information Criterion
        fit_stats_dict['bic'] = self.result.bic

        # Parameters and their uncertainties
        for param_name, param in self.result.params.items():
            fit_stats_dict[f"{param_name}_value"] = param.value
            fit_stats_dict[f"{param_name}_stderr"] = param.stderr

        return fit_stats_dict

''' def plot_fit(self)
    # def plot_fit(self):
    #     if self.result is None:
    #         return

    #     x, y = np.meshgrid(self.data_array.coords['q_xy'], self.data_array.coords['q_z'])
    #     z = self.data_array['intensity'].values

    #     fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    #     # Set the same contrast scaling based on the main dataarray
    #     vmin = np.min(z)
    #     vmax = np.max(z)

    #     # Font size settings
    #     xfontsize = 12
    #     yfontsize = 12

    #     # Original Data
    #     im0 = axs[0].imshow(z, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    #     axs[0].set_title('Original Data')
    #     axs[0].set_xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)', fontsize=xfontsize)
    #     axs[0].set_ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)', fontsize=yfontsize)
    #     fig.colorbar(im0, ax=axs[0], shrink=0.5)

    #     # Fit Data
    #     fit_data = self.model.func(x, y, **self.result.best_values).reshape(x.shape)
    #     im1 = axs[1].imshow(fit_data, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax, interpolation='none')
    #     axs[1].set_title('Fit')
    #     axs[1].set_xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)', fontsize=xfontsize)
    #     axs[1].set_ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)', fontsize=yfontsize)
    #     fig.colorbar(im1, ax=axs[1], shrink=0.5)

    #     # Residuals
    #     residuals = z - fit_data
    #     im2 = axs[2].imshow(residuals, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    #     axs[2].set_title('Residuals')
    #     axs[2].set_xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)', fontsize=xfontsize)
    #     axs[2].set_ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)', fontsize=yfontsize)
    #     fig.colorbar(im2, ax=axs[2], shrink=0.5)

    #     plt.show()
'''