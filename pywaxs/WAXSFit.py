import sys, os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import lmfit
from lmfit.models import Model
from lmfit.lineshapes import gaussian2d, lorentzian
from scipy.stats import multivariate_normal
from scipy.ndimage import rotate
from scipy.ndimage import affine_transform
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

# Fixing the fractal noise function to match shape dimensions
def fractal_noise(shape, exponent):
    freqs = np.fft.fftfreq(shape[0]), np.fft.fftfreq(shape[1])
    freq_grid = np.meshgrid(freqs[0], freqs[1], indexing="ij")
    radius = np.sqrt(freq_grid[0]**2 + freq_grid[1]**2)
    amplitude = np.power(1.0 + radius**2, exponent / 2.0)
    phase = np.random.rand(*shape) * 2.0 * np.pi
    noise = np.fft.ifft2(amplitude * np.exp(1j * phase)).real
    return noise / noise.std()

# Updated function to generate random 2D Gaussian with noise and random orientation
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

'''
def generate_polar_gaussian(shape=(200, 200), coords=None, fwhm_qr=0.2, fwhm_chi=10.0, 
                            center_qr=1.0, center_chi=45.0, add_noise=False, add_background=False):
    
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
    
    # Apply random fractal noise
    if add_noise:
        noise = fractal_noise(shape, -1)
        noise = gaussian_filter(noise, sigma=3)
        z += 0.05 * noise
    
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
'''

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
    
    # Apply random fractal noise
    if add_noise:
        noise = fractal_noise(shape, -1)
        noise = gaussian_filter(noise, sigma=3)
        z += 0.05 * noise
    
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

        print(type(z))
        print(z.shape)
        # Original Data
        axs[0].imshow(z, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axs[0].set_title('Original Data')

        # Fit Data
        fit_data = self.model.func(x, y, **self.result.best_values).reshape(x.shape)
        # axs[1].imshow(fit_data, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axs[1].imshow(fit_data, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax, interpolation='none')
        axs[1].set_title('Fit')

        # Residuals
        residuals = z - fit_data
        axs[2].imshow(residuals, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axs[2].set_title('Residuals')

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