import os, pathlib, tifffile, pyFAI, pygix, json, zarr, random, inspect
from PIL import Image
from typing import Union, Tuple, Optional
import matplotlib.pyplot as plt
from tifffile import TiffWriter
import xarray as xr
import numpy as np
from numpy.polynomial.polynomial import Polynomial
# -- SciPy Modules 
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_closing
from scipy.signal import find_peaks
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
from scipy.spatial import KDTree
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix
from scipy.interpolate import interp1d
import scipy.ndimage as ndi
# -- SciKit Modules 
from skimage import feature
from sklearn.neighbors import KDTree
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
from skimage.filters import sobel
from skimage.feature import canny
from sklearn.cluster import DBSCAN
# -- Hierarchical Density-Based Spatial Clustering 
import hdbscan
from hdbscan import HDBSCAN
from collections import defaultdict
from collections import Counter
from collections import namedtuple
from IPython.display import clear_output
from typing import Optional
import gc
from pathlib import Path
from datetime import datetime
# from matplotlib.path import Path
from matplotlib.path import Path as MatplotlibPath


# - Custom Imports
from WAXSTransform import WAXSTransform

class WAXSReduce:
    def __init__(self, 
                 poniPath: Union[str, pathlib.Path] = None, 
                 maskPath: Union[str, pathlib.Path, np.ndarray]  = None,
                 tiffPath: Union[str, pathlib.Path, np.ndarray]  = None,
                 metadata_keylist=[], 
                 inplane_config: str = 'q_xy', 
                 energy: float = 12.7,
                 incident_angle: float = 0.3,
                #  zarrPath: Union[str, pathlib.Path] = None,
                 hdf5Path: Union[str, pathlib.Path] = None,
                 projectName: str = 'test'):
       
        """
        Attributes:
        poniPath (pathlib Path or str): Path to .poni file for converting to q-space 
                                        & applying missing wedge correction
        maskPath (pathlib Path or str or np.array): Path to the mask file to use 
                                for the conversion, or a numpy array

        Description: Initialize instance with metadata key list. Default is an empty list.
        """
    
        # -- PATH INFORMATION -- ##
        self.basePath = None # datatype: 'str' or pathlib variable, 'root_folder' basePath used throughout the class methods to build additional paths.
        self.poniPath = poniPath # datatype: 'str' or pathlib variable, PONI File Path ('.poni')
        self.maskPath = maskPath # datatype: 'str' or pathlib variable, MASK File Path ('.edf' or '.json')
        self.tiffPath = tiffPath # datatype: 'str' or pathlib variable, TIFF image filepath
        self.inplane_config = inplane_config # datatype: 'str', in-plane scattering axes label
        self.energy = energy # datatype: float, energy of your X-rays in keV
        self.incident_angle = incident_angle

        # -- METADATA ATTRIBUTES -- ##
        self.metadata_keylist = metadata_keylist # datatype: list, 'md_naming_scheme'
        self.attribute_dict = None # datatype: dictionary, 'sample_dict'

        # -- TIFF IMAGE DATA -- ##
        self.rawtiff_np = None # datatype: numpy array, RAW TIFF (numpy)
        self.rawtiff_xr = None # datatype: xarray DataArray, RAW TIFF (xarray)

        # - Load the Single Image
        self.loadSingleImage(self.tiffPath)
        # self.loadMetaData(self.tiffPath, delim='_') # this is done in the loadSingleImage() method

        # - Reciprocal Space Image Corrections Data Allocation
        self.reciptiff_xr = None # datatype: xarray DataArray, Reciprocal Space Corrected TIFF (xarray)

        # - Caked Image Corrections Data Allocation
        self.cakedtiff_xr = None # datatype: xarray DataArray, Caked Space Corrected TIFF (xarray)
        self.cakedtiff_sinchi_xr = None # datatype: xarray DataArray, Caked Space w/ Sin Chi Correction TIFF (xarray)

        # -- Initialize GIXSTransform() object -- #3
        self.GIXSTransformObj = self.detCorrObj() # Create Detector Object
        self.apply_image_corrections() # Apply Image Corrections
        
        # - Image Smoothing & Normalization
        self.smoothed_img = None # Store Smoothed Image
        self.normalized_img = None # Store Normalized Image
        self.snrtemp = None # Temporary signal-to-noise ratio 

        # - General Data Operations
        self.ds = None # temporary dataarray

        # - 1D Integrations
        self.integrate1d_da = None
        self.integrate1d_method = 'bbox'
        self.npt = 1024
        self.correctSolidAngle=True
        self.polarization_factor = None
        
## --- DATA LOADING & METADATA EXTRACTION --- ##
    # -- Image Loading
    def loadSingleImage(self, tiffPath: Union[str, pathlib.Path, np.ndarray]):
        """
        Loads a single xarray DataArray from a filepath to a raw TIFF
        """

        # - Check that the path exists before continuing.
        if not pathlib.Path(tiffPath).is_file():
            raise ValueError(f"File {tiffPath} does not exist.")

        # - Open the image from the filepath
        image = Image.open(tiffPath)

        # - Create a numpy array from the image
        self.rawtiff_numpy = np.array(image)

        # Run the loadMetaData method to construct the attribute dictionary for the tiffPath.
        self.attribute_dict = self.loadMetaData(tiffPath)

        # - Convert the image numpy array into an xarray DataArray object.
        self.rawtiff_xr = xr.DataArray(data=self.rawtiff_numpy,
                                             dims=['pix_y', 'pix_x'],
                                             attrs=self.attribute_dict)
        
        # - Map the pixel dimensions to the xarray.
        self.rawtiff_xr = self.rawtiff_xr.assign_coords({
            'pix_x': self.rawtiff_xr.pix_x.data,
            'pix_y': self.rawtiff_xr.pix_y.data
        })

    def loadMultiImage(self):
        pass

    # -- Metadata Loading
    def loadMetaData(self, tiffPath, delim='_'):
        """
        Description: Uses metadata_keylist to generate attribute dictionary of metadata based on filename.
        Handle Variables
            tiffPath : string
                Filepath passed to the loadMetaData method that is used to extract metadata relevant to the TIFF image.
            delim : string
                String used as a delimiter in the filename. Defaults to an underscore '_' if no other delimiter is passed.
        
        Method Variables
            attribute_dict : dictionary
                Attributes ictionary of metadata attributes created using the filename and metadata list passed during initialization.
            metadata_list : list
                Metadata list - list of metadata keys used to segment the filename into a dictionary corresponding to said keys.
        """

        self.attribute_dict = {} # Initialize the dictionary.
        filename = pathlib.Path(tiffPath).stem # strip the filename from the tiffPath
        metadata_list = filename.split(delim) # splits the filename based on the delimter passed to the loadMetaData method.

        # - Error handling in case the input list of metadata attributes does not match the length of the recovered metadata from the delimiter.
        if len(metadata_list) != len(self.metadata_keylist):
            raise ValueError("Filename metadata items do not match with metadata keylist.")
        
        for i, metadata_item in enumerate(self.metadata_keylist):
            self.attribute_dict[metadata_item] = metadata_list[i]
        return self.attribute_dict

## --- PROJECT EXPORTING & IMPORTING --- ##
    # # -- Imports the current class instantiation when called.
    def load_xarray_dataset(self, file_path: Path) -> xr.Dataset:
        if not file_path.exists():
            raise FileNotFoundError(f"No file found at {file_path}")
        
        if file_path.suffix != '.nc':
            raise ValueError(f"Invalid file type {file_path.suffix}. Expected a .nc file.")
        
        self.ds = xr.open_dataset(file_path, engine='h5netcdf')

        print("Dataset info:")
        print(self.ds.info())
        return self.ds

## --- IMAGE PROCESSING (REQUIRED) --- ##
    # -- Apply Image Corrections
    def detCorrObj(self):
        """
        Creates a detector corrections object from the GIXSTransform class.
        Utilizes the poniPath and maskPath attributes.
        """

        # Instantiate the GIXSTransform object using necessary parameters
        GIXSTransformObj = WAXSTransform(poniPath = self.poniPath, 
                                         maskPath = self.maskPath,
                                         energy = self.energy) # Additional parameters if needed, such as pixel splitting method or corrections (solid angle)
        return GIXSTransformObj

    def run_pg_integrate1D(self, 
                            npt: int = None, 
                            method: str = None, 
                            correctSolidAngle: bool = None, 
                            polarization_factor: float = None):
        """
        Run the pg_integrate1D method from the WAXSTransform class.

        Parameters:
        - npt (int): Number of points in output data.
        - method : str
            Integration method. Can be "np", "cython", "bbox", "splitpix", "lut" or "lut_ocl" (if you want to go on GPU)
        - correctSolidAngle (bool): Whether to correct for the solid angle of each pixel.
        - polarization_factor (float): Polarization factor for the integration.
        """
        
        # Update attributes if provided
        if npt is not None:
            self.npt = npt
        if method is not None:
            self.integrate1d_method = method
        if correctSolidAngle is not None:
            self.correctSolidAngle = correctSolidAngle
        if polarization_factor is not None:
            self.polarization_factor = polarization_factor
        
        # Run the pg_integrate1D method from WAXSTransform object
        self.integrate1d_da = self.GIXSTransformObj.pg_integrate1D(self.rawtiff_xr,
                                                                npt=self.npt,
                                                                method=self.integrate1d_method,
                                                                correctSolidAngle=self.correctSolidAngle,
                                                                polarization_factor=self.polarization_factor)
        # Store the method parameters as attributes in the DataArray
        self.integrate1d_da.attrs['npt'] = self.npt
        self.integrate1d_da.attrs['method'] = self.integrate1d_method
        self.integrate1d_da.attrs['correctSolidAngle'] = self.correctSolidAngle
        self.integrate1d_da.attrs['polarization_factor'] = self.polarization_factor

    def plot_and_save_qr_intensity(self, output_path: Union[str, pathlib.Path] = None, save_xy: bool = False, save_png: bool = False):
        if output_path is None:
            output_path = pathlib.Path.cwd()
        else:
            output_path = pathlib.Path(output_path)

        # Generate dynamic filename
        timestamp = datetime.now().strftime('%y%m%d_%H%M%S_')
        metadata_values = '_'.join([str(self.attribute_dict[key]) for key in self.metadata_keylist[:3]])
        dynamic_filename = f"{timestamp}{metadata_values}_npt{self.npt}_method{self.integrate1d_method}"

        # Plot qr vs. intensity
        plt.plot(self.integrate1d_da['qr'], self.integrate1d_da)
        plt.xlabel('qr (1/Å)')
        plt.ylabel('Intensity (a.u.)')
        plt.title('1D Integration')

        if save_png:
            plt.savefig(output_path / f"{dynamic_filename}.png")
            
        plt.show()

        if save_xy:
            # Prepare header and save as .xy file
            header = f"# Metadata: npt={self.npt}, method={self.integrate1d_method}, correctSolidAngle={self.correctSolidAngle}, polarization_factor={self.polarization_factor}"
            data = np.column_stack((self.integrate1d_da['qr'], self.integrate1d_da))
            np.savetxt(output_path / f"{dynamic_filename}.xy", data, header=header)

        print(f"Files saved to: {output_path}")

    # -- Generate the caked and reciprocal space corrected datasets.
    def apply_image_corrections(self):
        """
        Utilizes the GIXSTransform object to create image corrections.
        Updates the reciptiff_xr and cakedtiff_xr attributes with the corrected xarray DataArrays.
        """
        
        # Call the pg_convert method using the rawtiff_xr xarray
        self.reciptiff_xr, self.cakedtiff_xr = self.GIXSTransformObj.pg_convert(self.rawtiff_xr)
        self.convert_to_numpy()
        
        # Update the coordinate system for the images
        self.coords = {
            'x_min': self.reciptiff_xr[self.inplane_config].min(),
            'x_max': self.reciptiff_xr[self.inplane_config].max(),
            'y_min': self.reciptiff_xr['q_z'].min(),
            'y_max': self.reciptiff_xr['q_z'].max()
            }

        # Calculate the Signal-to-Noise Ratio for each xarray in the class
        self.calculate_SNR_for_class()

    # -- Conversion to numpy to store in the object instance in case we need these.
    def convert_to_numpy(self):
        recip_da = self.reciptiff_xr
        caked_da = self.cakedtiff_xr
        
        self.recip_data_np = recip_da.data
        self.caked_data_np = caked_da.data
        self.qz_np = recip_da['q_z'].data
        self.qxy_np = recip_da[self.inplane_config].data
        self.chi_np = caked_da['chi'].data
        self.qr_np = caked_da['qr'].data

## --- IMAGE PROCESSING (OPTIONAL) --- ##
    # -- sin(chi) correction applied to a caked image
    def sinchi_corr(self, chicorr=True, qsqr=False):

        if not chicorr and qsqr:
            raise ValueError('chicorr must be enabled for qsqr correction to be applied. This will be updated in the future.')
        
        # Initialize original attributes
        original_attrs = self.cakedtiff_xr.attrs.copy()

        # Apply sin(chi) correction if chicorr is True
        if chicorr:
            sinchi_mask = np.sin(np.radians(np.abs(self.cakedtiff_xr.chi)))
            self.cakedtiff_sinchi_xr = self.cakedtiff_xr * sinchi_mask
            self.cakedtiff_sinchi_xr.attrs.update(original_attrs)
            self.cakedtiff_sinchi_xr.attrs['sinchi'] = True
            self.cakedtiff_sinchi_xr.attrs['qsqr'] = False

            # Apply qr^2 scaling if qsqr is True
            if qsqr:
                self.cakedtiff_sinchi_xr *= self.cakedtiff_sinchi_xr.qr  # Assuming 'qr' is a data variable
                self.cakedtiff_sinchi_xr.attrs['qsqr'] = True

        return self.cakedtiff_sinchi_xr

    # -- Normalize the image.
    def normalize_image(self, img=None, normalizerecip=False):
        # Check for invalid or incompatible types
        if img is None:
            if self.reciptiff_xr is None:
                raise ValueError("Reciprocal space image data is not available.")
            img = self.reciptiff_xr

        if not isinstance(img, (np.ndarray, xr.DataArray)):
            raise ValueError("The input image is not of a compatible type.")

        # Initialize original attributes
        original_attrs = {}

        # Handle xarray DataArray
        if isinstance(img, xr.DataArray):
            img_values = img.values
            original_attrs = img.attrs.copy()
            data_type = 'DataArray'
        else:
            img_values = img
            data_type = 'numpy'

        # Perform normalization
        max_val = np.max(img_values)
        if max_val <= 0:
            raise ValueError("Image maximum intensity is zero or negative, cannot normalize.")

        normalized_img_values = img_values / max_val

        # Find the coordinates of the maximum intensity pixel
        max_coords = np.unravel_index(np.argmax(img_values), img_values.shape)
        if isinstance(img, xr.DataArray):
            max_y = img.coords[img.dims[0]][max_coords[0]].values
            max_x = img.coords[img.dims[1]][max_coords[1]].values
        else:
            max_x, max_y = max_coords

        # Create xarray DataArray and set attributes
        normalized_img = xr.DataArray(normalized_img_values, coords=img.coords if isinstance(img, xr.DataArray) else None, dims=img.dims if isinstance(img, xr.DataArray) else None)
        normalized_img.attrs.update(original_attrs)
        normalized_img.attrs['original_name'] = inspect.currentframe().f_back.f_locals.get('img', 'unknown')
        normalized_img.attrs['original_type'] = data_type

        # Save to class attribute
        self.normalized_img = normalized_img

        if normalizerecip:
            self.reciptiff_xr.values = normalized_img_values
            self.reciptiff_xr.attrs['normalized'] = True

        return normalized_img, (max_x, max_y)

    # -- Image Smoothing Algorithm
    def smooth_image(self, img, method: str = 'gaussian', **kwargs) -> xr.DataArray:
        """
        Smooth the input image using the specified method and exclude zero-intensity regions.
        
        Parameters:
            img (xr.DataArray or np.ndarray): Input image to be smoothed.
            method (str): The smoothing method to use ('gaussian', 'bilateral', 'total_variation', 'anisotropic').
            **kwargs: Additional parameters for the smoothing method.
            
        Returns:
            xr.DataArray: Smoothed image with the same shape as the input.
        """
        
        # Convert to xarray if input is a numpy array
        if isinstance(img, np.ndarray):
            img = xr.DataArray(img)
            
        # Create a mask to exclude zero-intensity regions
        mask = img != 0
        
        # Backup original dtype
        original_dtype = img.dtype
        
        if method == 'gaussian':
            sigma = kwargs.get('sigma', 1)
            smoothed = gaussian_filter(img.where(mask), sigma)
        elif method == 'bilateral':
            sigma_color = kwargs.get('sigma_color', 0.05)
            sigma_spatial = kwargs.get('sigma_spatial', 15)
            smoothed = denoise_bilateral(img.where(mask).values, sigma_color=sigma_color, sigma_spatial=sigma_spatial, multichannel=False)
        elif method == 'total_variation':
            weight = kwargs.get('weight', 0.1)
            smoothed = denoise_tv_chambolle(img.where(mask).values, weight=weight)
        elif method == 'anisotropic':
            smoothed = img.where(mask).copy()
        else:
            raise ValueError("Invalid method. Choose from 'gaussian', 'bilateral', 'total_variation', 'anisotropic'.")
        
        # Reapply the original mask to set zero-intensity regions back to zero
        smoothed = xr.DataArray(smoothed, coords=img.coords, dims=img.dims).where(mask)
        
        # Update the smoothed_img attribute
        self.smoothed_img = smoothed.astype(original_dtype)
        
        return self.smoothed_img

    # -- Image Folding Algorithm Modifying the fold_image method to keep the data from the longer quadrant and append it to the folded image.
    def fold_image(self, data_array, fold_axis):
        """
        Method to fold image along a specified axis.
        
        Parameters:
        - data_array (xarray DataArray): The DataArray to fold
        - fold_axis (str): The axis along which to fold the image
        
        Returns:
        - xarray DataArray: The folded image
        """
        # Filter data for fold_axis >= 0 and fold_axis <= 0
        positive_data = data_array.where(data_array[fold_axis] >= 0, drop=True)
        negative_data = data_array.where(data_array[fold_axis] <= 0, drop=True)
        
        # Reverse negative_data for easier comparison
        negative_data = negative_data.reindex({fold_axis: negative_data[fold_axis][::-1]})
        
        # Find the maximum coordinate of the shorter quadrant (positive_data)
        max_positive_coord = float(positive_data[fold_axis].max())
        
        # Find the equivalent coordinate in the negative_data
        abs_diff = np.abs(negative_data[fold_axis].values + max_positive_coord)
        
        # Minimize the difference
        min_diff_idx = np.argmin(abs_diff)
        
        # Check if the lengths are equivalent
        len_pos = len(positive_data[fold_axis])
        len_neg = len(negative_data[fold_axis][:min_diff_idx+1])
        
        if len_pos != len_neg:
            # Adjust the coordinate range for negative_data
            for i in range(1, 4):  # Check 3 neighbors
                new_idx = min_diff_idx + i
                len_neg = len(negative_data[fold_axis][:new_idx+1])
                if len_pos == len_neg:
                    min_diff_idx = new_idx
                    break
                    
        # Crop the negative_data to match positive_data length
        negative_data_cropped = negative_data.isel({fold_axis: slice(0, min_diff_idx+1)})
        
        # Prepare the new data array
        new_data = xr.zeros_like(positive_data)
        
        # Fold the image
        for i in range(len(positive_data[fold_axis])):
            pos_val = positive_data.isel({fold_axis: i}).values
            neg_val = negative_data_cropped.isel({fold_axis: i}).values
            
            # Pixel comparison and averaging or summing
            new_data.values[i] = np.where(
                (pos_val > 0) & (neg_val > 0), (pos_val + neg_val) / 2,
                np.where((pos_val == 0) & (neg_val > 0), neg_val, pos_val)
            )
            
        # Append residual data from the longer quadrant if exists
        if len(negative_data[fold_axis]) > min_diff_idx+1:
            residual_data = negative_data.isel({fold_axis: slice(min_diff_idx+1, None)})
            residual_data[fold_axis] = np.abs(residual_data[fold_axis])
            new_data = xr.concat([new_data, residual_data], dim=fold_axis)
            
        # Update data_array with the folded image
        data_array = new_data.sortby(fold_axis)
        
        return data_array

## --- CALCULATE IMAGE STATISICS --- ##
    # -- Calculating Signal-to-Noise Ratio (Internal)
    def calculate_SNR_for_class(self):
        """
        Calculate the Signal-to-Noise Ratio (SNR) for all xarray DataArrays stored as class attributes.
        Saves the calculated SNR as an attribute of each xarray DataArray.
        """
        for attr_name in ['rawtiff_xr', 'reciptiff_xr', 'cakedtiff_xr']:
            xarray_obj = getattr(self, attr_name, None)
            if xarray_obj is not None:
                mask = xarray_obj != 0
                mean_val = np.mean(xarray_obj.where(mask).values)
                std_val = np.std(xarray_obj.where(mask).values)
                xarray_obj.attrs['snr'] = mean_val / std_val

    # -- Calculating Signal-to-Noise Ratio (External)
    def calculate_SNR(self, xarray_obj: xr.DataArray) -> float:
        """
        Calculate the Signal-to-Noise Ratio (SNR) for an input xarray DataArray.
        
        Parameters:
            xarray_obj (xr.DataArray): Input xarray DataArray for SNR calculation.
            
        Returns:
            float: Calculated SNR value.
        """
        if not isinstance(xarray_obj, xr.DataArray):
            raise ValueError("Input must be an xarray DataArray.")
            
        mask = xarray_obj != 0
        mean_val = np.mean(xarray_obj.where(mask).values)
        std_val = np.std(xarray_obj.where(mask).values)
        snr = mean_val / std_val
        xarray_obj.attrs['snr'] = snr
        self.snrtemp = snr
        return xarray_obj
    
## --- 2D IMAGE PLOTTING --- ##
    # -- Display the RAW TIFF using XArray
    def rawdisplay_xr(self):
        plt.close('all')
        self.rawtiff_xr.plot.imshow(interpolation='antialiased', cmap='turbo',
                                    vmin=np.nanpercentile(self.rawtiff_xr, 10),
                                    vmax=np.nanpercentile(self.rawtiff_xr, 99))
        plt.title('Raw TIFF Image')
        plt.show()

    # -- Display the Reciprocal Space Map using XArray
    def recipdisplay_xr(self):
        plt.close('all')
        self.reciptiff_xr.plot.imshow(interpolation='antialiased', cmap='turbo',
                                    vmin=np.nanpercentile(self.reciptiff_xr, 10),
                                    vmax=np.nanpercentile(self.reciptiff_xr, 99))
        plt.title('Missing Wedge Correction')
        plt.show()

    # -- Display the Caked Image using XArray
    def cakeddisplay_xr(self):
        plt.close('all')
        self.cakedtiff_xr.plot.imshow(interpolation='antialiased', cmap='turbo',
                                    vmin=np.nanpercentile(self.cakedtiff_xr, 10),
                                    vmax=np.nanpercentile(self.cakedtiff_xr, 99))
        plt.title('Caked Image')
        plt.show()

    # -- Display Image (General)
    def display_image(self, img, title='Image', cmap='turbo'):
        plt.close('all')

        # Check for invalid or incompatible types
        if img is None or not isinstance(img, (np.ndarray, xr.DataArray)):
            raise ValueError("The input image is None or not of a compatible type.")

        # Initialize extent
        extent = None

        # Check for xarray DataArray
        if isinstance(img, xr.DataArray):
            img_values = img.values

            # Extract extent and axis labels from xarray coordinates if available
            coords_names = list(img.coords.keys())
            if len(coords_names) == 2:
                extent = [
                    img.coords[coords_names[1]].min(),
                    img.coords[coords_names[1]].max(),
                    img.coords[coords_names[0]].min(),
                    img.coords[coords_names[0]].max()
                ]
                ylabel, xlabel = coords_names
        else:
            img_values = img

            # Use self.coords if available
            if self.coords is not None:
                extent = [
                    self.coords['x_min'],
                    self.coords['x_max'],
                    self.coords['y_min'],
                    self.coords['y_max']
                ]
                xlabel, ylabel = 'qxy', 'qz'

        # Check for empty or all NaN array
        if np.all(np.isnan(img_values)) or img_values.size == 0:
            raise ValueError("The input image is empty or contains only NaN values.")

        vmin = np.nanpercentile(img_values, 10)
        vmax = np.nanpercentile(img_values, 99)

        plt.imshow(np.flipud(img_values),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                aspect='auto')  # Ensure the aspect ratio is set automatically

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar()
        plt.show()
  
class Integration1D(WAXSReduce):
    def __init__(self, waxs_instance=None):
        if waxs_instance:
            self.__dict__.update(waxs_instance.__dict__)
        else:
            super().__init__()

        # -- 1D Image Processing Variables -- ##
        # - Caked Image Azimuthal Integration
        self.cakeslice1D_xr = None
        self.chislice = []
        self.qrslice = []
        self.cakeslicesum = 'chi'

        # - Boxcut Integration (qxy x qz)
        self.boxcut1D_xr = None
        self.boxcut1D_interp_xr = None
        self.qxyslice = []
        self.qzslice = []
        self.boxcutsum = 'qz'

        # - Pole Figure Integration
        self.pole_chislice = []
        self.pole_qrslice = []
        self.poleleveler = 'slinear'

        # - Reciprocal Space Map Azimuthal Integration
        self.azimuth1D_xr = None
        self.azi_chislice = []
        self.azi_qrslice = []
        self.azimuth1D_xr_sum = None
        self.azimuth_mask = None
        self.original_total_intensity = None

    def cakeslice1D(self, img, chislice = [-90, 90], qrslice = [0, 4], cakeslicesum = 'chi'):
            """
            Description: 
            Takes an input xarray DataArray of a caked image. Performs a sum along either the 'chi' or 'qr' 
            direction after slicing the DataArray based on the input range for 'chi' and 'qr'.
            
            Variables:
            - img: The input Xarray DataArray or DataSet. Should have 'chi' and 'qr' as coordinates.
            - chislice: A list [chimin, chimax] defining the range of 'chi' values for slicing.
            - qrslice: A list [qrmin, qrmax] defining the range of 'qr' values for slicing.
            - cakeslicesum: A string specifying the direction along which to sum ('chi' or 'qr').
            
            Attributes:
            - self.cakeslicesum: Stores the direction along which the sum was performed.
            - self.cakeslice1D: Stores the resulting 1D DataArray after slicing and summing.
            
            Output: 
            Returns the 1D DataArray after slicing and summing, and stores it in self.cakeslice1D.
            """
            
            # Check that cakeslicesum is either 'chi' or 'qr'
            if cakeslicesum not in ['chi', 'qr']:
                print("Invalid cakeslicesum value. Defaulting to 'chi'.")
                cakeslicesum = 'chi'
                
            # Check if the necessary dimensions are present in the DataArray
            chi_aliases = ['chi', 'Chi', 'CHI']
            qr_aliases = ['qr', 'q_r', 'QR', 'Q_R', 'Qr', 'Q_r']
            
            chi_dim = None
            qr_dim = None
            
            for dim in img.dims:
                if dim.lower() in [alias.lower() for alias in chi_aliases]:
                    chi_dim = dim
                if dim.lower() in [alias.lower() for alias in qr_aliases]:
                    qr_dim = dim
                    
            if chi_dim is None or qr_dim is None:
                raise ValueError("The input DataArray must contain dimensions corresponding to 'chi' and 'qr'.")

            self.cakeslicesum = cakeslicesum

            # Extract the slice boundaries
            qrmin, qrmax = qrslice
            chimin, chimax = chislice

            # Perform the slicing and summing
            cakeslice1D_xr = img.sel({qr_dim: slice(qrmin, qrmax), chi_dim: slice(chimin, chimax)}).sum(cakeslicesum)

            # Assign DataArray attributes
            cakeslice1D_xr.attrs['qrmin'] = qrmin
            cakeslice1D_xr.attrs['qrmax'] = qrmax
            cakeslice1D_xr.attrs['chimin'] = chimin
            cakeslice1D_xr.attrs['chimax'] = chimax
            cakeslice1D_xr.attrs['sumdir'] = cakeslicesum

            # Assign coordinate names
            coord_name = qr_dim if cakeslicesum == chi_dim else chi_dim
            cakeslice1D_xr = cakeslice1D_xr.rename({coord_name: 'intensity'})

            # Save as a class attribute
            self.cakeslice1D_xr = cakeslice1D_xr

            return self.cakeslice1D_xr

    def boxcut1D(self, img, qxyslice=[0, 2.5], qzslice=[0, 2.5], boxcutsum = None, interpolate_gaps=False, interp_method = 'slinear', order = None):
        """
        Description: 
        Receives an input Xarray DataArray and performs a sum along either the 'qxy' or 'qz' direction
        after slicing the DataArray based on the input range for 'qxy' and 'qz'. The method is flexible 
        to different names for the 'qxy' dimension.

        Variables:
        - img: Input Xarray DataArray. Should have 'qz' and optionally 'qxy', 'q_xy', 'q_para', 'q_perp' as coordinates.
        - qxyslice: A list [qxymin, qxymax] defining the range of 'qxy' values for slicing.
        - qzslice: A list [qzmin, qzmax] defining the range of 'qz' values for slicing.
        - boxcutsum: A string specifying the direction along which to sum ('qxy' or 'qz').
        - interpolate_gaps: Whether to interpolate over gaps in the data (True or False).
        - display_interpolation: Whether to display pre- and post-interpolated plots (True or False).

        Attributes:
        - self.boxcut1D_xr: Stores the resulting 1D DataArray after slicing and summing.

        Output: 
        Returns the 1D DataArray after slicing and summing, and stores it in self.boxcut1D_xr.
        """
        # Check if boxcutsum is either 'qxy' or 'qz'
        # if boxcutsum not in ['qxy', 'qz']:
        #     print("Invalid boxcutsum value. Defaulting to 'qz'.")
        #     boxcutsum = 'qz'

        # Normalize dimension names
        qz_aliases = ['qz', 'q_z', 'QZ', 'Q_z', 'Q_Z']
        qxy_aliases = ['qxy', 'q_xy', 'QXY', 'Qxy', 'Q_xy', 'Q_XY']

        qz_dim = None
        qxy_dim = None

        for dim in img.dims:
            if dim in qz_aliases:
                qz_dim = dim
            if dim in qxy_aliases:
                qxy_dim = dim

        if qz_dim is None:
            raise ValueError("The input image must have 'qz' or an alias as a dimension.")
        if qxy_dim is None:
            raise ValueError("The input image must have 'qxy' or an alias as a dimension.")

        # Align boxcutsum with the actual dimension name
        if boxcutsum in qz_aliases:
            boxcutsum = qz_dim
        elif boxcutsum in qxy_aliases:
            boxcutsum = qxy_dim

        # Extract slice boundaries
        qxymin, qxymax = qxyslice
        qzmin, qzmax = qzslice

        # Prepare slicing dictionary
        slicing_dict = {qz_dim: slice(qzmin, qzmax), qxy_dim: slice(qxymin, qxymax)}

        # Perform slicing and summing
        # boxcut1D_xr = img.sel(**slicing_dict).sum(boxcutsum)

        # Perform slicing
        sliced_img = img.sel(**slicing_dict)

        # Perform summing
        boxcut1D_xr = sliced_img.sum(boxcutsum)
        
        # Assign DataArray attributes
        boxcut1D_xr.attrs['qxymin'] = qxymin
        boxcut1D_xr.attrs['qxymax'] = qxymax
        boxcut1D_xr.attrs['qzmin'] = qzmin
        boxcut1D_xr.attrs['qzmax'] = qzmax
        boxcut1D_xr.attrs['sumdir'] = boxcutsum

        # Identify coordinate name for later use
        coord_name = qxy_dim if boxcutsum == qz_dim else qz_dim
        boxcut1D_xr.attrs['xcoord'] = coord_name

        # Check if the dimension still exists after summing
        if coord_name in boxcut1D_xr.dims:
            boxcut1D_xr = boxcut1D_xr.rename({coord_name: 'intensity'})

        # Save as a class attribute
        self.boxcut1D_xr = boxcut1D_xr
        
        # Handle interpolation on the sliced image if necessary
        if interpolate_gaps:
            # Replace zero values with NaN for interpolation
            # sliced_img_interpolated = sliced_img.where(sliced_img != 0)
            
            if interp_method == 'polynomial':
                # Interpolation along the specific dimension
                if order is None or order is np.NaN or order is 0:
                    order = 3

                sliced_img_interpolated = sliced_img.where(sliced_img != 0).compute().interpolate_na(dim=coord_name, method=interp_method, order = order)
            
            else: 
                sliced_img_interpolated = sliced_img.where(sliced_img != 0).compute().interpolate_na(dim=coord_name, method=interp_method)

            # Save as a class attribute
            self.sliced_img_interpolated_xr = sliced_img_interpolated

        # If interpolated slice exists, sum it as well
        if hasattr(self, 'sliced_img_interpolated_xr'):
            boxcut1D_interp_xr = self.sliced_img_interpolated_xr.sum(boxcutsum)
            
            # Rename the dimension to 'intensity' for the interpolated array
            if coord_name in boxcut1D_interp_xr.dims:
                boxcut1D_interp_xr = boxcut1D_interp_xr.rename({coord_name: 'intensity'})
            
            self.boxcut1D_interp_xr = boxcut1D_interp_xr

        return self.boxcut1D_xr

    def polefig1D(self, img, pole_chislice, pole_qrslice, qrcenter=None, chicenter=0, poleleveler='linear'):
        """
        Description:
            Generates a 1D pole figure to analyze the distribution of X-ray scattering intensity over a specified peak center in reciprocal space.
            The method slices the input 2D data array based on the given ranges for 'qr' and 'chi' dimensions. Gaps in the 'qr' data are interpolated
            using spline interpolation of order 1 ('slinear'). The method then sums or averages the intensity in the specified dimension ('qr' by default)
            for each value in the opposing dimension ('chi'). Optionally, a linear background subtraction can be applied to the summed or averaged data.

        Variables:
            img (xr.DataArray): Input 2D Xarray DataArray containing the scattering intensity.
            pole_chislice (tuple): Specifies the slice range for the 'chi' dimension, e.g., (-90, 90).
            pole_qrslice (tuple): Specifies the slice range for the 'qr' dimension, e.g., (0.2, 3).
            qrcenter (float, optional): Center value for 'qr' in the pole figure. Defaults to the midpoint of pole_qrslice.
            chicenter (float, optional): Center value for 'chi' in the pole figure. Defaults to 0.
            poleleveler (str, optional): Specifies the type of background leveling. Options are 'linear', 'average', or None. Defaults to 'linear'.

        Attributes:
            self.pole_chislice: Stores the 'chi' slice range.
            self.pole_qrslice: Stores the 'qr' slice range.
            self.poleleveler: Stores the type of background leveling applied.

        Output:
            pole_fig_da (xr.DataArray): 1D Xarray DataArray representing the pole figure. The DataArray includes attributes that store
                                        the input slice ranges and the type of background leveling applied.
            
        Note:
            Gaps in 'qr' are interpolated. Gaps in 'chi' are ignored.
            The linear background fit is constrained to be below the actual intensity values.
        """
        
        # Check if the necessary dimensions are present in the DataArray
        chi_aliases = ['chi', 'Chi', 'CHI']
        qr_aliases = ['qr', 'q_r', 'QR', 'Q_R', 'Qr', 'Q_r']
        
        chi_dim = None
        qr_dim = None
        
        for dim in img.dims:
            if dim.lower() in [alias.lower() for alias in chi_aliases]:
                chi_dim = dim
            if dim.lower() in [alias.lower() for alias in qr_aliases]:
                qr_dim = dim
                
        if chi_dim is None or qr_dim is None:
            raise ValueError("The input DataArray must contain dimensions corresponding to 'chi' and 'qr'.")
        
        # Initialize instance variables
        self.pole_chislice = pole_chislice
        self.pole_qrslice = pole_qrslice
        self.poleleveler = poleleveler

        # Slice the DataArray
        sliced_img = img.sel(qr=slice(*pole_qrslice), chi=slice(*pole_chislice))

        # Interpolate gaps along 'qr' dimension using 'slinear' (spline interpolation of order 1)
        interp_img = sliced_img.interpolate_na(dim='qr', method='slinear')

        # Perform background leveling
        if poleleveler == 'linear':
            # Fit a linear model to the background and subtract it from the data
            chi_values = interp_img['chi'].values
            intensity_values = interp_img.sum(dim='qr').values  # summing over qr for each chi

            coeffs = np.polyfit(chi_values, intensity_values, 1)  # Linear fit
            linear_background = np.polyval(coeffs, chi_values)

            # Ensure linear fit doesn't exceed actual intensity values
            linear_background = np.minimum(linear_background, intensity_values)

            # Subtract background but ensure intensity remains non-negative
            pole_fig = np.maximum(intensity_values - linear_background, 0)

        elif poleleveler == 'average':
            pole_fig = interp_img.mean(dim='qr').values  # average over qr for each chi

        else:
            pole_fig = interp_img.sum(dim='qr').values  # sum over qr for each chi

        # Create output DataArray
        pole_fig_da = xr.DataArray(pole_fig,
                                coords=[('chi', interp_img['chi'].values)],
                                attrs={'pole_chislice': pole_chislice, 'pole_qrslice': pole_qrslice, 'poleleveler': poleleveler})
        
        # Update instance variable
        self.polefig1D_xr = pole_fig_da

        return pole_fig_da  

    @staticmethod
    def convert_to_chi_qr(qxy, qz):
        qxy_mesh, qz_mesh = np.meshgrid(qxy, qz)
        chi = np.arctan2(qz_mesh, qxy_mesh) * (180 / np.pi)
        qr = np.sqrt(qxy_mesh ** 2 + qz_mesh ** 2)
        return chi, qr

    @staticmethod
    def generate_mock_data_with_variable_intensity(size=200, num_arcs=5, intensity_base=100, intensity_variation=20):
        x, y = np.linspace(-size // 2, size // 2, size), np.linspace(-size // 2, size // 2, size)
        xx, yy = np.meshgrid(x, y)
        rr = np.sqrt(xx ** 2 + yy ** 2)
        angle = np.arctan2(yy, xx)

        img = np.zeros((size, size))
        for i in range(1, num_arcs + 1):
            arc_mask = np.abs(rr - i * 10) < 1
            img[arc_mask] = intensity_base + intensity_variation * np.sin(3 * angle[arc_mask])

        img[:, :size // 2] = 0
        return img

    @staticmethod
    def interpolate_along_chi(chi, qr, img):
        chi_values = np.unique(chi)
        interpolated_img = np.zeros_like(img)

        for chi_val in chi_values:
            mask = chi == chi_val
            qr_values = qr[mask]
            img_values = img[mask]
            if len(qr_values) < 2:
                continue
            interpolator = interp1d(qr_values, img_values, kind='linear', fill_value='extrapolate')
            interpolated_img[mask] = interpolator(qr[mask])

        return interpolated_img
        
    def azimuth1D(self, img, chi_range, qr_range, sum_direction, discretization=0.1):
        """
        Description:
        Takes a 2D Xarray DataArray and performs azimuthal integration over a specified range of chi and qr values.

        Variables:
        - img: Input Xarray DataArray with 'qxy' and 'qz' as coordinates.
        - chi_range: A tuple (chi_min, chi_max) specifying the range of chi values for integration.
        - qr_range: A tuple (qr_min, qr_max) specifying the range of qr values for integration.
        - sum_direction: A string specifying the direction along which to sum ('chi' or 'qr').
        - discretization: A float specifying the step size for the azimuthal sum.

        Attributes:
        - self.azimuth1D_xr_sum: Stores the resulting 1D DataArray after azimuthal integration.
        - self.azimuth_mask: Stores the mask used for slicing the image based on chi and qr ranges.
        - self.original_total_intensity: Stores the original total intensity of the masked image for verification.

        Output:
        Returns the 1D DataArray after azimuthal integration and stores it in self.azimuth1D_xr_sum.
        """

        qz_aliases = ['qz', 'q_z', 'QZ', 'Q_z', 'Q_Z']
        qxy_aliases = ['qxy', 'q_xy', 'QXY', 'Qxy', 'Q_xy', 'Q_XY']
        
        qz_dim = None
        qxy_dim = None
        for dim in img.dims:
            if dim in qz_aliases:
                qz_dim = dim
            if dim in qxy_aliases:
                qxy_dim = dim
        
        if qz_dim is None or qxy_dim is None:
            raise ValueError("The input DataArray must have 'qxy' or an alias and 'qz' or an alias as dimensions.")
        
        chi_min, chi_max = chi_range
        qr_min, qr_max = qr_range
        
        if sum_direction not in ['chi', 'qr']:
            raise ValueError("Invalid sum_direction. Choose either 'chi' or 'qr'.")
        
        # Step 2: Coordinate Conversion
        qxy = img.coords[qxy_dim].values
        qz = img.coords[qz_dim].values
        
        chi, qr = self.convert_to_chi_qr(qxy, qz)
        
        # Step 3: Masking and Slicing
        mask = (chi >= chi_range[0]) & (chi <= chi_range[1]) & (qr >= qr_range[0]) & (qr <= qr_range[1])
        img_masked = img.where(mask)
        self.azimuth_mask = img_masked
        self.original_total_intensity = img.where(mask).sum().item()
        
        # Step 4: Interpolation Along Chi
        img_interpolated = self.interpolate_along_chi(chi, qr, img_masked.values)
        self.img_interpolated = img_interpolated

        # Step 5: Azimuthal Summation
        self.azimuth1D_xr_sum = xr.DataArray(img_interpolated, 
            coords=[('chi', np.linspace(chi_range[0], chi_range[1], img_interpolated.shape[0])),
                    ('qr', np.linspace(qr_range[0], qr_range[1], img_interpolated.shape[1]))],
            dims=['chi', 'qr'])
        
        return self.azimuth1D_xr_sum
   
    def generate_test_data(self, shape=(100, 100), num_arcs=3, arc_width=3):
        """
        Generate a test 2D DataArray with concentric half-arcs.

        Parameters:
        - shape: tuple, shape of the image
        - num_arcs: int, number of concentric arcs
        - arc_width: int, pixel width of each arc

        Returns:
        - xr.DataArray, the generated test data
        """
        y, x = np.ogrid[-shape[0]//2:shape[0]//2, -shape[1]//2:shape[1]//2]
        mask = np.zeros(shape)
        for i in range(num_arcs):
            radius_inner = i * (arc_width + 5)  # 5 is the spacing between arcs
            radius_outer = radius_inner + arc_width
            mask_arc = (x ** 2 + y ** 2 >= radius_inner ** 2) & (x ** 2 + y ** 2 < radius_outer ** 2)
            mask += mask_arc

        # Only consider the half arcs (y > 0)
        mask = mask * (y >= 0)

        # Create a xarray DataArray
        test_data = xr.DataArray(mask, coords=[('qz', np.linspace(-1, 1, shape[0])), ('qxy', np.linspace(-1, 1, shape[1]))])
        return test_data

    def coord_transform(self, img):
        """
        Perform the coordinate transformation from (qz, qxy) to (chi, qr).
        
        Parameters:
        - img: xr.DataArray, input image in (qz, qxy) coordinates
        
        Returns:
        - chi: xr.DataArray, image in chi coordinate
        - qr: xr.DataArray, image in qr coordinate
        """
        qz = img.coords['qz']
        qxy = img.coords['qxy']
        chi = np.arctan2(qz, qxy) * (180 / np.pi)
        qr = np.sqrt(qxy ** 2 + qz ** 2)
        return chi, qr

    # Updated 'display_image1D' method to include save_path option
    def display_image1D(self, img1D, color='red', title='1D Integrated Image', save_image=False, samplenameprefix=None, savePath=".", plot_interpolated=False):
        """
        Description: Plots a 1D DataArray using matplotlib.

        Variables:
        - integrator: The integrator object containing the 1D Xarray DataArray to be plotted.
        - color: Color of the plot line for the original data.
        - title: Title of the plot.
        - save_image: Whether to save the image as a .png file (True or False).
        - samplenameprefix: Prefix for the saved image file name if save_image is True.
        - save_path: Path where the image will be saved if save_image is True.
        - plot_interpolated: Whether to plot the interpolated data (True or False).

        Output:
        Displays the plot and optionally saves it as a .png file.
        """
        
        # img1D = integrator.boxcut1D_xr  # Extract the DataArray from the integrator object

        plt.close('all')
        plt.figure(figsize=(10, 6))
        
        # Choose a different color for interpolated data if it matches the original color
        interp_color = 'green' if color != 'green' else 'blue'
        
        # Plot interpolated data if requested
        if plot_interpolated:
            try:
                # integrator.boxcut1D_interp_xr.plot.line(color=interp_color, label='Interpolated')
                img1D.boxcut1D_interp_xr.plot.line(color=interp_color, label='Interpolated')
            except AttributeError:
                print("Interpolated data is not available. Only original data will be plotted.")
        
        # Plot original data
        img1D.plot.line(color=color, label='Original')
        
        plt.ylabel('Intensity (arb. units)')
        # plt.xlabel(img1D.attrs.get('sumdir', 'Coordinate'))
        plt.xlabel(img1D.attrs.get('xcoord', 'Coordinate'))
        plt.title(title)
        plt.grid(True)
        plt.legend()  # Add legend
            
        if save_image:
            if samplenameprefix is None:
                raise ValueError("samplenameprefix must be provided if save_image is True")
            
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            filename = f"{samplenameprefix}_{timestamp}.png"
            full_save_path = Path(savePath) / filename
            plt.savefig(full_save_path)
            
        plt.show()
        
    def plot_data_transform_interpolation(original_data, chi, qr, transformed_data, interpolated_data, chi_range, qr_range):
        fig, ax = plt.subplots(1, 3, figsize=(20, 6))
        
        # Original data in qxy vs qz space
        ax[0].imshow(original_data, cmap='viridis', origin='lower')
        ax[0].set_title('Original Data (qxy vs qz)')
        ax[0].set_xlabel('qxy')
        ax[0].set_ylabel('qz')
        
        # Transformed data in chi vs qr space
        ax[1].scatter(chi.flatten(), qr.flatten(), c=transformed_data.flatten(), cmap='viridis', s=1)
        ax[1].set_title('Transformed Data (chi vs qr)')
        ax[1].set_xlabel('chi (degrees)')
        ax[1].set_ylabel('qr')
        ax[1].set_xlim([-180, 180])
        
        # Interpolated data in chi vs qr space (2D image)
        ax[2].imshow(interpolated_data, aspect='auto', cmap='viridis', 
                    extent=[chi_range[0], chi_range[1], qr_range[0], qr_range[1]], origin='lower')
        ax[2].set_title('Interpolated Data (chi vs qr)')
        ax[2].set_xlabel('chi (degrees)')
        ax[2].set_ylabel('qr')
        
        plt.tight_layout()
        plt.show()

## -- WAXS TOPAS Class: Generates input files for TOPAS -- ## 
class WAXSTOPAS(WAXSReduce):
    def __init__(self, 
                 projectPath: Union[str, Path], 
                 waxs_instance=None, 
                 *args, **kwargs):
        if waxs_instance:
            self.__dict__.update(waxs_instance.__dict__)
        else:
            super().__init__(*args, **kwargs)
        
        # Calculate the wavelength based on the energy
        self.wavelength = np.round((4.1357e-15 * 2.99792458e8) / (self.energy * 1000), 13)
        
        # Generate the TOPAS project path
        self.generate_topas_projectpath(projectPath)

        # Generate project name
        self.projectname = projectPath.name if isinstance(projectPath, Path) else Path(projectPath).name
        
    def generate_topas_projectpath(self, projectPath: Union[str, Path]):
        """
        Generate the project path where TOPAS related files will be stored.
        
        Parameters:
        - projectPath (Union[str, Path]): The project directory where the 'topas7_project' folder will be created.
        """
        
        # Make sure the projectPath exists
        projectPath = Path(projectPath)
        if not projectPath.exists():
            raise ValueError(f"The provided project path {projectPath} does not exist.")
        
        # Create 'topas7_project' folder if it doesn't exist
        self.topasPath = projectPath / 'topas7_project'
        self.topasPath.mkdir(exist_ok=True)
        
    def generate_xye_file(self, output_filename: Optional[str] = None):
        """
        Generate a .xye file from the 1D integrated DataArray.
        
        Parameters:
        - output_filename (Optional[str]): The name of the output .xye file. If not provided, a default name will be generated.
        """
        
        # Check if the integrate1d_da DataArray is None or contains only NaN values
        if self.integrate1d_da is None or np.isnan(self.integrate1d_da.values).all():
            # If it's None or filled with NaNs, run the 1D integration
            self.run_pg_integrate1D(npt=1024, 
                                    method='bbox', 
                                    correctSolidAngle=True, 
                                    polarization_factor=None)
        
        # Create 'twotheta' dimension based on 'qr'
        
        # twotheta_values = 2 * np.arcsin(self.wavelength * self.integrate1d_da['qr'].values / (4 * np.pi))
        # twotheta_values = 2 * (180 / np.pi) * np.arcsin(self.wavelength * self.integrate1d_da['qr'].values / (4 * np.pi))
                # Inline conversion from meters to Angstroms, and then calculate 'twotheta'
        twotheta_values = 2 * np.arcsin((self.wavelength * 1e10) * self.integrate1d_da['qr'].values / (4 * np.pi)) * (180 / np.pi)  # Convert radians to degrees
        self.integrate1d_da = self.integrate1d_da.assign_coords({'twotheta': ('qr', twotheta_values)})
        
        # Create 'root_int_error' dimension based on intensity
        root_int_error = np.sqrt(self.integrate1d_da.values)
        self.integrate1d_da = self.integrate1d_da.assign_coords({'root_int_error': ('qr', root_int_error)})

        # Define the output file path
        if output_filename is None:
            timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
            output_filename = f"{self.projectname}_{timestamp}_topas.xye"
        
        output_filepath = self.topasPath / output_filename

        # Write the .xye file
        with output_filepath.open('w') as f:
            for x, y, e in zip(self.integrate1d_da['twotheta'].values, self.integrate1d_da.values, self.integrate1d_da['root_int_error'].values):
                f.write(f"{x} {y} {e}\n")
        
        # Generate the complementary .txt file for metadata
        txt_filename = output_filename.replace('.xye', '_metadata.txt')
        self.generate_metadata_txt(txt_filename)

    def generate_metadata_txt(self, output_filename: str):
        """
        Generate a .txt file containing metadata attributes.
        
        Parameters:
        - output_filename (str): The name of the output .txt file.
        """
        
        output_filepath = self.topasPath / output_filename
        
        with output_filepath.open('w') as f:
            f.write("Metadata Information\n")
            f.write("====================\n")
            
            # Extract attributes from the 1D integrated DataArray
            f.write(f"Integration Method: {self.integrate1d_method}\n")
            f.write(f"Number of Points (npt): {self.npt}\n")
            f.write(f"Correct Solid Angle: {self.correctSolidAngle}\n")
            f.write(f"Polarization Factor: {self.polarization_factor}\n")
            
            # Additional attributes from WAXSReduce
            f.write(f"Energy: {self.energy} keV\n")
            f.write(f"Wavelength: {self.wavelength} Å\n")
            f.write(f"PONI Path: {str(self.poniPath)}\n")
            f.write(f"Mask Path: {str(self.maskPath)}\n")
            f.write(f"Incident Angle: {self.incident_angle}\n")
            
            # Metadata dictionary
            f.write("Additional Metadata:\n")
            for key, value in self.attribute_dict.items():
                f.write(f"{key}: {value}\n")

## -- Custom Azimuthal Integration & Pixel Splitting -- ##
class Azimuth1D(Integration1D):
    def __init__(self, parent_instance=None):
        # Check if a parent_instance is provided, otherwise create a new instance
        if parent_instance is not None:
            self.__dict__.update(parent_instance.__dict__)
        else:
            super().__init__()

        # Initialize specific attributes for Azimuth1D
        self.mask = None
        self.cropped_data = None
        self.shape = []
        self.corners = None
        self.gaps = None

    def find_closest_point(self, pairs, target_q_xy, target_q_z, tol1=0.02, tol2=0.01):
        filtered_pairs = [pair for pair in pairs if abs(pair[0] - target_q_xy) <= tol1]
        filtered_pairs = sorted(filtered_pairs, key=lambda x: abs(x[1] - target_q_z))
        closest_by_qz = [pair for pair in filtered_pairs if abs(pair[1] - target_q_z) <= tol1]
        closest_by_qz = sorted(closest_by_qz, key=lambda x: abs(x[0] - target_q_xy))
        closest_by_qxy = [pair for pair in closest_by_qz if abs(pair[0] - target_q_xy) <= tol2]
        closest_point = min(closest_by_qxy, key=lambda x: abs(x[1] - target_q_z))
        return closest_point

    def generate_overlay_shape(self, qr_range, chi_range):
        shape = []
        qr_min, qr_max = qr_range
        chi_min, chi_max = chi_range

        # Generate points along the arc for each qr value
        for qr in [qr_min, qr_max]:
            chi_values = np.linspace(chi_min, chi_max, 100)
            for chi in chi_values:
                chi_rad = np.radians(chi)
                q_xy = qr * np.sin(chi_rad)
                q_z = qr * np.cos(chi_rad)
                shape.append((q_xy, q_z))

        # Generate points along the lines for each chi value
        for chi in [chi_min, chi_max]:
            chi_rad = np.radians(chi)
            for qr in np.linspace(qr_min, qr_max, 100):
                q_xy = qr * np.sin(chi_rad)
                q_z = qr * np.cos(chi_rad)
                shape.append((q_xy, q_z))

        self.shape = shape
        return shape

    def azimuthal_crop(self, data, qr_range, chi_range):
        shape = self.generate_overlay_shape(qr_range, chi_range)
        path = MatplotlibPath(shape)

        # Create an empty mask with the same shape as the input DataArray
        mask = np.zeros(data.shape, dtype=bool)

        qr_values = np.sqrt(data['q_xy'] ** 2 + data['q_z'] ** 2)  # Calculate qr values

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                point = np.array([data['q_xy'].values[j], data['q_z'].values[i]])
                mask[i, j] = path.contains_point(point)
                
        # Calculate chi for each q_xy and q_z coordinate in degrees
        chi_values = np.degrees(np.arctan2(data['q_xy'], data['q_z']))
        
        # Step 4: Fill the inside of the mask
        filled_mask = ndi.binary_fill_holes(mask)
        
        # Step 5: Apply the mask to the data
        self.cropped_data = data.where(filled_mask)
        
        # Calculate qr values for the cropped data and store it
        qr_values_cropped = np.sqrt(self.cropped_data['q_xy'] ** 2 + self.cropped_data['q_z'] ** 2)
        self.cropped_data['qr'] = qr_values_cropped
        
        # Store the calculated chi values in degrees in the cropped data
        self.cropped_data['chi'] = chi_values

        return self.cropped_data

    def azimuthal_integration(self, num_qr_bins, qr_range):
        qr_min, qr_max = qr_range  # Extract the specified qr range
        
        # Initialize an array to store the azimuthal sum
        azimuthal_sum = np.zeros(num_qr_bins)

        # Initialize an array to store the gaps
        gaps = np.zeros(num_qr_bins, dtype=bool)

        # Determine qr bin edges within the specified range
        qr_bins = np.linspace(qr_min, qr_max, num_qr_bins + 1)

        # Loop through each qr bin to perform the azimuthal sum
        for i in range(num_qr_bins):
            qr_low = qr_bins[i]
            qr_high = qr_bins[i + 1]

            # Extract the data in the current qr bin within the specified range
            bin_data = self.cropped_data.where((self.cropped_data['qr'] >= qr_low) & (self.cropped_data['qr'] < qr_high), drop=True)

            # Check for gaps (bins where all values are NaN)
            if np.all(np.isnan(bin_data)):
                gaps[i] = True
                continue

            # Perform the azimuthal sum (ignoring NaNs)
            azimuthal_sum[i] = np.nansum(bin_data)

        # Store the gaps information
        self.gaps = gaps

        return qr_bins[:-1], azimuthal_sum, gaps

    def azimuthal_integration_by_chi(self, num_chi_bins, chi_range):
        chi_min, chi_max = chi_range  # Extract the specified chi range

        # Initialize an array to store the azimuthal sum by chi
        azimuthal_sum_by_chi = np.zeros(num_chi_bins)

        # Initialize an array to store the gaps
        gaps_by_chi = np.zeros(num_chi_bins, dtype=bool)

        # Determine chi bin edges within the specified range
        chi_bins = np.linspace(chi_min, chi_max, num_chi_bins + 1)

        # Loop through each chi bin to perform the azimuthal sum
        for i in range(num_chi_bins):
            chi_low = chi_bins[i]
            chi_high = chi_bins[i + 1]

            # Extract the data in the current chi bin within the specified range
            bin_data = self.cropped_data.where((self.cropped_data['chi'] >= chi_low) & (self.cropped_data['chi'] < chi_high), drop=True)

            # Check for gaps (bins where all values are NaN)
            if np.all(np.isnan(bin_data)):
                gaps_by_chi[i] = True
                continue

            # Perform the azimuthal sum by chi (ignoring NaNs)
            azimuthal_sum_by_chi[i] = np.nansum(bin_data)

        # Store the gaps information
        self.gaps_by_chi = gaps_by_chi

        return chi_bins[:-1], azimuthal_sum_by_chi, gaps_by_chi

    def plot_data(self, original_data, cropped_data, shape):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        extent_original = [original_data['q_xy'].min(), original_data['q_xy'].max(), 
                           original_data['q_z'].min(), original_data['q_z'].max()]
        
        extent_cropped = [cropped_data['q_xy'].min(), cropped_data['q_xy'].max(), 
                          cropped_data['q_z'].min(), cropped_data['q_z'].max()]
        
        vmin1 = np.nanpercentile(original_data, 10)
        vmax1 = np.nanpercentile(original_data, 99)

        ax1 = axes[0]
        im1 = ax1.imshow(original_data, 
                         origin='lower', 
                         extent=extent_original, 
                         vmin=vmin1,
                         vmax=vmax1,
                         cmap='turbo')
        
        shape_x, shape_y = zip(*shape)
        ax1.scatter(shape_x, shape_y, c='red', s=6, label='Boundary Points')
        ax1.set_title('Original Data')
        ax1.set_xlabel('q_xy')
        ax1.set_ylabel('q_z')
        plt.colorbar(im1, ax=ax1)

        # vmin2 = np.nanpercentile(cropped_data, 10)
        # vmax2 = np.nanpercentile(cropped_data, 99)
        
        ax2 = axes[1]
        im2 = ax2.imshow(cropped_data, origin='lower', 
                         extent=extent_cropped, 
                         vmin=vmin1,
                         vmax=vmax1,
                         cmap='turbo')
        
        ax2.set_title('Cropped Data')
        ax2.set_xlabel('q_xy')
        ax2.set_ylabel('q_z')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.show()

    def plot_azimuthal_integration(self, qr_bins, azimuthal_sum, gaps):
        plt.figure(figsize=(10, 6))
        
        # Create a 1D line plot with continuous lines
        plt.plot(qr_bins, azimuthal_sum, color='blue', linestyle='-')
        
        # # Highlight bins with gaps in red
        # for i, gap in enumerate(gaps):
        #     if gap:
        #         plt.axvline(qr_bins[i], color='red', linestyle='--', linewidth=2)

        plt.xlabel('qr')
        plt.ylabel('Summed Intensity')
        plt.title('Azimuthal Integration')
        plt.grid(True)
        plt.show()

    def plot_azimuthal_integration_by_chi(self, chi_bins, azimuthal_sum_by_chi, gaps_by_chi):
        plt.figure(figsize=(10, 6))
        
        # Create a 1D line plot by chi with continuous lines
        plt.plot(chi_bins, azimuthal_sum_by_chi, color='blue', linestyle='-')
        
        # # Highlight bins with gaps in red
        # for i, gap in enumerate(gaps_by_chi):
        #     if gap:
        #         plt.axvline(chi_bins[i], color='red', linestyle='--', linewidth=2)

        plt.xlabel('chi')
        plt.ylabel('Summed Intensity')
        plt.title('Azimuthal Integration by Chi')
        plt.grid(True)
        plt.show()

## --- IMAGE INTERPOLATION  --- ##
class ImageTransform:    
    @staticmethod
    def convert_to_chi_qr(qxy, qz):
        qxy_mesh, qz_mesh = np.meshgrid(qxy, qz)
        chi = np.arctan2(qz_mesh, qxy_mesh) * (180 / np.pi)
        qr = np.sqrt(qxy_mesh ** 2 + qz_mesh ** 2)
        return chi, qr
    
    @staticmethod
    def generate_mock_data(size=100, num_arcs=5):
        x, y = np.linspace(-size // 2, size // 2, size), np.linspace(-size // 2, size // 2, size)
        xx, yy = np.meshgrid(x, y)
        rr = np.sqrt(xx ** 2 + yy ** 2)
        angle = np.arctan2(yy, xx)

        img = np.zeros((size, size))
        for i in range(1, num_arcs + 1):
            arc_mask = np.abs(rr - i * 10) < 1
            img[arc_mask] = 1

        img[:, :size // 2] = 0
        return img
    
    @staticmethod
    def interpolate_along_chi(chi, qr, img):
        unique_qr = np.unique(qr)
        new_chi_values = np.linspace(-90, 90, 100)  # 100 points from -90 to 90 degrees
        interpolated_img = []

        for u_qr in unique_qr:
            mask = (qr == u_qr)
            subset_chi = chi[mask]
            subset_data = img[mask]
            if len(subset_chi) < 2:
                continue
            
            interp_func = interp1d(subset_chi, subset_data, kind='linear', bounds_error=False, fill_value=0)
            interpolated_row = interp_func(new_chi_values)
            interpolated_img.append(interpolated_row)
        
        return np.array(interpolated_img)
    
    @staticmethod
    def plot_data(mock_data, chi, qr, interpolated_img):
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original data
        ax[0].imshow(mock_data, cmap='viridis', origin='lower')
        ax[0].set_title('Original Data')
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')

        # Transformed data
        ax[1].scatter(chi, qr, c=mock_data.flatten(), cmap='viridis', s=1)
        ax[1].set_title('Transformed Data (chi vs qr)')
        ax[1].set_xlabel('chi')
        ax[1].set_ylabel('qr')
        
        # Interpolated data
        ax[2].imshow(interpolated_img, aspect='auto', cmap='viridis', origin='lower')
        ax[2].set_title('Interpolated Data')
        ax[2].set_xlabel('chi')
        ax[2].set_ylabel('qr')
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def extractcoord(img):

        qz_aliases = ['qz', 'q_z', 'QZ', 'Q_z', 'Q_Z']
        qxy_aliases = ['qxy', 'q_xy', 'QXY', 'Qxy', 'Q_xy', 'Q_XY']
        
        qz_dim = None
        qxy_dim = None
        for dim in img.dims:
            if dim in qz_aliases:
                qz_dim = dim
            if dim in qxy_aliases:
                qxy_dim = dim
        
        if qz_dim is None or qxy_dim is None:
            raise ValueError("The input DataArray must have 'qxy' or an alias and 'qz' or an alias as dimensions.")
        
        # chi_min, chi_max = chi_range
        # qr_min, qr_max = qr_range
        
        # if sum_direction not in ['chi', 'qr']:
        #     raise ValueError("Invalid sum_direction. Choose either 'chi' or 'qr'.")
        
        # Step 2: Coordinate Conversion
        qxy = img.coords[qxy_dim].values
        qz = img.coords[qz_dim].values

        return qxy, qz

class GapInfo:
    def __init__(self, start, end, length):
        self.start = start
        self.end = end
        self.length = length

class ImageInterpolator:
    def __init__(self):
        pass

    def simple_interpolate(self, img, direction="vertical", method="linear"):
        
            if direction not in ["vertical", "horizontal"]:
                raise ValueError(f"Invalid direction. Expected 'vertical' or 'horizontal', got {direction}.")
            
            if method not in ["linear", "nearest", "zero", "slinear", "quadratic", "cubic",
                            "polynomial", "barycentric", "krog", "pchip", "spline", "akima"]:
                raise ValueError(f"Invalid method. Got {method}.")
            
            # coords = list(img.coords.keys())
            coords = img.coords
            img = img.where(img > 0, np.nan)
            if direction == "vertical":
                # img = img.interp({coords[0]: img[coords[0]].values}, method=method)
                coord_label = list(coords.keys())[1]  # Assuming the vertical coordinate label is the second key
                img_interp = img.interpolate_na(dim=coord_label, method=method)
            else:
                # img = img.interp({coords[1]: img[coords[1]].values}, method=method)
                coord_label = list(coords.keys())[0]  # Assuming the horizontal coordinate label is the first key
                img_interp = img.interpolate_na(dim=coord_label, method=method)
                
            return img_interp

    @staticmethod
    def linear_interpolate(img, direction):
        """
        Perform linear interpolation using xarray's built-in function.

        Parameters:
        - img (xarray.DataArray): The image data in xarray format.
        - direction (str): Direction of interpolation, either 'vertical' or 'horizontal'.

        Returns:
        - xarray.DataArray: Interpolated image.
        """
        coords = img.coords
        img = img.where(img > 0, np.nan)
        if direction == 'vertical':
            coord_label = list(coords.keys())[1]  # Assuming the vertical coordinate label is the second key
            img_interp = img.interpolate_na(dim=coord_label, method='linear')
        elif direction == 'horizontal':
            coord_label = list(coords.keys())[0]  # Assuming the horizontal coordinate label is the first key
            img_interp = img.interpolate_na(dim=coord_label, method='linear')
        else:
            raise ValueError("Invalid direction. Choose either 'vertical' or 'horizontal'.")
        
        return img_interp
    
    def save_dataarray_to_netcdf(self, dataarray, filename):
        dataarray.to_netcdf(filename)

    def find_gaps(self, img):
        gap_indices = np.argwhere(np.isnan(img.values))
        if len(gap_indices) == 0:
            return None, None

        vertical_gaps = defaultdict(list)
        horizontal_gaps = defaultdict(list)

        visited = set()

        for x, y in gap_indices:
            # Skip if this point is already part of another gap
            if (x, y) in visited:
                continue

            # Check vertical gaps
            if x > 0 and x < img.shape[0] - 1:
                if not np.isnan(img.values[x - 1, y]):
                    start_x = x
                    end_x = x
                    while end_x < img.shape[0] - 1 and np.isnan(img.values[end_x, y]):
                        visited.add((end_x, y))
                        end_x += 1
                    if not np.isnan(img.values[end_x, y]):
                        length = end_x - start_x
                        vertical_gaps[length].append(GapInfo(start=(start_x, y), end=(end_x, y), length=length))

            # Check horizontal gaps
            if y > 0 and y < img.shape[1] - 1:
                if not np.isnan(img.values[x, y - 1]):
                    start_y = y
                    end_y = y
                    while end_y < img.shape[1] - 1 and np.isnan(img.values[x, end_y]):
                        visited.add((x, end_y))
                        end_y += 1
                    if not np.isnan(img.values[x, end_y]):
                        length = end_y - start_y
                        horizontal_gaps[length].append(GapInfo(start=(x, start_y), end=(x, end_y), length=length))

        return vertical_gaps, horizontal_gaps

    def fill_smallest_gaps(self, img, vertical_gaps, horizontal_gaps):
        filled = False

        if vertical_gaps:
            min_vertical_length = min(vertical_gaps.keys())
            for gap_info in vertical_gaps[min_vertical_length]:
                start_x, start_y = gap_info.start
                end_x, end_y = gap_info.end

                top_neighbor = img.values[start_x - 1, start_y] if not np.isnan(img.values[start_x - 1, start_y]) else 0
                bottom_neighbor = img.values[end_x, end_y] if not np.isnan(img.values[end_x, end_y]) else 0
                fill_value = (top_neighbor + bottom_neighbor) / 2
                img.values[start_x:end_x, start_y] = fill_value
                filled = True

        if horizontal_gaps:
            min_horizontal_length = min(horizontal_gaps.keys())
            for gap_info in horizontal_gaps[min_horizontal_length]:
                start_x, start_y = gap_info.start
                end_x, end_y = gap_info.end

                left_neighbor = img.values[start_x, start_y - 1] if not np.isnan(img.values[start_x, start_y - 1]) else 0
                right_neighbor = img.values[end_x, end_y] if not np.isnan(img.values[end_x, end_y]) else 0
                fill_value = (left_neighbor + right_neighbor) / 2
                img.values[start_x, start_y:end_y] = fill_value
                filled = True

        return filled

    def patch_interpolate(self, img):
        iteration_count = 0
        img = img.where(img > 0, np.nan)

        while True:
            vertical_gaps, horizontal_gaps = self.find_gaps(img)
            if vertical_gaps is None and horizontal_gaps is None:
                break
            min_vertical_length = min(vertical_gaps.keys()) if vertical_gaps else float('inf')
            min_horizontal_length = min(horizontal_gaps.keys()) if horizontal_gaps else float('inf')
            if min_vertical_length == float('inf') and min_horizontal_length == float('inf'):
                break
            filled = self.fill_smallest_gaps(img, vertical_gaps, horizontal_gaps)
            if not filled:
                break
            iteration_count += 1
            clear_output(wait=True)
            print(f"Iteration {iteration_count}: Smallest gap size = min({min_vertical_length}, {min_horizontal_length})")

        img = img.where(~np.isnan(img), 0)
        return img

## --- Custom-Built Interpolation Methods ---- ##
    def brute_force_interpolation(self, data_array, pixelthreshold_range, step_size):
        new_data = data_array.copy(deep=True)
        
        # Using numpy's arange to create a float-compatible range
        for pc in np.arange(pixelthreshold_range[0], pixelthreshold_range[1] + step_size, step_size):
            for axis in data_array.dims:
                other_axis = [dim for dim in data_array.dims if dim != axis][0]
                
                for i in range(len(data_array[other_axis].values)):
                    intensity_values = data_array.isel({other_axis: i}).values
                    start_gap = None
                    gap_length = 0

                    for j in range(len(intensity_values)):
                        if np.isnan(intensity_values[j]) or intensity_values[j] == 0:
                            if start_gap is None and j > 0:
                                start_gap = j - 1
                            gap_length += 1
                        else:
                            if start_gap is not None:
                                end_gap = j
                                if 0 < gap_length <= pc and end_gap < len(intensity_values) - 1:
                                    interp_values = np.linspace(intensity_values[start_gap], intensity_values[end_gap], gap_length + 2)
                                    if len(interp_values[1:-1]) == end_gap - start_gap - 1:
                                        new_data.isel({other_axis: i}).values[start_gap + 1:end_gap] = interp_values[1:-1]

                                start_gap = None
                                gap_length = 0
                        
        return new_data

    @staticmethod
    def gaussian(x, a, b, c):
        return a * np.exp(-((x - b)**2) / (2 * c**2))

    def edge_detection(self, img, method='sobel'):
        if method == 'sobel':
            edge_img = sobel(img.values)
        elif method == 'canny':
            edge_img = canny(img.values)
        else:
            raise ValueError("Invalid edge detection method.")
        
        # Define a kernel to check for adjacent zero-valued pixels
        kernel = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]])
        
        # Convolve the edge image with the kernel
        zero_adjacency = convolve2d(edge_img, kernel, mode='same')
        
        # Filter out edge pixels that are not adjacent to zero-valued pixels
        filtered_edge_img = edge_img * (zero_adjacency > 0)
        
        return filtered_edge_img

    def classify_gaps(self, edge_img, tolerance = 5, padding = 5):
        # Initialize gap information
        gaps = {'horizontal': [], 'vertical': [], 'missing_wedge': []}
        n_rows, n_cols = edge_img.shape
        
        # Loop through rows and columns to identify gaps
        for i, row in enumerate(edge_img):
            gap_start = None
            for j, pixel in enumerate(row):
                if pixel == 0 and gap_start is None:
                    gap_start = j
                elif pixel != 0 and gap_start is not None:
                    gaps['horizontal'].append((i, gap_start, j))
                    gap_start = None

        for i, col in enumerate(edge_img.T):
            gap_start = None
            for j, pixel in enumerate(col):
                if pixel == 0 and gap_start is None:
                    gap_start = j
                elif pixel != 0 and gap_start is not None:
                    gaps['vertical'].append((i, gap_start, j))
                    gap_start = None
        
        # Identify potential missing wedges (gaps centered at q_z = 0)
        center = edge_img.shape[0] // 2
        # tolerance = 5  # Adjust as needed
        for i, start, end in gaps['horizontal']:
            if abs(i - center) < tolerance:
                gaps['missing_wedge'].append((i, start, end))

        # Curvature checking logic
        for direction, gap_list in gaps.items():
            new_gap_list = []
            for gap in gap_list:
                # padding = 5  # Default padding for curvature check
                
                if direction == 'horizontal':
                    start = max(gap[1] - padding, 0)
                    end = min(gap[2] + padding, n_cols)
                    surrounding_pixels = edge_img[gap[0], start:end]
                else:
                    start = max(gap[1] - padding, 0)
                    end = min(gap[2] + padding, n_rows)
                    surrounding_pixels = edge_img[start:end, gap[0]]

                p = Polynomial.fit(np.arange(len(surrounding_pixels)), surrounding_pixels, 2)
                
                # Check curvature. You may adjust the tolerance.
                if len(p.convert().coef) > 2 and abs(p.convert().coef[2]) > 1e-5:
                    new_gap_list.append(gap)

            gaps[direction] = new_gap_list

        return gaps

    def interpolate_gaps(self, img, gaps, threshold=0.1):
        interpolated_img = img.copy()

        # Horizontal interpolation
        for i, start, end in gaps['horizontal']:
            if end - start < img.shape[1] * threshold:
                interpolated_img[i, start:end] = np.interp(
                    np.arange(start, end),
                    [start - 1, end],
                    [img[i, start - 1], img[i, end]]
                )

        # Vertical interpolation
        for i, start, end in gaps['vertical']:
            interpolated_img[start:end, i] = np.interp(
                np.arange(start, end),
                [start - 1, end],
                [img[start - 1, i], img[end, i]]
            )

        # Missing wedge interpolation logic
        # Identify potential missing wedges (gaps centered at q_z = 0)
        potential_wedges = [gap for gap in gaps['horizontal'] if abs(gap[0] - img.shape[0] // 2) < 5]

        for i, start, end in potential_wedges:
            # Check if the gap size is below the threshold
            if end - start < img.shape[1] * threshold:
                # Interpolate across the gap along q_z at q_xy = 0
                interpolated_img[i, start:end] = np.interp(
                    np.arange(start, end),
                    [start - 1, end],
                    [img[i, start - 1], img[i, end]]
                )

        return interpolated_img

    def handle_interpolation(self, img, edge_method='Sobel', threshold=0.1):
        edge_img = self.edge_detection(img, method=edge_method)
        gaps = self.classify_gaps(edge_img)
        return self.interpolate_gaps(img, gaps, threshold)
    

    '''
    # -- Exports the current class instantiation when called.
    # def exportzarr(self, zarrPath: Union[str, pathlib.Path], projectName: str):
    #     # Create the project directory
    #     project_path = pathlib.Path(zarrPath) / projectName
    #     if project_path.exists():
    #         # Handle existing project folder (e.g., ask for confirmation or raise an error)
    #         raise FileExistsError(f"Project folder '{project_path}' already exists. Choose a different project name or remove the existing folder.")
    #     project_path.mkdir(parents=True, exist_ok=False)  # exist_ok=False ensures that an error is raised if the folder exists

    #     # Save xarray DataArrays as Zarr files and TIFF images
    #     for key in ['rawtiff_xr', 'reciptiff_xr', 'cakedtiff_xr']:
    #         ds = self.__dict__[key].to_dataset(name=key)
    #         ds_path = project_path / f"{key}.zarr"
    #         ds.to_zarr(ds_path)

    #         # Convert the xarray DataArray to a numpy array and save as TIFF
    #         tiff_image = ds[key].values
    #         tiff_path = project_path / f"{projectName}_{key}.tiff"
    #         with TiffWriter(str(tiff_path)) as tif:
    #             tif.save(tiff_image.astype(np.uint16))  # Adjust dtype as needed

    #     # Save other attributes to a JSON file
    #     attributes_to_save = {
    #         'basePath': str(self.basePath),
    #         'poniPath': str(self.poniPath),
    #         'maskPath': str(self.maskPath),
    #         'tiffPath': str(self.tiffPath),
    #         'metadata_keylist': self.metadata_keylist,
    #         'attribute_dict': self.attribute_dict,
    #         'energy': self.energy,
    #     }
    #     json_path = project_path / "attributes.json"
    #     with open(json_path, 'w') as file:
    #         json.dump(attributes_to_save, file)

    '''