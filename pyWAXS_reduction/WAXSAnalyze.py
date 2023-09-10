import os, pathlib, tifffile, pyFAI, pygix, json, zarr, random, inspect
from PIL import Image
from typing import Union, Tuple
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

# - Custom Imports
from WAXSTransform import WAXSTransform

class WAXSAnalyze:
    def __init__(self, 
                 poniPath: Union[str, pathlib.Path] = None, 
                 maskPath: Union[str, pathlib.Path, np.ndarray]  = None,
                 tiffPath: Union[str, pathlib.Path, np.ndarray]  = None,
                 metadata_keylist=[], 
                 inplane_config: str = 'q_xy', 
                 energy: float = 12.7,
                 zarrPath: Union[str, pathlib.Path] = None,
                 projectName: str = 'test'):
       
        """
        Attributes:
        poniPath (pathlib Path or str): Path to .poni file for converting to q-space 
                                        & applying missing wedge correction
        maskPath (pathlib Path or str or np.array): Path to the mask file to use 
                                for the conversion, or a numpy array

        Description: Initialize instance with metadata key list. Default is an empty list.
        """
    
        # - Path Information
        self.basePath = None # datatype: 'str' or pathlib variable, 'root_folder' basePath used throughout the class methods to build additional paths.
        self.poniPath = poniPath # datatype: 'str' or pathlib variable, PONI File Path ('.poni')
        self.maskPath = maskPath # datatype: 'str' or pathlib variable, MASK File Path ('.edf' or '.json')
        self.tiffPath = tiffPath # datatype: 'str' or pathlib variable, TIFF image filepath
        self.inplane_config = inplane_config # datatype: 'str', in-plane scattering axes label
        self.energy = energy # datatype: float, energy of your X-rays in keV

        # - Metadata Attributes
        self.metadata_keylist = metadata_keylist # datatype: list, 'md_naming_scheme'
        self.attribute_dict = None # datatype: dictionary, 'sample_dict'

        # - TIFF Image Data
        self.rawtiff_np = None # datatype: numpy array, RAW TIFF (numpy)
        self.rawtiff_xr = None # datatype: xarray DataArray, RAW TIFF (xarray)
        
        # Check if zarrPath is provided
        if zarrPath is not None:
            self.zarrPath = zarrPath
            if projectName:
                self.projectName = projectName
                self.loadzarr(zarrPath = self.zarrPath, 
                              projectName = self.projectName)
        else:
            # Check that the required parameters are provided
            if poniPath is None or maskPath is None or tiffPath is None:
                raise ValueError("Must provide either zarrPath or poniPath, maskPath, and tiffPath.")

        # - Load the Single Image
        self.loadSingleImage(self.tiffPath)
        # self.loadMetaData(self.tiffPath, delim='_') # this is done in the loadSingleImage() method

        # - Reciprocal Space Image Corrections Data Allocation
        self.reciptiff_xr = None # datatype: xarray DataArray, Reciprocal Space Corrected TIFF (xarray)

        # - Caked Image Corrections Data Allocation
        self.cakedtiff_xr = None # datatype: xarray DataArray, Caked Space Corrected TIFF (xarray)
        self.cakedtiff_sinchi_xr = None # datatype: xarray DataArray, Caked Space w/ Sin Chi Correction TIFF (xarray)

        # - Initialize GIXSTransform() object
        self.GIXSTransformObj = self.detCorrObj() # Create Detector Object
        self.apply_image_corrections() # Apply Image Corrections
        
        # - Image Smoothing & Normalization
        self.smoothed_img = None # Store Smoothed Image
        self.normalized_img = None # Store Normalized Image
        self.snrtemp = None # Temporary signal-to-noise ratio 

        self.DoG = None # Difference of Gaussians 
        self.maskedDoG = None # masked difference of Gaussians

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
    # -- Exports the current class instantiation when called.
    def exportzarr(self, zarrPath: Union[str, pathlib.Path], projectName: str):
        # Create the project directory
        project_path = pathlib.Path(zarrPath) / projectName
        if project_path.exists():
            # Handle existing project folder (e.g., ask for confirmation or raise an error)
            raise FileExistsError(f"Project folder '{project_path}' already exists. Choose a different project name or remove the existing folder.")
        project_path.mkdir(parents=True, exist_ok=False)  # exist_ok=False ensures that an error is raised if the folder exists

        # Save xarray DataArrays as Zarr files and TIFF images
        for key in ['rawtiff_xr', 'reciptiff_xr', 'cakedtiff_xr']:
            ds = self.__dict__[key].to_dataset(name=key)
            ds_path = project_path / f"{key}.zarr"
            ds.to_zarr(ds_path)

            # Convert the xarray DataArray to a numpy array and save as TIFF
            tiff_image = ds[key].values
            tiff_path = project_path / f"{projectName}_{key}.tiff"
            with TiffWriter(str(tiff_path)) as tif:
                tif.save(tiff_image.astype(np.uint16))  # Adjust dtype as needed

        # Save other attributes to a JSON file
        attributes_to_save = {
            'basePath': str(self.basePath),
            'poniPath': str(self.poniPath),
            'maskPath': str(self.maskPath),
            'tiffPath': str(self.tiffPath),
            'metadata_keylist': self.metadata_keylist,
            'attribute_dict': self.attribute_dict,
            'energy': self.energy,
        }
        json_path = project_path / "attributes.json"
        with open(json_path, 'w') as file:
            json.dump(attributes_to_save, file)

    # -- Imports the current class instantiation when called.
    def loadzarr(self, zarrPath: Union[str, pathlib.Path], projectName: str):
        # Define the project directory
        project_path = pathlib.Path(zarrPath) / projectName

        # Load xarray DataArrays from Zarr files
        for key in ['rawtiff_xr', 'reciptiff_xr', 'cakedtiff_xr']:
            ds_path = project_path / f"{key}.zarr"
            ds = xr.open_zarr(ds_path)
            self.__dict__[key] = ds[key]

        # Load other attributes from the JSON file
        json_path = project_path / "attributes.json"
        with open(json_path, 'r') as file:
            attributes = json.load(file)
            self.basePath = pathlib.Path(attributes['basePath'])
            self.poniPath = pathlib.Path(attributes['poniPath'])
            self.maskPath = pathlib.Path(attributes['maskPath'])
            self.tiffPath = pathlib.Path(attributes['tiffPath'])
            self.metadata_keylist = attributes['metadata_keylist']
            self.attribute_dict = attributes['attribute_dict']
            self.energy = attributes['energy']

        # Rebuild GIXSTransformObj and load single image
        self.GIXSTransformObj = self.detCorrObj()
        self.loadSingleImage(self.tiffPath)

## --- RAW IMAGE CORRECTIONS --- ##
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

## --- IMAGE PLOTTING --- ##
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

## --- IMAGE CORRECTION - SIN(CHI) --- ##
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

## --- IMAGE NORMALIZATION --- ##
    # -- Normalize Image
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

## --- IMAGE SMOOTHING --- ##
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

## --- IMAGE FOLDING --- ##
# Modifying the fold_image method to keep the data from the longer quadrant and append it to the folded image.
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

## --- CALCULATE SIGNAL-to-NOISE  --- ##
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

class WAXSSearch:
    def __init__(self):
        self.DoG = None  # Difference of Gaussians (DoG)
        self.maskedDoG = None  # Masked version of DoG

    # (STEP 1) Normalization: The input DataArray is normalized to '1' at the maximum value.
    def normalize_data(self, img_xr):
        return img_xr / np.nanmax(img_xr.values)

    # (STEP 2) Initial Peak Identification: Peaks are initially identified using the Difference of Gaussians (DoG).
    def initial_peak_identification(self, img_xr, sigma1, sigma2, threshold):
        img_smooth1 = gaussian_filter(img_xr.fillna(0).values, sigma=sigma1)
        img_smooth2 = gaussian_filter(img_xr.fillna(0).values, sigma=sigma2)
        self.DoG = img_smooth1 - img_smooth2
        return np.where(self.DoG >= threshold, self.DoG, np.nan)

    # (STEP 3) Edge Case Handling: Peaks at the edges or interfaces of data/no data regions are discarded or reassessed.
    def create_padding_mask(self, shape, pad_width):
        """
        Create a padding mask for a 2D array of given shape.
        
        Parameters:
        - shape (tuple): Shape of the 2D array.
        - pad_width (int): Width of the padding.
        
        Returns:
        - np.ndarray: Padding mask with 1s in the padded region and 0s elsewhere.
        """
        mask = np.ones(shape)
        mask[pad_width:-pad_width, pad_width:-pad_width] = 0
        return mask
    
    def handle_edge_cases(self, img_xr, initial_peaks, pad_width=5):
        """
        Handle edge cases by incorporating a padding mask.
        """
        edge_mask = np.isnan(gaussian_filter(img_xr.fillna(0).values, sigma=.5))
        padding_mask = self.create_padding_mask(initial_peaks.shape, pad_width)
        combined_mask = np.logical_or(edge_mask, padding_mask)
        initial_peaks[combined_mask] = np.nan
        return initial_peaks
    
    # (STEP 4) Clustering Local Maxima: For multiple peaks found around a single local maxima, we use clustering (DBSCAN or HDBSCAN) to identify groups of localized peaks.
    def cluster_local_maxima(self, initial_peaks, method='DBSCAN', eps=3, min_samples=2):
        peak_coords = np.column_stack(np.where(~np.isnan(initial_peaks)))
        if peak_coords.shape[0] > 0:
            if method == 'DBSCAN':
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(peak_coords)
            elif method == 'HDBSCAN':
                clustering = hdbscan.HDBSCAN(min_cluster_size=min_samples).fit(peak_coords)
            else:
                raise ValueError("Invalid clustering method. Choose either 'DBSCAN' or 'HDBSCAN'.")

            cluster_labels = clustering.labels_

            for cluster_id in set(cluster_labels):
                if cluster_id == -1:
                    continue
                coords_in_cluster = peak_coords[cluster_labels == cluster_id]
                intensities = self.DoG[coords_in_cluster[:, 0], coords_in_cluster[:, 1]]
                max_idx = np.argmax(intensities)
                max_coord = coords_in_cluster[max_idx]
                initial_peaks[coords_in_cluster[:, 0], coords_in_cluster[:, 1]] = np.nan
                initial_peaks[max_coord[0], max_coord[1]] = self.DoG[max_coord[0], max_coord[1]]

        return initial_peaks

    # (STEP 5) Recentering Algorithm: Recenter peaks to the local maxima.
    def recenter_peaks(self, peaks, k=3, radius=5):
        tree = KDTree(peaks)
        recentered_peaks = []
        for peak in peaks:
            k_local = min(k, len(tree.data) - 1)
            distances, neighbor_indices = tree.query(peak.reshape(1, -1), k=k_local)
            valid_indices = neighbor_indices[distances <= radius]
            neighbors = peaks[valid_indices]
            local_maxima = np.argmax(self.DoG[neighbors[:, 0], neighbors[:, 1]])
            recentered_peak = neighbors[local_maxima]
            recentered_peaks.append(recentered_peak)

        return np.array(recentered_peaks)

    # (STEP 6): Remove peaks that are caught at the NaN boundaries of the image.
    def discard_edge_peaks(self, peaks):
        """
        Discard peaks that are adjacent to NaN positions.
        
        Parameters:
        - peaks (numpy.ndarray): 2D array representing peak positions. Peaks are non-NaN values.
        
        Returns:
        - numpy.ndarray: Modified 2D array with edge peaks removed.
        """
        # Initialize a copy of the peaks array to modify
        modified_peaks = np.copy(peaks)
        
        # Loop through each position in the peaks array
        for i in range(peaks.shape[0]):
            for j in range(peaks.shape[1]):
                # Check if the position is a peak (not NaN)
                if not np.isnan(peaks[i, j]):
                    # Define the adjacent neighborhood of the peak
                    neighborhood = peaks[max(i-1, 0):min(i+2, peaks.shape[0]), max(j-1, 0):min(j+2, peaks.shape[1])]
                    
                    # Check if the neighborhood contains any NaN values
                    if np.isnan(neighborhood).any():
                        # Discard the peak if it is adjacent to a NaN position
                        modified_peaks[i, j] = np.nan
                        
        return modified_peaks
    
    # (MAIN) Find Peaks Using Difference of Gaussians: Ver. 1, Recenter -> Cluster
    def find_peaks_DoG(self, img_xr, sigma1=1.0, sigma2=2.0, threshold=0.006, clustering_method='DBSCAN', eps=1, min_samples=2, k=3, radius=5):
        """
        Find peaks in a 2D array using the Difference of Gaussians (DoG) method.

        Parameters:
        - img_xr (xarray DataArray): The input 2D array containing intensity values.
        - sigma1 (float, default=1.0): The standard deviation for the first Gaussian filter.
        - sigma2 (float, default=2.0): The standard deviation for the second Gaussian filter.
        - threshold (float, default=0.2): Threshold for initial peak identification.
        - clustering_method (str, default='DBSCAN'): The clustering method to use ('DBSCAN' or 'HDBSCAN').
        - eps (float, default=3): The maximum distance between two samples for them to be considered as in the same cluster (DBSCAN).
        - min_samples (int, default=2): The number of samples in a neighborhood for a point to be considered as a core point (DBSCAN).
        - k (int, default=3): The number of nearest neighbors to consider for the recentering algorithm.
        - radius (float, default=5): The radius within which to search for neighbors in the recentering algorithm.

        Returns:
        - img_xr (xarray DataArray): The input array with peak information stored in its attrs attribute.
        """
        # Validate sigma values
        if sigma2 <= sigma1:
            raise ValueError("sigma2 must be greater than sigma1.")
        
        # Step 0: Append mask to DataArray
        # img_xr = self.append_mask_to_dataarray(img_xr)

        # Step 1: Normalize Data
        img_normalized = self.normalize_data(img_xr)
        
        # Step 2: Initial Peak Identification
        initial_peaks = self.initial_peak_identification(img_normalized, sigma1, sigma2, threshold)
        
        # Step 3: Handle Edge Cases
        initial_peaks = self.handle_edge_cases(img_normalized, initial_peaks)
        
        # Debugging information
        print("Number of initial peaks:", np.count_nonzero(~np.isnan(initial_peaks)))

        # Step 4: Recenter Peaks
        valid_peak_coords = np.column_stack(np.where(~np.isnan(initial_peaks)))
        recentered_peak_coords = self.recenter_peaks(valid_peak_coords, k=k, radius=radius)

        # Create a new array for recentered peaks based on the original shape
        recentered_peaks = np.full(initial_peaks.shape, np.nan)
        for coord in recentered_peak_coords:
            recentered_peaks[coord[0], coord[1]] = self.DoG[coord[0], coord[1]]

        print("Number of recentered peaks:", np.count_nonzero(~np.isnan(recentered_peaks)))
        
        # Step 5: Cluster Local Maxima
        peaks = self.cluster_local_maxima(recentered_peaks, method=clustering_method, eps=eps, min_samples=min_samples)
        
        # peaks = self.discard_edge_peaks(peaks)

        print("Number of final peaks:", np.count_nonzero(~np.isnan(peaks)))
        # Initialize output DataArray for peaks
        peaks_xr = xr.DataArray(peaks, coords=img_xr.coords, dims=img_xr.dims)
        
        # Store peak information in the attrs attribute of the original DataArray
        img_xr.attrs['peaks'] = peaks_xr
        
        return img_xr

    #  Display Image Output (w/ peaks & DoG): Modified version of the display_image method to overlay scatter points for peak locations
    def display_image_with_peaks_and_DoG(self, img, title='Image with Peaks', cmap='turbo'):
        plt.close('all')
        plt.figure(figsize=(15, 7))

        DoG = self.DoG
        extent = None

        if isinstance(img, xr.DataArray):
            img_values = img.values
            peaks = img.attrs.get('peaks', None)
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
            peaks = None

        # Calculate vmin and vmax for the original image
        vmin = np.nanpercentile(img_values, 10)
        vmax = np.nanpercentile(img_values, 99)

        plt.subplot(1, 2, 1)
        plt.imshow(np.flipud(img_values),
                cmap=cmap,
                vmin=vmin,  # Use calculated vmin and vmax
                vmax=vmax,  
                extent=extent,
                aspect='auto')
        plt.colorbar()
        plt.title(f"{title} - Original")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if peaks is not None:
            peak_coords = np.column_stack(np.where(~np.isnan(peaks.values)))
            peak_x_values = peaks.coords[coords_names[1]].values[peak_coords[:, 1]]
            peak_y_values = peaks.coords[coords_names[0]].values[peak_coords[:, 0]]
            plt.scatter(peak_x_values, peak_y_values, c='red', marker='o')

        plt.subplot(1, 2, 2)
        plt.imshow(np.flipud(DoG),
                cmap=cmap,
                extent=extent,
                aspect='auto')
        plt.colorbar()
        plt.title(f"{title} - DoG")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.tight_layout()
        plt.show()

    #  Display Image Output (w/ peaks): Modified version of the display_image method to overlay scatter points for peak locations
    def display_image_with_peaks(self, img, title='Image with Peaks', cmap='turbo'):
        plt.close('all')

        # Check for invalid or incompatible types
        if img is None or not isinstance(img, (np.ndarray, xr.DataArray)):
            raise ValueError("The input image is None or not of a compatible type.")

        # Initialize extent
        extent = None

        # Check for xarray DataArray
        if isinstance(img, xr.DataArray):
            img_values = img.values
            peaks = img.attrs.get('peaks', None)  # Retrieve peaks from attrs

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
            peaks = None

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

        plt.colorbar()

        # Overlay scatter points for peak locations
        if peaks is not None:
            peak_coords = np.column_stack(np.where(~np.isnan(peaks.values)))
            peak_x_values = peaks.coords[coords_names[1]].values[peak_coords[:, 1]]
            peak_y_values = peaks.coords[coords_names[0]].values[peak_coords[:, 0]]
            plt.scatter(peak_x_values, peak_y_values, c='red', marker='o')

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.colorbar()
        plt.show()
        
    '''
## --- PEAK FINDING ALGORITHM ---- ##
    # (STEP 1) Normalization: The input DataArray is normalized to '1' at the maximum value.
    def normalize_data(self, img_xr):
        return img_xr / np.nanmax(img_xr.values)

    # (STEP 2) Initial Peak Identification: Peaks are initially identified using the Difference of Gaussians (DoG).
    def initial_peak_identification(self, img_xr, sigma1, sigma2, threshold):
        img_smooth1 = gaussian_filter(img_xr.fillna(0).values, sigma=sigma1)
        img_smooth2 = gaussian_filter(img_xr.fillna(0).values, sigma=sigma2)
        DoG = img_smooth1 - img_smooth2 # Difference of Gaussians Value
        self.DoG = DoG
        return np.where(DoG >= threshold, DoG, np.nan)

    # (STEP 3) Edge Case Handling: Peaks at the edges or interfaces of data/no data regions are discarded or reassessed.
    def handle_edge_cases(self, img_xr, initial_peaks):
        edge_mask = np.isnan(gaussian_filter(img_xr.fillna(0).values, sigma=1))
        initial_peaks[edge_mask] = np.nan
        return initial_peaks

    # (STEP 4) Clustering Local Maxima: For multiple peaks found around a single local maxima, we use clustering (DBSCAN or HDBSCAN) to identify groups of localized peaks.
    def cluster_local_maxima(self, initial_peaks, method='DBSCAN', eps=3, min_samples=2):
        peak_coords = np.column_stack(np.where(~np.isnan(initial_peaks)))
        if peak_coords.shape[0] > 0:
            if method == 'DBSCAN':
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(peak_coords)
            elif method == 'HDBSCAN':
                clustering = hdbscan.HDBSCAN(min_cluster_size=min_samples).fit(peak_coords)
            else:
                raise ValueError("Invalid clustering method. Choose either 'DBSCAN' or 'HDBSCAN'.")
            
            cluster_labels = clustering.labels_
            # print("Unique cluster labels:", set(cluster_labels))

            for cluster_id in set(cluster_labels):
                if cluster_id == -1:
                    continue
                # coords_in_cluster = peak_coords[cluster_labels == cluster_id]
                # max_coord = coords_in_cluster[np.argmax(initial_peaks[coords_in_cluster[:, 0], coords_in_cluster[:, 1]])]
                # initial_peaks[coords_in_cluster[:, 0], coords_in_cluster[:, 1]] = np.nan
                # initial_peaks[max_coord[0], max_coord[1]] = initial_peaks[max_coord[0], max_coord[1]]
            
                coords_in_cluster = peak_coords[cluster_labels == cluster_id]
                
                # Find the local maxima within the cluster instead of median
                intensities = self.DoG[coords_in_cluster[:, 0], coords_in_cluster[:, 1]]
                max_idx = np.argmax(intensities)
                max_coord = coords_in_cluster[max_idx]
                
                # Nullify other peaks in the cluster
                initial_peaks[coords_in_cluster[:, 0], coords_in_cluster[:, 1]] = np.nan
                
                # Store the local maxima
                initial_peaks[max_coord[0], max_coord[1]] = self.DoG[max_coord[0], max_coord[1]]
        
        return initial_peaks
    
    # (STEP 5) Recentering Algorithm: Recenter peaks to the local maxima.
    def recenter_peaks(self, peaks, k=3, radius=5):
        """
        Recenter peaks to the local maxima in their neighborhoods.
        
        Parameters:
        - peaks (array): 2D array of peaks with x and y coordinates.
        - k (int): Number of nearest neighbors to consider for recentering.
        - radius (float): Radius within which to search for neighbors.
        
        Returns:
        - recentered_peaks (array): Recentered peaks.
        """
        # Create a KDTree for efficient nearest neighbor search
        tree = KDTree(peaks)
        
        recentered_peaks = []
        for peak in peaks:
            # Query the KDTree to find the nearest peaks
            # distances, neighbor_indices = tree.query(peak.reshape(1, -1), k=k)
            k_local = min(k, len(tree.data) - 1)  # Adjust k based on the number of available points
            distances, neighbor_indices = tree.query(peak.reshape(1, -1), k=k_local)

            # Filter neighbors within the given radius
            valid_indices = neighbor_indices[distances <= radius]
            neighbors = peaks[valid_indices]
            
            # Check the local maxima among neighbors
            local_maxima = np.argmax(self.DoG[neighbors[:, 0], neighbors[:, 1]])
            recentered_peak = neighbors[local_maxima]
            
            # Recenter the peak
            recentered_peaks.append(recentered_peak)
        
        return np.array(recentered_peaks)
    
    # (PROCEDURE) Find Peaks Using Difference of Gaussians
    # -- Variant 1: Recenter -> Cluster
    def find_peaks_DoG(self, img_xr, sigma1=1.0, sigma2=2.0, threshold=0.006, clustering_method='DBSCAN', eps=1, min_samples=2, k=3, radius=5):
        """
        Find peaks in a 2D array using the Difference of Gaussians (DoG) method.

        Parameters:
        - img_xr (xarray DataArray): The input 2D array containing intensity values.
        - sigma1 (float, default=1.0): The standard deviation for the first Gaussian filter.
        - sigma2 (float, default=2.0): The standard deviation for the second Gaussian filter.
        - threshold (float, default=0.2): Threshold for initial peak identification.
        - clustering_method (str, default='DBSCAN'): The clustering method to use ('DBSCAN' or 'HDBSCAN').
        - eps (float, default=3): The maximum distance between two samples for them to be considered as in the same cluster (DBSCAN).
        - min_samples (int, default=2): The number of samples in a neighborhood for a point to be considered as a core point (DBSCAN).
        - k (int, default=3): The number of nearest neighbors to consider for the recentering algorithm.
        - radius (float, default=5): The radius within which to search for neighbors in the recentering algorithm.

        Returns:
        - img_xr (xarray DataArray): The input array with peak information stored in its attrs attribute.
        """
        # Validate sigma values
        if sigma2 <= sigma1:
            raise ValueError("sigma2 must be greater than sigma1.")
        
        # Step 0: Append mask to DataArray
        # img_xr = self.append_mask_to_dataarray(img_xr)

        # Step 1: Normalize Data
        img_normalized = self.normalize_data(img_xr)
        
        # Step 2: Initial Peak Identification
        initial_peaks = self.initial_peak_identification(img_normalized, sigma1, sigma2, threshold)
        
        # Step 3: Handle Edge Cases
        initial_peaks = self.handle_edge_cases(img_normalized, initial_peaks)
        
        # Debugging information
        print("Number of initial peaks:", np.count_nonzero(~np.isnan(initial_peaks)))

        # Step 4: Recenter Peaks
        valid_peak_coords = np.column_stack(np.where(~np.isnan(initial_peaks)))
        recentered_peak_coords = self.recenter_peaks(valid_peak_coords, k=k, radius=radius)

        # Create a new array for recentered peaks based on the original shape
        recentered_peaks = np.full(initial_peaks.shape, np.nan)
        for coord in recentered_peak_coords:
            recentered_peaks[coord[0], coord[1]] = self.DoG[coord[0], coord[1]]

        print("Number of recentered peaks:", np.count_nonzero(~np.isnan(recentered_peaks)))
        
        # Step 5: Cluster Local Maxima
        clustered_peaks = self.cluster_local_maxima(recentered_peaks, method=clustering_method, eps=eps, min_samples=min_samples)
        
        print("Number of final peaks:", np.count_nonzero(~np.isnan(clustered_peaks)))
        
        # Initialize output DataArray for peaks
        peaks_xr = xr.DataArray(clustered_peaks, coords=img_xr.coords, dims=img_xr.dims)
        
        # Store peak information in the attrs attribute of the original DataArray
        img_xr.attrs['peaks'] = peaks_xr
        
        return img_xr

    #  Display Image Output (w/ peaks & DoG): Modified version of the display_image method to overlay scatter points for peak locations
    def display_image_with_peaks_and_DoG(self, img, title='Image with Peaks', cmap='turbo'):
        plt.close('all')
        plt.figure(figsize=(15, 7))

        DoG = self.DoG
        extent = None

        if isinstance(img, xr.DataArray):
            img_values = img.values
            peaks = img.attrs.get('peaks', None)
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
            peaks = None

        # Calculate vmin and vmax for the original image
        vmin = np.nanpercentile(img_values, 10)
        vmax = np.nanpercentile(img_values, 99)

        plt.subplot(1, 2, 1)
        plt.imshow(np.flipud(img_values),
                cmap=cmap,
                vmin=vmin,  # Use calculated vmin and vmax
                vmax=vmax,  
                extent=extent,
                aspect='auto')
        plt.colorbar()
        plt.title(f"{title} - Original")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if peaks is not None:
            peak_coords = np.column_stack(np.where(~np.isnan(peaks.values)))
            peak_x_values = peaks.coords[coords_names[1]].values[peak_coords[:, 1]]
            peak_y_values = peaks.coords[coords_names[0]].values[peak_coords[:, 0]]
            plt.scatter(peak_x_values, peak_y_values, c='red', marker='o')

        plt.subplot(1, 2, 2)
        plt.imshow(np.flipud(DoG),
                cmap=cmap,
                extent=extent,
                aspect='auto')
        plt.colorbar()
        plt.title(f"{title} - DoG")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.tight_layout()
        plt.show()

    #  Display Image Output (w/ peaks): Modified version of the display_image method to overlay scatter points for peak locations
    def display_image_with_peaks(self, img, title='Image with Peaks', cmap='turbo'):
        plt.close('all')

        # Check for invalid or incompatible types
        if img is None or not isinstance(img, (np.ndarray, xr.DataArray)):
            raise ValueError("The input image is None or not of a compatible type.")

        # Initialize extent
        extent = None

        # Check for xarray DataArray
        if isinstance(img, xr.DataArray):
            img_values = img.values
            peaks = img.attrs.get('peaks', None)  # Retrieve peaks from attrs

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
            peaks = None

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

        plt.colorbar()

        # Overlay scatter points for peak locations
        if peaks is not None:
            peak_coords = np.column_stack(np.where(~np.isnan(peaks.values)))
            peak_x_values = peaks.coords[coords_names[1]].values[peak_coords[:, 1]]
            peak_y_values = peaks.coords[coords_names[0]].values[peak_coords[:, 0]]
            plt.scatter(peak_x_values, peak_y_values, c='red', marker='o')

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.colorbar()
        plt.show()
        
    # Variant 2: Cluster -> Recenter
    def find_peaks_DoG_CR(self, img_xr, sigma1=1.0, sigma2=2.0, threshold=0.006, clustering_method='DBSCAN', eps=1, min_samples=2, k=3, radius=5):
        """
        Find peaks in a 2D array using the Difference of Gaussians (DoG) method.

        Parameters:
        - img_xr (xarray DataArray): The input 2D array containing intensity values.
        - sigma1 (float, default=1.0): The standard deviation for the first Gaussian filter.
        - sigma2 (float, default=2.0): The standard deviation for the second Gaussian filter.
        - threshold (float, default=0.2): Threshold for initial peak identification.
        - clustering_method (str, default='DBSCAN'): The clustering method to use ('DBSCAN' or 'HDBSCAN').
        - eps (float, default=3): The maximum distance between two samples for them to be considered as in the same cluster (DBSCAN).
        - min_samples (int, default=2): The number of samples in a neighborhood for a point to be considered as a core point (DBSCAN).
        - k (int, default=3): The number of nearest neighbors to consider for the recentering algorithm.
        - radius (float, default=5): The radius within which to search for neighbors in the recentering algorithm.

        Returns:
        - img_xr (xarray DataArray): The input array with peak information stored in its attrs attribute.
        """

        if sigma2 <= sigma1:
            raise ValueError("sigma2 must be greater than sigma1.")
        
        # Step 1: Normalize Data
        img_normalized = self.normalize_data(img_xr)
        
        # Step 2: Initial Peak Identification
        initial_peaks = self.initial_peak_identification(img_normalized, sigma1, sigma2, threshold)
        
        # Step 3: Handle Edge Cases
        initial_peaks = self.handle_edge_cases(img_normalized, initial_peaks)
        
        # Debugging information
        print("Number of initial peaks:", np.count_nonzero(~np.isnan(initial_peaks)))

        # Step 4: Cluster Local Maxima
        clustered_peaks = self.cluster_local_maxima(initial_peaks, method=clustering_method, eps=eps, min_samples=min_samples)
        
        print("Number of final peaks:", np.count_nonzero(~np.isnan(clustered_peaks)))

        # Convert clustered_peaks to valid coordinates for KDTree
        valid_peak_coords = np.column_stack(np.where(~np.isnan(clustered_peaks)))
        
        # Step 5: Recenter Peaks
        recentered_peak_coords = self.recenter_peaks(valid_peak_coords, k=k, radius=radius)
        
        # Create a new array for recentered peaks based on the original shape
        recentered_peaks = np.full(clustered_peaks.shape, np.nan)
        for coord in recentered_peak_coords:
            recentered_peaks[coord[0], coord[1]] = self.DoG[coord[0], coord[1]]

        print("Number of recentered peaks:", np.count_nonzero(~np.isnan(recentered_peaks)))

        # Initialize output DataArray for peaks
        peaks_xr = xr.DataArray(recentered_peaks, coords=img_xr.coords, dims=img_xr.dims)
        
        # Store peak information in the attrs attribute of the original DataArray
        img_xr.attrs['peaks'] = peaks_xr
        
        return img_xr

    def find_peaks_DoG_DS(self, img_xr, sigma1=1.0, sigma2=2.0, threshold=0.006, clustering_method='DBSCAN', eps=1, min_samples=2, k=3, radius=5):
        """
        Find peaks in a 2D array using the Difference of Gaussians (DoG) method.

        Parameters:
        - img_xr (xarray DataArray): The input 2D array containing intensity values.
        - sigma1 (float, default=1.0): The standard deviation for the first Gaussian filter.
        - sigma2 (float, default=2.0): The standard deviation for the second Gaussian filter.
        - threshold (float, default=0.2): Threshold for initial peak identification.
        - clustering_method (str, default='DBSCAN'): The clustering method to use ('DBSCAN' or 'HDBSCAN').
        - eps (float, default=3): The maximum distance between two samples for them to be considered as in the same cluster (DBSCAN).
        - min_samples (int, default=2): The number of samples in a neighborhood for a point to be considered as a core point (DBSCAN).
        - k (int, default=3): The number of nearest neighbors to consider for the recentering algorithm.
        - radius (float, default=5): The radius within which to search for neighbors in the recentering algorithm.

        Returns:
        - img_xr (xarray DataArray): The input array with peak information stored in its attrs attribute.
        """
        # Validate sigma values
        if sigma2 <= sigma1:
            raise ValueError("sigma2 must be greater than sigma1.")
        
        # Step 0: Append mask to DataArray
        # img_xr = self.append_mask_to_dataarray(img_xr)

        # Step 1: Normalize Data
        img_normalized = self.normalize_data(img_xr)
        
        # Step 2: Initial Peak Identification
        initial_peaks = self.initial_peak_identification(img_normalized, sigma1, sigma2, threshold)
        
        # Step 3: Handle Edge Cases
        initial_peaks = self.handle_edge_cases(img_normalized, initial_peaks)
        
        # Debugging information
        print("Number of initial peaks:", np.count_nonzero(~np.isnan(initial_peaks)))

        # Step 4: Recenter Peaks
        valid_peak_coords = np.column_stack(np.where(~np.isnan(initial_peaks)))
        recentered_peak_coords = self.recenter_peaks(valid_peak_coords, k=k, radius=radius)

        # Create a new array for recentered peaks based on the original shape
        recentered_peaks = np.full(initial_peaks.shape, np.nan)
        for coord in recentered_peak_coords:
            recentered_peaks[coord[0], coord[1]] = self.DoG[coord[0], coord[1]]

        print("Number of recentered peaks:", np.count_nonzero(~np.isnan(recentered_peaks)))
        
        # Step 5: Cluster Local Maxima
        clustered_peaks = self.cluster_local_maxima(recentered_peaks, method=clustering_method, eps=eps, min_samples=min_samples)
        
        print("Number of final peaks:", np.count_nonzero(~np.isnan(clustered_peaks)))
        
        # Initialize output DataArray for peaks
        peaks_xr = xr.DataArray(clustered_peaks, coords=img_xr.coords, dims=img_xr.dims)

        # Create DataSet to include original data, mask, DoG, and maskedDoG
        output_ds = xr.Dataset({
            'original_data': img_xr,
            'mask': self.append_mask_to_dataarray(img_xr)['mask'],
            'DoG': self.DoG,
            'masked_DoG': self.maskedDoG,
            'peaks': peaks_xr
        })

        # Return the DataSet instead of DataArray
        return output_ds

    '''

class PeakSearch:
    def __init__(self, img_xr) -> None:
        self.img_xr = img_xr # Accepts an input DataArray
        self.img_ds = self.normalize_data() # Image DataSet (XArray)
            # Must convert DataArray into DataSet
        
        self.DoG = None # Difference of Gaussians (initialization)
        self.mask = None # Overlay mask (initialization)
        self.maskedDoG = None # Masked difference of Gaussians

        pass

    # (STEP 0) Append a boolean mask of null value sto the dataarray to overlay on the difference of Gaussians.
    def append_mask_to_dataarray_DS(self, img_xr):
        """
        Appends a mask layer to the xarray DataArray to set zero-intensity regions to NaN.

        Parameters:
        - img_xr (xarray DataArray): The input 2D array containing intensity values.

        Returns:
        - img_ds (xarray DataSet): The modified DataSet with an additional mask layer.
        """
        zero_intensity_mask = (img_xr > 0).astype(int)  # Convert to boolean and then to int

        # Create a DataSet from the original DataArray
        img_ds = xr.Dataset({'original_data': img_xr})

        # Add the mask as a new DataArray inside the DataSet
        img_ds['mask'] = xr.DataArray(zero_intensity_mask.values, coords=img_xr.coords, dims=img_xr.dims)

        return img_ds
    
    # (STEP 1) Normalization: The input DataArray is normalized to '1' at the maximum value.
    def normalize_data(self, img_xr):
        return img_xr['original_data'] / np.nanmax(img_xr['original_data'].values)
    
        # (STEP 2) Initial Peak Identification: Peaks are initially identified using the Difference of Gaussians (DoG).
    def initial_peak_identification(self, img_ds, sigma1, sigma2, threshold):
        img_xr = img_ds['original_data']
        img_smooth1 = gaussian_filter(img_xr.fillna(0).values, sigma=sigma1)
        img_smooth2 = gaussian_filter(img_xr.fillna(0).values, sigma=sigma2)
        DoG = img_smooth1 - img_smooth2  # Difference of Gaussians Value
        
        # Apply NaN mask to DoG using the mask attribute if present
        mask = img_xr.attrs.get('mask', None)
        if mask is not None:
            maskedDoG = np.where(mask == 0, np.nan, DoG)
        else:
            maskedDoG = DoG
            
        self.DoG = xr.DataArray(DoG, coords=img_xr.coords, dims=img_xr.dims)
        self.maskedDoG = xr.DataArray(maskedDoG, coords=img_xr.coords, dims=img_xr.dims)
        img_ds['DoG'] = self.DoG
        img_ds['maskedDoG'] = self.maskedDoG
        
        return img_ds

    # (STEP 3) Edge Case Handling: Peaks at the edges or interfaces of data/no data regions are discarded or reassessed.
    def handle_edge_cases(self, img_ds, initial_peaks):
        img_xr = img_ds['original_data']
        edge_mask = np.isnan(gaussian_filter(img_xr.fillna(0).values, sigma=1))
        initial_peaks[edge_mask] = np.nan

        # return img_ds
        return initial_peaks
    
## --- IMAGE INTERPOLATION  --- ##
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