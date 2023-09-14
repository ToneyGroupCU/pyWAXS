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
from scipy.sparse import csr_matrix
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
import gc

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
        
        # # Check if zarrPath is provided
        # if zarrPath is not None:
        #     self.zarrPath = zarrPath
        #     if projectName:
        #         self.projectName = projectName
        #         self.loadzarr(zarrPath = self.zarrPath, 
        #                       projectName = self.projectName)
        # else:
        #     # Check that the required parameters are provided
        #     if poniPath is None or maskPath is None or tiffPath is None:
        #         raise ValueError("Must provide either zarrPath or poniPath, maskPath, and tiffPath.")

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

    # # -- Imports the current class instantiation when called.
    # def loadzarr(self, zarrPath: Union[str, pathlib.Path], projectName: str):
    #     # Define the project directory
    #     project_path = pathlib.Path(zarrPath) / projectName

    #     # Load xarray DataArrays from Zarr files
    #     for key in ['rawtiff_xr', 'reciptiff_xr', 'cakedtiff_xr']:
    #         ds_path = project_path / f"{key}.zarr"
    #         ds = xr.open_zarr(ds_path)
    #         self.__dict__[key] = ds[key]

    #     # Load other attributes from the JSON file
    #     json_path = project_path / "attributes.json"
    #     with open(json_path, 'r') as file:
    #         attributes = json.load(file)
    #         self.basePath = pathlib.Path(attributes['basePath'])
    #         self.poniPath = pathlib.Path(attributes['poniPath'])
    #         self.maskPath = pathlib.Path(attributes['maskPath'])
    #         self.tiffPath = pathlib.Path(attributes['tiffPath'])
    #         self.metadata_keylist = attributes['metadata_keylist']
    #         self.attribute_dict = attributes['attribute_dict']
    #         self.energy = attributes['energy']

    #     # Rebuild GIXSTransformObj and load single image
    #     self.GIXSTransformObj = self.detCorrObj()
    #     self.loadSingleImage(self.tiffPath)

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

    def boxcut1D(self, img, qxyslice=[0, 2.5], qzslice=[0, 2.5], boxcutsum='qz'):
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

        Attributes:
        - self.boxcut1D_xr: Stores the resulting 1D DataArray after slicing and summing.

        Output: 
        Returns the 1D DataArray after slicing and summing, and stores it in self.boxcut1D_xr.
        """
        # Check if boxcutsum is either 'qxy' or 'qz'
        if boxcutsum not in ['qxy', 'qz']:
            print("Invalid boxcutsum value. Defaulting to 'qz'.")
            boxcutsum = 'qz'

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
        boxcut1D_xr = img.sel(**slicing_dict).sum(boxcutsum)

        # Assign DataArray attributes
        boxcut1D_xr.attrs['qxymin'] = qxymin
        boxcut1D_xr.attrs['qxymax'] = qxymax
        boxcut1D_xr.attrs['qzmin'] = qzmin
        boxcut1D_xr.attrs['qzmax'] = qzmax
        boxcut1D_xr.attrs['sumdir'] = boxcutsum

        # Assign coordinate names
        coord_name = qxy_dim if boxcutsum == 'qz' else qz_dim
        # Check if the dimension still exists after summing
        if coord_name in boxcut1D_xr.dims:
            boxcut1D_xr = boxcut1D_xr.rename({coord_name: 'intensity'})

        # Save as a class attribute
        self.boxcut1D_xr = boxcut1D_xr

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

        '''
        # # Step 2: Coordinate Conversion
        # qxy = img[qxy_dim]
        # qz = img[qz_dim]

        # # Debugging: Check the range of qxy, qz, chi, and qr
        # print(f"qxy range: {qxy.min().item()}, {qxy.max().item()}")
        # print(f"qz range: {qz.min().item()}, {qz.max().item()}")

        # chi = np.arctan2(qz, qxy) * (180 / np.pi)
        # qr = np.sqrt(qxy**2 + qz**2)

        # print(f"chi range: {chi.min().item()}, {chi.max().item()}")
        # print(f"qr range: {qr.min().item()}, {qr.max().item()}")
        
        # # Step 3: Masking and Slicing
        # # mask = (chi >= chi_min) & (chi <= chi_max) & (qr >= qr_min) & (qr <= qr_max)
        # # self.azimuth_mask = mask
        # # self.original_total_intensity = img.where(mask).sum().item()

        # # # Step 4: Construct the Azimuthal Matrix
        # # A = self._create_azimuthal_matrix(chi, qr, mask, chi_range, qr_range, discretization, sum_direction)
        
        # # # Step 5: Azimuthal Summation Using Sparse Matrix
        # # I = img.values.flatten()
        # # S = A.dot(I)
        
        # chi, qr = self.convert_to_chi_qr(img.coords['qxy'].values, img.coords['qz'].values)

        # mask = (chi >= chi_range[0]) & (chi <= chi_range[1]) & (qr >= qr_range[0]) & (qr <= qr_range[1])
        # img_masked = img.where(mask)

        # self.azimuth_mask = img_masked
        # self.original_total_intensity = img.where(mask).sum().item()
        
        # # Convert to chi and qr space
        # img_transformed = self.convert_to_chi_qr(img.coords['qxy'].values, img.coords['qz'].values)

        # # Interpolate along chi
        # img_interpolated = self.interpolate_along_chi(chi, qr, img_transformed)

        # # Your existing azimuthal integration code here, now working on img_interpolated
        # self.azimuth1D_xr_sum = xr.DataArray(img_interpolated, coords=[np.arange(chi_range[0], chi_range[1], discretization) if sum_direction == 'chi' else np.arange(qr_range[0], qr_range[1], discretization)], dims=[sum_direction])

        # # return img_interpolated
        
        # # Step 6: Verification of Pixel Splitting
        # if not np.isclose(self.azimuth1D_xr_sum.sum().item(), self.original_total_intensity, atol=1e-6):
        #     raise ValueError("Pixel splitting error: Total intensity mismatch.")
            
        # return self.azimuth1D_xr_sum
        '''
   
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
    
    def display_image1D(self, img1D, color='red'):
        """
        Description: Plots a 1D DataArray using matplotlib.

        Variables:
        - img1D: The 1D Xarray DataArray to be plotted.
        - color: Color of the plot line.

        Output:
        Displays the plot.
        """

        plt.figure(figsize=(10, 6))
        img1D.plot.line(color=color)

        plt.ylabel('Intensity (arb. units)')
        plt.xlabel(img1D.attrs.get('sumdir', 'Coordinate'))
        
        plt.title('1D Integrated Image')
        plt.grid(True)

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

# Define the methods and functions
class AzimuthalIntegrator:

    @staticmethod
    def convert_to_chi_qr(qxy, qz):
        qxy_mesh, qz_mesh = np.meshgrid(qxy, qz)
        chi = np.arctan2(qz_mesh, qxy_mesh) * (180 / np.pi)
        qr = np.sqrt(qxy_mesh ** 2 + qz_mesh ** 2)
        return chi, qr

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

        qxy = img.coords[qxy_dim].values
        qz = img.coords[qz_dim].values

        chi, qr = self.convert_to_chi_qr(qxy, qz)

        mask = (chi >= chi_range[0]) & (chi <= chi_range[1]) & (qr >= qr_range[0]) & (qr <= qr_range[1])
        img_masked = img.where(mask)
        self.azimuth_mask = img_masked
        self.original_total_intensity = img.where(mask).sum().item()

        img_interpolated = self.interpolate_along_chi(chi, qr, img_masked.values)

        self.azimuth1D_xr_sum = xr.DataArray(img_interpolated, 
            coords=[np.arange(chi_range[0], chi_range[1], discretization) if sum_direction == 'chi' else np.arange(qr_range[0], qr_range[1], discretization)], 
            dims=[sum_direction])

        return self.azimuth1D_xr_sum

    def generate_test_data(self, shape=(100, 100), num_arcs=3, arc_width=3):
        y, x = np.ogrid[-shape[0]//2:shape[0]//2, -shape[1]//2:shape[1]//2]
        mask = np.zeros(shape)
        for i in range(num_arcs):
            radius_inner = i * (arc_width + 5)
            radius_outer = radius_inner + arc_width
            mask_arc = (x ** 2 + y ** 2 >= radius_inner ** 2) & (x ** 2 + y ** 2 < radius_outer ** 2)
            mask += mask_arc

        mask = mask * (y >= 0)

        test_data = xr.DataArray(mask, coords=[('qz', np.linspace(-1, 1, shape[0])), ('qxy', np.linspace(-1, 1, shape[1]))])
        return test_data
    
    @staticmethod
    def generate_test_data(qxy_min, qxy_max, qz_min, qz_max, num_points):
        """
        Generate a test dataset simulating concentric half arcs in qxy vs qz space.
        
        Parameters:
        - qxy_min, qxy_max: Min and max values for qxy coordinate.
        - qz_min, qz_max: Min and max values for qz coordinate.
        - num_points: Number of points along each dimension.
        
        Returns: 
        - test_data: 2D Xarray DataArray representing the test data.
        """
        qxy_values = np.linspace(qxy_min, qxy_max, num_points)
        qz_values = np.linspace(qz_min, qz_max, num_points)
        test_data = np.zeros((num_points, num_points))
        
        for i in range(num_points):
            for j in range(num_points):
                qxy = qxy_values[j]
                qz = qz_values[i]
                r = np.sqrt(qxy**2 + qz**2)
                
                # Create concentric half arcs
                if 0.5 <= r <= 0.6 or 0.9 <= r <= 1.0:
                    angle = np.arctan2(qz, qxy) * (180 / np.pi)
                    if 0 <= angle <= 180:
                        test_data[i, j] = 1

        coords = {'qxy': qxy_values, 'qz': qz_values}
        test_data = xr.DataArray(test_data, coords=coords, dims=['qz', 'qxy'])
        return test_data
    
    @staticmethod
    def plot_test_data(test_data):
        """
        Plot the test data in qxy vs qz space.
        
        Parameters:
        - test_data: 2D Xarray DataArray representing the test data.
        """
        plt.imshow(test_data, origin='lower', aspect='auto', extent=[test_data.qxy.min(), test_data.qxy.max(), test_data.qz.min(), test_data.qz.max()])
        plt.colorbar(label='Intensity')
        plt.xlabel('qxy')
        plt.ylabel('qz')
        plt.title('Test Data (qxy vs qz)')
        plt.show()

    def convert_to_chi_qr(qxy, qz):
        chi = np.arctan2(qz, qxy) * (180 / np.pi)
        qr = np.sqrt(qxy**2 + qz**2)
        return chi, qr

    def generate_mock_data_with_variable_intensity(size=200, num_arcs=5, intensity_base=100, intensity_variation=20):
        x, y = np.linspace(-size // 2, size // 2, size), np.linspace(-size // 2, size // 2, size)
        xx, yy = np.meshgrid(x, y)
        rr = np.sqrt(xx**2 + yy**2)
        angle = np.arctan2(yy, xx)  # Angle in radians

        img = np.zeros((size, size))
        for i in range(1, num_arcs + 1):
            arc_mask = np.abs(rr - i * 10) < 1
            img[arc_mask] = intensity_base + intensity_variation * np.sin(3 * angle[arc_mask])

        img[:, :size // 2] = 0
        return img

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

'''
    def _create_azimuthal_matrix(self, chi, qr, mask, chi_range, qr_range, discretization, sum_direction):
        rows, cols, data = [], [], []
        values = np.arange(chi_range[0], chi_range[1], discretization) if sum_direction == 'chi' else np.arange(qr_range[0], qr_range[1], discretization)
        
        for i, val in enumerate(values):
            lower_bound = val
            upper_bound = val + discretization
            local_mask = mask & (chi >= lower_bound) & (chi < upper_bound) if sum_direction == 'chi' else \
                        mask & (qr >= lower_bound) & (qr < upper_bound)
                        
            pixel_indices = np.nonzero(local_mask)
            
            # Add a check to see if pixel_indices[0] is empty
            if pixel_indices[0].size > 0:
                # Debug: Check the maximum index value in pixel_indices[0]
                print("Max index in pixel_indices[0]:", np.max(pixel_indices[0]))
            else:
                print("pixel_indices[0] is empty.")
            
            # Debug: Check the shapes of chi and qr
            print("Shape of chi:", chi.shape)
            print("Shape of qr:", qr.shape)
            print("Max index in pixel_indices[0]:", np.max(pixel_indices[0]) if pixel_indices[0].size > 0 else "N/A")

            # Remove out-of-bounds indices
            pixel_indices = [idx for idx in pixel_indices[0] if idx < chi.shape[0] - 1]

            for j in pixel_indices:
                if j >= chi.shape[0]:
                    print(f"Index out of bounds for chi: {j}")
                elif j >= qr.shape[0]:
                    print(f"Index out of bounds for qr: {j}")
                
                rows.append(i)
                cols.append(j)
                chi_np = chi.values
                qr_np = qr.values

                pixel_split_factor = np.abs(chi_np[j] - val) / discretization if sum_direction == 'chi' else \
                                    np.abs(qr_np[j] - val) / discretization

                data.append(pixel_split_factor)


        return csr_matrix((data, (rows, cols)))

        @staticmethod
        def convert_to_chi_qr(qxy, qz):
            chi = np.arctan2(qz, qxy) * (180 / np.pi)
            qr = np.sqrt(qxy ** 2 + qz ** 2)
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
'''

'''d
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
        # Step 1: Input Validation
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

        # if 'qxy' not in img.coords or 'qz' not in img.coords:
        #     raise ValueError("The input DataArray must have 'qxy' and 'qz' as coordinates.")

        chi_min, chi_max = chi_range
        qr_min, qr_max = qr_range

        if sum_direction not in ['chi', 'qr']:
            raise ValueError("Invalid sum_direction. Choose either 'chi' or 'qr'.")

        # Step 2: Coordinate Conversion
        qxy = img[qxy_dim]
        qz = img[qz_dim]

        # Debugging: Check the range of qxy, qz, chi, and qr
        print(f"qxy range: {qxy.min().item()}, {qxy.max().item()}")
        print(f"qz range: {qz.min().item()}, {qz.max().item()}")

        chi = np.arctan2(qz, qxy) * (180 / np.pi)
        qr = np.sqrt(qxy**2 + qz**2)

        print(f"chi range: {chi.min().item()}, {chi.max().item()}")
        print(f"qr range: {qr.min().item()}, {qr.max().item()}")

        # Step 3: Masking and Slicing
        mask = (chi >= chi_min) & (chi <= chi_max) & (qr >= qr_min) & (qr <= qr_max)

        # Debug: Check the min/max values in the mask
        print(f"Mask min/max: {mask.min().item()}, {mask.max().item()}")

        # Debug: Check the number of True values in the mask
        print(f"Number of True values in mask: {mask.sum().item()}")

        self.azimuth_mask = mask
        self.original_total_intensity = img.where(mask).sum().item()

        # Debug: Check original_total_intensity
        print(f"Original total intensity: {self.original_total_intensity}")

        self.azimuth_mask = mask
        self.original_total_intensity = img.where(mask).sum().item()

        # Step 4: Azimuthal Summation
        chi_values = np.arange(chi_min, chi_max, discretization)
        qr_values = np.arange(qr_min, qr_max, discretization)

        sum_array = np.zeros(len(chi_values) if sum_direction == 'chi' else len(qr_values))

        print(f"Original total intensity: {self.original_total_intensity}")

        # Inside the for loop
        for i, val in enumerate(chi_values if sum_direction == 'chi' else qr_values):
            lower_bound = val
            upper_bound = val + discretization
            
            # Debug: Check the lower and upper bounds
            print(f"Lower bound for index {i}: {lower_bound}, Upper bound for index {i}: {upper_bound}")

            local_mask = mask & (chi >= lower_bound) & (chi < upper_bound) if sum_direction == 'chi' else \
                        mask & (qr >= lower_bound) & (qr < upper_bound)

            # Debug: Print the number of true values in local_mask
            print(f"Number of True values in local_mask for index {i}: {local_mask.sum().item()}")

            # Debug: Print the local intensity sum
            local_intensity = img.where(local_mask).sum().item()
            print(f"Local intensity sum for index {i}: {local_intensity}")

            # Debug: Print the pixel split factor
            pixel_split_factor = np.abs(chi - val) / discretization if sum_direction == 'chi' else \
                                np.abs(qr - val) / discretization
            print(f"Pixel split factor for index {i}: {pixel_split_factor.sum().item()}")

            sum_array[i] = (local_intensity * pixel_split_factor).sum().item()

        self.azimuth1D_xr_sum = xr.DataArray(sum_array, coords=[chi_values if sum_direction == 'chi' else qr_values], dims=[sum_direction])

        # Debugging prints
        print(f"Sum of azimuth1D_xr_sum: {self.azimuth1D_xr_sum.sum().item()}")

        if not np.isclose(self.azimuth1D_xr_sum.sum().item(), self.original_total_intensity, atol=1e-6):
            raise ValueError("Pixel splitting error: Total intensity mismatch.")

        return self.azimuth1D_xr_sum

        '''