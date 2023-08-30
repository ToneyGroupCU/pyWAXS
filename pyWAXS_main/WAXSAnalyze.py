import os, pathlib, tifffile, pyFAI, pygix, json, zarr, random, inspect
import xarray as xr
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
from scipy.spatial import KDTree
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from skimage import feature
from sklearn.neighbors import KDTree
from scipy.signal import convolve2d
from PIL import Image
from typing import Union, Tuple
import matplotlib.pyplot as plt
from tifffile import TiffWriter
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
from skimage.filters import sobel
from skimage.feature import canny
from collections import defaultdict
from numpy.polynomial.polynomial import Polynomial


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
        self.rawtiff_xr.plot.imshow(interpolation='antialiased', cmap='jet',
                                    vmin=np.nanpercentile(self.rawtiff_xr, 10),
                                    vmax=np.nanpercentile(self.rawtiff_xr, 99))
        plt.title('Raw TIFF Image')
        plt.show()

    # -- Display the Reciprocal Space Map using XArray
    def recipdisplay_xr(self):
        plt.close('all')
        self.reciptiff_xr.plot.imshow(interpolation='antialiased', cmap='jet',
                                    vmin=np.nanpercentile(self.reciptiff_xr, 10),
                                    vmax=np.nanpercentile(self.reciptiff_xr, 99))
        plt.title('Missing Wedge Correction')
        plt.show()

    # -- Display the Caked Image using XArray
    def cakeddisplay_xr(self):
        plt.close('all')
        self.cakedtiff_xr.plot.imshow(interpolation='antialiased', cmap='jet',
                                    vmin=np.nanpercentile(self.cakedtiff_xr, 10),
                                    vmax=np.nanpercentile(self.cakedtiff_xr, 99))
        plt.title('Caked Image')
        plt.show()

    # -- Display Image (General)
    def display_image(self, img, title='Image', cmap='jet'):
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

## --- IMAGE INTERPOLATION  --- ##
    def brute_force_interpolate(self, img: xr.DataArray, gap_threshold: int = 5) -> xr.DataArray:
        """
        Brute-force interpolation method to fill gaps in an image.
        
        Parameters:
            img (xr.DataArray): Input image data
            gap_threshold (int): Maximum size of gaps to interpolate
        
        Returns:
            xr.DataArray: Interpolated image
        """
        interpolated_img = img.copy()
        
        rows, cols = img.shape
        
        for col in range(cols):
            gap_start = None
            for row in range(rows):
                if img[row, col] != 0 and gap_start is not None:
                    gap_end = row
                    
                    # Check if the gap size is within the threshold
                    if gap_end - gap_start <= gap_threshold:
                        # Interpolate
                        interpolated_img[gap_start:gap_end, col] = np.interp(
                            np.arange(gap_start, gap_end),
                            [gap_start - 1, gap_end],
                            [img[gap_start - 1, col], img[gap_end, col]]
                        )
                    
                    gap_start = None
                elif img[row, col] == 0 and gap_start is None:
                    gap_start = row
                    
        return interpolated_img

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

## --- PEAK SEARCHING --- ##
    # Method for generating Monte Carlo points
    def generate_monte_carlo_points(self, num_points=100):
        img = self.reciptiff_xr.values
        x_max, y_max = img.shape
        points = [(random.randint(0, x_max-1), random.randint(0, y_max-1)) for _ in range(num_points)]
        return points

    # Method for adaptive gradient ascent
    def adaptive_gradient_ascent(self, point, threshold=0.001, initial_neighborhood=3):
        x, y = point
        img = self.normalized_img
        neighborhood_size = initial_neighborhood
        max_x, max_y = img.shape
        # Add history to check for convergence
        history = []

        while True:
            x = max(0, min(x, max_x - 1))
            y = max(0, min(y, max_y - 1))

            if img[x, y] <= 0:
                x, y = x + 1, y + 1
                continue

            x_min = max(0, x - neighborhood_size)
            x_max = min(img.shape[0], x + neighborhood_size)
            y_min = max(0, y - neighborhood_size)
            y_max = min(img.shape[1], y + neighborhood_size)
            
            neighborhood = img[x_min:x_max, y_min:y_max]
            local_max_x, local_max_y = np.unravel_index(neighborhood.argmax(), neighborhood.shape)
            local_max_x += x_min
            local_max_y += y_min

            history.append((local_max_x, local_max_y))
            if len(history) > 50:
                last_50 = np.array(history[-50:])
                avg = np.mean(last_50, axis=0)
                if np.all(np.abs((last_50 - avg) / avg) < 0.05):
                    return avg

            if np.abs(local_max_x - x) <= threshold and np.abs(local_max_y - y) <= threshold:
                return (local_max_x, local_max_y)

            x, y = local_max_x, local_max_y

            if self.snr < 1.0:
                neighborhood_size *= 2

    # Method to find peaks
    def find_peaks(self, num_points=100, threshold=0.1, initial_neighborhood=3):
        self.normalize_and_calculate_SNR()
        points = self.generate_monte_carlo_points(num_points)
        peak_points = []
        for point in points:
            peak_point = self.adaptive_gradient_ascent(point, threshold, initial_neighborhood)
            if peak_point:
                peak_points.append(peak_point)
        peak_points = self.group_and_find_median(peak_points)
        self.visualize_peaks(peak_points)

    # Method to group similar points and find the median
    def group_and_find_median(self, peak_points, distance_threshold=0.03):
        peak_points = np.array(peak_points)
        grouped_peaks = defaultdict(list)
        visited = set()

        for i, point1 in enumerate(peak_points):
            if i in visited:
                continue
            group = [point1]
            visited.add(i)

            for j, point2 in enumerate(peak_points[i+1:], start=i+1):
                dist = np.linalg.norm(point1 - point2)
                if dist < distance_threshold:
                    group.append(point2)
                    visited.add(j)

            median_point = np.median(np.array(group), axis=0)
            grouped_peaks[tuple(median_point)] = len(group)

        return grouped_peaks

    # Method to visualize peaks
    def visualize_peaks(self, peak_points, point_size_factor=20, coords: dict = None):
        img = self.reciptiff_xr.values
        plt.close('all')
        vmin = np.nanpercentile(img, 10)
        vmax = np.nanpercentile(img, 99)
        plt.imshow(np.flipud(img), 
                cmap='jet', 
                vmin=vmin, 
                vmax=vmax, 
                extent=[coords['x_min'], 
                        coords['x_max'], 
                        coords['y_min'], 
                        coords['y_max']])
        for point, count in peak_points.items():
            plt.scatter(point[1], point[0], s=count * point_size_factor, c='white')
        plt.title('Peaks Visualized')
        plt.xlabel('qxy')
        plt.ylabel('qz')
        plt.colorbar()
        plt.show()

'''
# -- Alternative 2D Peak Finder
    def detect_2D_peaks(self, threshold=0.5):
        # Finding peaks in image intensity
        peaks = find_peaks(self.corrected_tiff, threshold)
        self.peak_positions_pixel = np.array(peaks[0])

        # Getting coordinates with respect to the image
        self.peak_positions_coords = [self.pixel_to_coords(pixel) for pixel in self.peak_positions_pixel]
'''

## -- PSEUDOCODE LOGIC SEGMENT -- ##
# This will normalize the image, calculate SNR, generate random points,
# perform adaptive gradient ascent to find peaks, group them, and visualize them.

    # Normalize Intensity
    # Filter (Gaussian Noise)
    # Singular Value Decomposition (SVD) to remove noise and reconstruct peaks.
    # Find Peaks (Monte Carlo - Momentum Gradient Ascent)
    # Group Peak Points/Bin Peaks
        # Manual Regrouping Option
    # Tabulate Coordinates & Intensity of Peaks
        # Store Normalization, Filtering, SVD Compression, Search Algorithm (neighborhood) and Peak Position Data as a .json
    
    # -- Experiment Class
    # Predictive Algorithm for 'Best Initial Guess'
        # Input constraints on lattice parameters
        # Look for and tabulate multiples of peaks in q_xy 
        # Look for and tabulate multiples of peaks in q_z
        # Provide best guess with confidence and error for the lattice constants
        # Select set of peaks to evaluate sigma values for azimuthal smearing
    # Load CIFs into DiffSim object
    # Load 'Best Initial Guess' to match appropriate CIF and orientation parameters
    # Simulate Bragg Peak Positions in CIFs (no intensity)
        # Load possible CIFs to test against a, b, c, alpha, beta, gamma values
            # Comparator Bragg Peak Simulation (Compute Crystal)
        # Best match for position, simulate intensities and smearing from initial guesses.
            # diffraction .py