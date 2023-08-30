import os, pathlib, tifffile, pyFAI, pygix, json, zarr, random
import xarray as xr
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
from scipy.spatial import KDTree
from scipy.interpolate import griddata
from PIL import Image
from typing import Union, Tuple
import matplotlib.pyplot as plt
from tifffile import TiffWriter
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
from collections import defaultdict


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

        # - Initialize GIXSTransform() object
        self.GIXSTransformObj = self.detCorrObj() # Create Detector Object
        self.apply_image_corrections() # Apply Image Corrections
        
        # - Image Smoothing & Normalization
        self.smoothed_img = None # Store Smoothed Image
        self.normalized_img = None # Store Normalized Image
        self.snrtemp = None # Temporary signal-to-noise ratio 

        # self.caked_data_np, self.recip_data_np, self.qz_np, self.qxy_np, self.chi_np, self.qr_np = None, None, None, None, None, None # datatypes: numpy arrays
        # self.peak_positions_pixel = None
        # self.peak_positions_coords = None
        # self.apply_detector_corrections()

    # -- Imports/Exports the current class instantiation when called.
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

 # -- Display Image
    def display_image(self, img, title='Image', cmap='jet'):
        plt.close('all')
        
        # Check for invalid or incompatible types
        if img is None or not isinstance(img, (np.ndarray, xr.DataArray)):
            raise ValueError("The input image is None or not of a compatible type.")
        
        # Check for xarray DataArray and convert to numpy array if needed
        if isinstance(img, xr.DataArray):
            img = img.values
            
        # Check for empty or all NaN array
        if np.all(np.isnan(img)) or img.size == 0:
            raise ValueError("The input image is empty or contains only NaN values.")
                
        vmin = np.nanpercentile(img, 10)
        vmax = np.nanpercentile(img, 99)

        if self.coords is not None:
            plt.imshow(np.flipud(img), cmap=cmap, vmin=vmin, vmax=vmax, 
                    extent=[self.coords['x_min'], 
                            self.coords['x_max'], 
                            self.coords['y_min'], 
                            self.coords['y_max']])
        else:
            plt.imshow(np.flipud(img), 
                    cmap=cmap, 
                    vmin=vmin, 
                    vmax=vmax)

        plt.title(title)
        plt.xlabel('qxy')
        plt.ylabel('qz')
        plt.colorbar()
        plt.show()

# -- Normalize Image
    def normalize_image(self, normalizerecip = False):
        if self.reciptiff_xr is None:
            raise ValueError("Reciprocal space image data is not available.")
            
        img = self.reciptiff_xr.values
        max_val = np.max(img)
        if max_val <= 0:
            raise ValueError("Image maximum intensity is zero or negative, cannot normalize.")
        
        self.normalized_img = img / max_val

        if normalizerecip == True:
            self.reciptiff_xr.values = self.normalized_img
            self.reciptiff_xr.attrs['normalized'] = True
        
        return self.normalized_img

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

    '''
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
        
        smoothed = xr.DataArray(smoothed, coords=img.coords, dims=img.dims)
        self.smoothed_img = smoothed.astype(original_dtype)
        return self.smoothed_img
    '''

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

    def cartesian_to_polar(q_xy, q_z):
        """
        Convert Cartesian coordinates (q_xy, q_z) to polar coordinates (q_r, chi).
        
        Parameters:
            q_xy (np.ndarray): Array of in-plane momentum transfer values.
            q_z (np.ndarray): Array of out-of-plane momentum transfer values.
            
        Returns:
            q_r (np.ndarray): Array of radial distance values.
            chi (np.ndarray): Array of angle values in degrees.
        """
        q_r = np.sqrt(q_xy ** 2 + q_z ** 2)
        chi = np.degrees(np.arctan2(q_xy, q_z))
        return q_r, chi

    def interpolate_image(self, img: xr.DataArray, pixel_tolerance: float = 0.1) -> xr.DataArray:
        # Step 1: Conversion to Polar Coordinates
        q_xy, q_z = np.meshgrid(img.coords['q_xy'], img.coords['q_z'])
        q_r = np.sqrt(q_xy**2 + q_z**2)
        chi = np.arctan2(q_xy, q_z)
        
        # Step 2: Masking
        mask = img != 0
        
        # Create KDTree for efficient spatial search
        coords = np.column_stack((q_r[mask], chi[mask]))
        tree = KDTree(coords)
        
        # Step 3: Interpolation along q_r
        zero_pixels = np.column_stack((q_r[~mask], chi[~mask]))
        interpolated_values = np.zeros(zero_pixels.shape[0])
        
        for i, (q, c) in enumerate(zero_pixels):
            # Find pixels within the pixel_tolerance in q_r
            idx = tree.query_ball_point([q, c], pixel_tolerance)
            if len(idx) == 0:
                continue  # Skip if no neighbors found
            neighbors = coords[idx]
            values = img.values[(neighbors[:, 0], neighbors[:, 1])]
            interpolated_values[i] = np.mean(values)
            
        # Step 4: Special Interpolation at q_z = 0
        qz_zero_pixels = zero_pixels[np.isclose(zero_pixels[:, 1], 0, atol=1e-6)]
        for q in qz_zero_pixels[:, 0]:
            # Gather 40 pixels on either side
            idx = tree.query_ball_point([q, 0], pixel_tolerance, count_only=40)
            neighbors = coords[idx]
            values = img.values[(neighbors[:, 0], neighbors[:, 1])]
            
            # Fit Gaussian
            popt, _ = curve_fit(self.gaussian, neighbors[:, 0], values)
            interpolated_values[qz_zero_pixels[:, 0] == q] = self.gaussian(q, *popt)
            
        # Step 5: Apply Interpolation
        img.values[~mask] = interpolated_values
        
        return img
    
    # Gaussian function for curve fitting
    def gaussian(self, x, a, b, c):
        return a * np.exp(-((x - b)**2) / (2 * c**2))

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
# -- Alternative Image Processing: Filtering/Smoothing, Peak Searching, Indexing
    def filter_and_smooth(self, sigma=1):
        # Applying Gaussian smoothing to the corrected TIFF
        self.corrected_tiff = gaussian_filter(self.corrected_tiff, sigma=sigma)

# -- Alternative 2D Peak Finder
    def detect_2D_peaks(self, threshold=0.5):
        # Finding peaks in image intensity
        peaks = find_peaks(self.corrected_tiff, threshold)
        self.peak_positions_pixel = np.array(peaks[0])

        # Getting coordinates with respect to the image
        self.peak_positions_coords = [self.pixel_to_coords(pixel) for pixel in self.peak_positions_pixel]

    def interpolate_masked_image(img, mask, q_z_center, q_r_tolerance=5, num_points=40, method='linear'):
            """
            Interpolate masked regions in an image considering a polar coordinate framework (qr, chi).
            
            Parameters:
                img (np.ndarray): Input image with masked regions.
                mask (np.ndarray): Boolean mask indicating the masked regions in the image.
                q_z_center (tuple): The (x, y) coordinate of the origin (0, 0) in Cartesian coordinates.
                q_r_tolerance (float): Tolerance for considering pixels as similar in q_r value.
                num_points (int): Number of points to consider when fitting Gaussian across the q_z axis.
                method (str): Interpolation method ('linear', 'cubic', etc.).
                
            Returns:
                np.ndarray: Interpolated image.
            """
            
            # Get indices of non-masked and masked pixels
            non_masked_indices = np.column_stack(np.where(mask))
            masked_indices = np.column_stack(np.where(~mask))
            
            # Translate indices to Cartesian coordinates centered at q_z_center
            non_masked_coords = non_masked_indices - np.array(q_z_center)
            masked_coords = masked_indices - np.array(q_z_center)
            
            # Compute q_r values for non-masked and masked coordinates
            q_r_non_masked = np.linalg.norm(non_masked_coords, axis=1)
            q_r_masked = np.linalg.norm(masked_coords, axis=1)
            
            # Initialize interpolated image
            interpolated_img = img.copy()
            
            # Interpolate masked pixels
            for i, (x, y) in enumerate(masked_indices):
                q_r_value = q_r_masked[i]
                
                # Find non-masked pixels within the q_r tolerance
                within_tolerance = np.abs(q_r_non_masked - q_r_value) <= q_r_tolerance
                if np.sum(within_tolerance) == 0:
                    continue  # No points within tolerance to interpolate
                
                # Interpolate pixel value
                points_to_use = non_masked_indices[within_tolerance]
                values_to_use = img[points_to_use[:, 0], points_to_use[:, 1]]
                interpolated_value = griddata(points_to_use, values_to_use, (x, y), method=method)
                interpolated_img[x, y] = interpolated_value
            
            # Fit a Gaussian profile across the q_z axis (chi = 0)
            q_z_x, q_z_y = q_z_center
            x_indices = np.arange(q_z_x - num_points, q_z_x + num_points + 1)
            y_indices = np.full_like(x_indices, q_z_y)
            gaussian_points = img[x_indices, y_indices]
            
            # Fit Gaussian and interpolate across q_z axis
            # Here, you would fit a Gaussian model to gaussian_points and populate the q_z axis
            # For demonstration, the mean value is used
            interpolated_img[q_z_x - num_points:q_z_x + num_points + 1, q_z_y] = np.mean(gaussian_points)
            
            return interpolated_img

'''

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
    # 


'''
# -- Image Smoothing Algorithm
    def smooth_image(self, img, method: str = 'gaussian', sigma: float = 1.0, **kwargs) -> xr.DataArray:
        """
        Smooth the input image using the specified method.

        Parameters:
            img (xarray.DataArray or np.ndarray): Input image to be smoothed.
            method (str): The smoothing method to use ('gaussian', 'bilateral', 'total_variation', 'anisotropic').
            sigma (float): Sigma value for Gaussian smoothing. Default is 1.0.
            **kwargs: Additional parameters for the smoothing method.

        Returns:
            xarray.DataArray: Smoothed image.
        """
        
        # Check if input is xarray DataArray, and keep track
        is_xarray = False
        if isinstance(img, xr.DataArray):
            is_xarray = True
            original_coords = img.coords
            original_attrs = img.attrs
            img = img.values
        else:
            if not isinstance(img, np.ndarray):
                raise ValueError("Input must be either an xarray DataArray or a numpy array.")

        original_dtype = img.dtype

        # Perform smoothing
        if method == 'gaussian':
            smoothed = gaussian_filter(img, sigma)
        elif method == 'bilateral':
            sigma_color = kwargs.get('sigma_color', 0.1)
            sigma_spatial = kwargs.get('sigma_spatial', 15)
            smoothed = denoise_bilateral(img, sigma_color=sigma_color, sigma_spatial=sigma_spatial, multichannel=False)
        elif method == 'total_variation':
            weight = kwargs.get('weight', 0.1)
            smoothed = denoise_tv_chambolle(img, weight=weight)
        elif method == 'anisotropic':
            # Placeholder for anisotropic diffusion method (needs to be implemented)
            smoothed = img.copy()
        else:
            raise ValueError("Invalid method. Choose from 'gaussian', 'bilateral', 'total_variation', 'anisotropic'.")

        # Convert the smoothed image back to xarray DataArray if original input was xarray
        if is_xarray:
            smoothed = xr.DataArray(smoothed, coords=original_coords, attrs=original_attrs)

        self.smoothed_img = smoothed

        return self.smoothed_img

#     def smooth_image(self, img: np.ndarray, method: str = 'gaussian', **kwargs) -> np.ndarray:
#         """
#         Smooth the input image using the specified method.
        
#         Parameters:
#             img (np.ndarray): Input image to be smoothed.
#             method (str): The smoothing method to use ('gaussian', 'bilateral', 'total_variation', 'anisotropic').
#             **kwargs: Additional parameters for the smoothing method.
            
#         Returns:
#             np.ndarray: Smoothed image with the same data type and shape as the input.
#         """
#         original_dtype = img.dtype
        
#         if method == 'gaussian':
#             sigma = kwargs.get('sigma', 1)
#             smoothed = gaussian_filter(img, sigma)
#         elif method == 'bilateral':
#             print('This method is broken...') # note to fix.
#             sigma_color = kwargs.get('sigma_color', 0.05)
#             sigma_spatial = kwargs.get('sigma_spatial', 15)
#             smoothed = denoise_bilateral(img, sigma_color=sigma_color, sigma_spatial=sigma_spatial, multichannel=False)
#         elif method == 'total_variation':
#             weight = kwargs.get('weight', 0.1)
#             smoothed = denoise_tv_chambolle(img, weight=weight)
#         elif method == 'anisotropic':
#             # Placeholder for anisotropic diffusion method (needs to be implemented)
#             smoothed = img.copy()
#         else:
#             raise ValueError("Invalid method. Choose from 'gaussian', 'bilateral', 'total_variation', 'anisotropic'.")
        
#         self.smoothed_img = smoothed.astype(original_dtype)

#         # Convert the smoothed image back to the original data type
#         # return smoothed.astype(original_dtype)
#         return self.smoothed_img


# -- Calculating Signal-to-Noise Ratio (Internal)
    def calculate_SNR_for_class(self):
        """
        Calculate the Signal-to-Noise Ratio (SNR) for each xarray DataArray in the class.
        The SNR is stored as an attribute for each DataArray.
        """
        for attr_name in ['rawtiff_xr', 'reciptiff_xr', 'cakedtiff_xr']:
            xarray_obj = getattr(self, attr_name, None)
            if xarray_obj is not None:
                mean_val = np.mean(xarray_obj.values)
                std_val = np.std(xarray_obj.values)
                snr = mean_val / std_val if std_val != 0 else 0
                xarray_obj.attrs['SNR'] = snr

    def calculate_SNR(self, xarray_obj):
        """
        Calculate the Signal-to-Noise Ratio (SNR) for an external xarray DataArray or numpy array.
        The SNR is stored as a temporary attribute 'snrtemp'.

        Parameters:
            xarray_obj (xarray.DataArray or np.ndarray): The DataArray or numpy array for which to calculate SNR.

        Returns:
            None
        """
        if not isinstance(xarray_obj, (xr.DataArray, np.ndarray)):
            raise ValueError("Input must be either an xarray DataArray or a numpy array.")
        
        # If the input is a numpy array, convert it to xarray DataArray
        if isinstance(xarray_obj, np.ndarray):
            xarray_obj = xr.DataArray(xarray_obj)
        
        mean_val = np.mean(xarray_obj.values)
        std_val = np.std(xarray_obj.values)
        snr = mean_val / std_val if std_val != 0 else 0
        xarray_obj.attrs['SNR_temp'] = snr
        self.snrtemp = snr

        return xarray_obj
    '''

'''
# def calculate_SNR(self):
    #     if not hasattr(self, 'normalized_img'):
    #         raise AttributeError("'WAXSAnalyze' object has no attribute 'normalized_img'. Run 'normalize_image()' first.")
    #     self.snr = np.mean(self.normalized_img) / np.std(self.normalized_img)

# def normalize_image(self):
    #     img = self.reciptiff_xr.values.copy()
    #     max_val = np.max(img)
        
    #     if max_val <= 0:
    #         raise ValueError("Image maximum intensity is zero or negative, cannot normalize.")
        
    #     self.normalized_img = img / max_val
    #     # self.reciptiff_xr.values = self.normalized_img
        
    #     # return self.reciptiff_xr  # return the xarray DataArray
    #     return self.normalized_img

  # def display_image(self, img: np.ndarray, title: str = 'Image', cmap: str = 'jet', coords: dict = None):
    #     """
    #     Display the image using matplotlib.

    #     Parameters:
    #         img (np.ndarray): Image to be displayed.
    #         title (str): Title of the plot.
    #         cmap (str): Colormap to be used.
    #         coords (dict): Coordinate system to be used for plotting.

    #     Returns:
    #         None
    #     """
    #     plt.close('all')
    #     vmin = np.nanpercentile(img, 10)
    #     vmax = np.nanpercentile(img, 99)
    #     plt.imshow(np.flipud(img), 
    #                cmap='jet', 
    #                vmin=vmin, 
    #                vmax=vmax, 
    #                extent=[coords['x_min'], 
    #                        coords['x_max'], 
    #                        coords['y_min'], 
    #                        coords['y_max']])
    #     plt.title(title)
    #     plt.xlabel('qxy')
    #     plt.ylabel('qz')
    #     plt.colorbar()
    #     plt.show()

    def createSampleDictionary(self, root_folder):
        """
        Loads and creates a sample dictionary from a root folder path.
        The dictionary will contain: sample name, scanID list, series scanID list, 
        a pathlib object variable for each sample's data folder (which contains the /maxs/raw/ subfolders),
        and time_start and exposure_time for each series of scans.
        
        The method uses alias mappings to identify important metadata from the filenames:
        SCAN ID : Defines the scan ID number in the convention used at 11-BM (CMS), specific to a single shot exposure or time series.
            aliases : scan_id: 'scanid', 'id', 'scannum', 'scan', 'scan_id', 'scan_ID'
        SERIES NUMBER : Within a series (fixed SCAN ID), the exposure number in the series with respect to the starting TIME START (clocktime)
            aliases : series_number: 'seriesnum', 'seriesid', 'series_id', 'series_ID', 'series', 'series_number', 'series_num'
        TIME START : Also generically referred to as CLOCK TIME, logs the start of the exposure or series acquisition. This time is constant for all exposures within a series.
            aliases : time_start: 'start_time', 'starttime', 'start', 'clocktime', 'clock', 'clockpos', 'clock_time', 'time', 'time_start'
        EXPOSURE TIME : The duration of a single shot or exposure, either in a single image or within a series.
            aliases : 'exptime', 'exp_time', 'exposuretime', 'etime', 'exp', 'expt', 'exposure_time'
        """

        # Ensure the root_folder is a pathlib.Path object
        self.root_folder = pathlib.Path(root_folder)
        if not self.root_folder.is_dir():
            raise ValueError(f"Directory {self.root_folder} does not exist.")
        
        # Initialize the sample dictionary
        sample_dict = {}
        
        # Alias mappings for scan_id, series_number, time_start, and exposure_time
        scan_id_aliases = ['scanid', 'id', 'scannum', 'scan', 'scan_id', 'scan_ID']
        series_number_aliases = ['seriesnum', 'seriesid', 'series_id', 'series_ID', 'series', 'series_number', 'series_num']
        time_start_aliases = ['start_time', 'starttime', 'start', 'clocktime', 'clock', 'clockpos', 'clock_time', 'time', 'time_start']
        exposure_time_aliases = ['exptime', 'exp_time', 'exposuretime', 'etime', 'exp', 'expt', 'exposure_time']

        # Identify the indices of the required metadata in the naming scheme
        for idx, alias in enumerate(self.md_naming_scheme):
            if alias.lower() in [alias.lower() for alias in scan_id_aliases]:
                self.scan_id_index = idx
            if alias.lower() in [alias.lower() for alias in series_number_aliases]:
                self.series_number_index = idx

        if self.scan_id_index is None or self.series_number_index is None:
            raise ValueError('md_naming_scheme does not contain keys for scan_id or series_number.')

        # Update sample_dict with new information
        for sample_folder in self.root_folder.iterdir():
            if sample_folder.is_dir():
                # Confirm that this is a sample folder by checking for /maxs/raw/ subfolder
                maxs_raw_dir = sample_folder / 'maxs' / 'raw'
                if maxs_raw_dir.is_dir():
                    # Sample folder checks out, extract scan_id, series_number, time_start, and exposure_time
                    sample_name = sample_folder.name
                    scan_list = []
                    series_list = {}  # Initialize series_list as an empty dictionary
                    
                    for image_file in maxs_raw_dir.glob('*'):
                        # Load metadata from image
                        metadata = self.loadMd(image_file)
                        
                        # Lowercase all metadata keys for case insensitivity
                        metadata_lower = {k.lower(): v for k, v in metadata.items()}
                        
                        # Find and store scan_id, series_number, time_start, and exposure_time
                        scan_id = metadata_lower.get(self.md_naming_scheme[self.scan_id_index].lower())
                        series_number = metadata_lower.get(self.md_naming_scheme[self.series_number_index].lower())
                        time_start = next((metadata_lower[key] for key in metadata_lower if key in time_start_aliases), None)
                        exposure_time = next((metadata_lower[key] for key in metadata_lower if key in exposure_time_aliases), None)

                        # Add them to our lists
                        scan_list.append(scan_id)
                        
                        # Check if scan_id is in series_list, if not, create a new list
                        if scan_id not in series_list:
                            series_list[scan_id] = []

                        series_list[scan_id].append((series_number, time_start, exposure_time))
                    
                    # Store data in dictionary
                    sample_dict[sample_name] = {
                        'scanlist': scan_list,
                        'serieslist': series_list,
                        'path': sample_folder
                    }

        self.sample_dict = sample_dict
        return sample_dict

    def selectSampleAndSeries(self):
            """
            Prompts the user to select a sample and one or more series of scans from that sample.
            The user can choose to select all series of scans.
            The selections will be stored as the 'selected_series' attribute and returned.
            """
            # Check if sample_dict has been generated
            if not self.sample_dict:
                print("Error: Sample dictionary has not been generated. Please run createSampleDictionary() first.")
                return

            while True:
                # Show the user a list of sample names and get their selection
                print("Please select a sample (or 'q' to exit):")
                sample_names = list(self.sample_dict.keys())
                for i, sample_name in enumerate(sample_names, 1):
                    print(f"[{i}] {sample_name}")
                print("[q] Exit")
                selection = input("Enter the number of your choice: ")
                if selection.lower() == 'q':
                    print("Exiting selection.")
                    return self.selected_series
                else:
                    sample_index = int(selection) - 1
                    selected_sample = sample_names[sample_index]

                # Show the user a choice between single image or image series and get their selection
                print("\nWould you like to choose a single image or an image series? (or 'q' to exit)")
                print("[1] Single Image")
                print("[2] Image Series")
                print("[q] Exit")
                choice = input("Enter the number of your choice: ")
                if choice.lower() == 'q':
                    print("Exiting selection.")
                    return self.selected_series
                choice = int(choice)

                # Get the selected sample's scan list and series list
                scan_list = self.sample_dict[selected_sample]['scanlist']
                series_list = self.sample_dict[selected_sample]['serieslist']

                # Identify series scan IDs and single image scan IDs
                series_scan_ids = set(series_list.keys())
                single_image_scan_ids = [scan_id for scan_id in scan_list if scan_id not in series_scan_ids]

                if choice == 1:
                    # The user has chosen to select a single image
                    print("\nPlease select a scan ID (or 'q' to exit):")
                    for i, scan_id in enumerate(single_image_scan_ids, 1):
                        print(f"[{i}] {scan_id}")
                    print("[q] Exit")
                    selection = input("Enter the number of your choice: ")
                    if selection.lower() == 'q':
                        print("Exiting selection.")
                        return self.selected_series
                    else:
                        scan_id_index = int(selection) - 1
                        selected_scan = single_image_scan_ids[scan_id_index]
                        self.selected_series.append((selected_sample, selected_scan))
                else:
                    # The user has chosen to select an image series
                    print("\nPlease select one or more series (Enter 'a' to select all series, 'q' to finish selection):")
                    selected_series = []
                    while True:
                        for i, series_scan_id in enumerate(series_scan_ids, 1):
                            series_data = series_list[series_scan_id]
                            print(f"[{i}] Series {series_scan_id} (start time: {series_data[0][1]}, exposure time: {series_data[0][2]})")
                        print("[a] All series")
                        print("[q] Finish selection")
                        selection = input("Enter the number(s) of your choice (comma-separated), 'a', or 'q': ")
                        if selection.lower() == 'q':
                            if selected_series:
                                break
                            else:
                                print("Exiting selection.")
                                return self.selected_series
                        elif selection.lower() == 'a':
                            selected_series = list(series_scan_ids)
                            break
                        else:
                            # Get the series indices from the user's input
                            series_indices = list(map(int, selection.split(',')))
                            selected_series += [list(series_scan_ids)[i-1] for i in series_indices]
                    self.selected_series.extend([(selected_sample, series) for series in selected_series])

                print("\nSelection completed.")
            return self.selected_series

# -- Display the RAW TIFF using Matplotlib
    def rawdisplay(self):
        plt.close('all')
        # plt.imshow(self.rawtiff_xr, cmap='jet')
        lb = np.nanpercentile(self.rawtiff_xr, 10)
        ub = np.nanpercentile(self.rawtiff_xr, 99)

        extent = [self.rawtiff_xr['pix_x'].min(), self.rawtiff_xr['pix_x'].max(),
          self.rawtiff_xr['pix_y'].min(), self.rawtiff_xr['pix_y'].max()]
        
        plt.imshow(self.rawtiff_xr, 
                   interpolation='nearest', 
                   cmap='jet',
                   origin='lower', 
                   vmax=ub, 
                   vmin=lb,
                   extent=extent)
        plt.xlabel(self.rawtiff_xr['pix_x'].name)
        plt.ylabel(self.rawtiff_xr['pix_y'].name)
        plt.title('Raw TIFF Image')
        plt.colorbar(label='Intensity')
        plt.show()

# -- Display the Reciprocal Space Map Corrected TIFF using Matplotlib
    def recipdisplay(self):
        plt.close('all')
        lb = np.nanpercentile(self.reciptiff_xr, 10)
        ub = np.nanpercentile(self.reciptiff_xr, 99)

        extent = [self.reciptiff_xr[self.inplane_config].min(), self.reciptiff_xr[self.inplane_config].max(),
                self.reciptiff_xr['q_z'].min(), self.reciptiff_xr['q_z'].max()]

        plt.imshow(self.reciptiff_xr,
                interpolation='nearest',
                cmap='jet',
                origin='lower',
                vmax=ub,
                vmin=lb,
                extent=extent)

        plt.xlabel(self.reciptiff_xr[self.inplane_config].name) # 'q$_{xy}$ (1/$\AA$)'
        plt.ylabel(self.reciptiff_xr['q_z'].name) # 'q$_{z}$ (1/$\AA$)'
        plt.title('Reciprocal Space Corrected Image')
        plt.colorbar(label='Intensity')
        plt.show()

# -- Display the Caked TIFF using Matplotlib
    def cakeddisplay(self):
        plt.close('all')
        # plt.imshow(self.cakedtiff_xr, cmap='jet')
        lb = np.nanpercentile(self.cakedtiff_xr, 10)
        ub = np.nanpercentile(self.cakedtiff_xr, 99)

        extent = [self.cakedtiff_xr['qr'].min(), self.cakedtiff_xr['qr'].max(),
                self.cakedtiff_xr['chi'].min(), self.cakedtiff_xr['chi'].max()]

        plt.imshow(self.cakedtiff_xr,
                interpolation='nearest',
                cmap='jet',
                origin='lower',
                vmax=ub,
                vmin=lb,
                extent=extent)

        plt.xlabel(self.cakedtiff_xr['qr'].name)
        plt.ylabel(self.cakedtiff_xr['chi'].name)
        plt.title('Caked Corrected Image')
        plt.colorbar(label='Intensity')
        plt.show()

'''