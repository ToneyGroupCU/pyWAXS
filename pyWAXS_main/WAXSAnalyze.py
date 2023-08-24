import tifffile
import pyFAI
import pygix
import xarray as xr
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import find_peaks
import os
import pathlib
from PIL import Image
from typing import Union, Tuple

# - Custom Imports
import WAXSTransform

class WAXSAnalyze:
    def __init__(self, 
                 poniPath: Union[str, pathlib.Path], 
                 maskPath: Union[str, pathlib.Path, np.ndarray],
                 inplane_config: str = 'q_xy', 
                 energy: float = None,
                 metadata_keylist=[], 
                 tiffPath = None):
       
        """
        Attributes:
        poniPath (pathlib Path or str): Path to .poni file for converting to q-space 
                                        & applying missing wedge correction
        maskPath (pathlib Path or str or np.array): Path to the mask file to use 
                                for the conversion, or a numpy array

        Description: Initialize instance with metadata key list. Default is an empty list.
        """

        # - Path Information
        self.basePath = None # 'root_folder' basePath used throughout the class methods to build additional paths.
        self.tiffPath = None # TIFF image filepath
        self.poniPath = None # PONI File Path ('.poni')
        self.maskPath = None # MASK File Path ('.edf' or '.json')

        # - Metadata Attributes
        self.metadata_keylist = metadata_keylist # 'md_naming_scheme'
        self.attribute_dict = None # 'sample_dict'

        # - TIFF Image Data
        self.rawtiff_np = None # RAW TIFF (numpy)
        self.rawtiff_xr = None # RAW TIFF (xarray)
        
        # - Load the Single Image
        self.loadSingleImage(self.tiffPath)

        # - Reciprocal Space Image Corrections Data
        # self.reciptiff_np = None # Reciprocal Space Corrected TIFF (numpy)
        self.reciptiff_xr = None # Reciprocal Space Corrected TIFF (xarray)

        # - Caked Image Corrections Data
        # self.cakedtiff_np = None # Caked Corrected TIFF (numpy)
        self.cakedtiff_xr = None # Caked Space Corrected TIFF (xarray)

        # - Initialize GIXSTransform() object
        self.gixs_transform = self.detCorrObj()
        self.caked_data_np, self.recip_data_np, self.qz_np, self.qxy_np, self.chi_np, self.qr_np = None, None, None, None, None, None
        self.apply_image_corrections()

        # self.peak_positions_pixel = None
        # self.peak_positions_coords = None
        # self.apply_detector_corrections()

# -- Image & Metadata Loading
    def loadSingleImage(self, tiffPath):
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
        self.rawtiff_xarray = xr.DataArray(data=self.rawtiff_numpy,
                                             dims=['pix_y', 'pix_x'],
                                             attrs=self.attribute_dict)
        
        # - Map the pixel dimensions to the xarray.
        self.rawtiff_xarray = self.rawtiff_xarray.assign_coords({
            'pix_x': self.rawtiff_xarray.pix_x.data,
            'pix_y': self.rawtiff_xarray.pix_y.data
        })

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
        gixs_transform = WAXSTransform(self.poniPath, self.maskPath, ...) # Additional parameters if needed, such as pixel splitting method or corrections (solid angle)
        return gixs_transform

    def apply_image_corrections(self):
        """
        Utilizes the GIXSTransform object to create image corrections.
        Updates the reciptiff_xr and cakedtiff_xr attributes with the corrected xarray DataArrays.
        """

        # Call the pg_convert method using the rawtiff_xr xarray
        self.reciptiff_xr, self.cakedtiff_xr = self.gixs_transform.pg_convert(self.rawtiff_xr)
        self.convert_to_numpy()

    def convert_to_numpy(self):
        recip_da = self.reciptiff_xr
        caked_da = self.cakedtiff_xr
        
        self.recip_data_np = recip_da.data
        self.caked_data_np = caked_da.data
        self.qz_np = recip_da['q_z'].data
        self.qxy_np = recip_da[self.inplane_config].data
        self.chi_np = caked_da['chi'].data
        self.qr_np = caked_da['qr'].data

# -- Image Processing: Filtering/Smoothing, Peak Searching, Indexing
    def detect_2D_peaks(self, threshold=0.5):
        # Finding peaks in image intensity
        peaks = find_peaks(self.corrected_tiff, threshold)
        self.peak_positions_pixel = np.array(peaks[0])

        # Getting coordinates with respect to the image
        self.peak_positions_coords = [self.pixel_to_coords(pixel) for pixel in self.peak_positions_pixel]

    def filter_and_smooth(self, sigma=1):
        # Applying Gaussian smoothing to the corrected TIFF
        self.corrected_tiff = gaussian_filter(self.corrected_tiff, sigma=sigma)


'''
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
'''