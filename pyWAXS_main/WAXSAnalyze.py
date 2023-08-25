import os, pathlib, tifffile, pyFAI
import pygix
import xarray as xr
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import find_peaks
from PIL import Image
from typing import Union, Tuple
import matplotlib.pyplot as plt
import json, zarr
# from imageio import imwrite
from tifffile import TiffWriter

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
        self.GIXSTransformObj = self.detCorrObj()
        self.apply_image_corrections()

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

# -- Image & Metadata Loading
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

# -- Image Processing: Filtering/Smoothing, Peak Searching, Indexing
    def filter_and_smooth(self, sigma=1):
        # Applying Gaussian smoothing to the corrected TIFF
        self.corrected_tiff = gaussian_filter(self.corrected_tiff, sigma=sigma)

    def detect_2D_peaks(self, threshold=0.5):
        # Finding peaks in image intensity
        peaks = find_peaks(self.corrected_tiff, threshold)
        self.peak_positions_pixel = np.array(peaks[0])

        # Getting coordinates with respect to the image
        self.peak_positions_coords = [self.pixel_to_coords(pixel) for pixel in self.peak_positions_pixel]


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