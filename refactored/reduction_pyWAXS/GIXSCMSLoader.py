import os, pathlib, datetime, warnings, json
from PIL import Image
import xarray as xr
import pandas as pd
import numpy as np

class GIXSCMSLoader():
    """
    GIXS Data Loader Class | NSLS-II 11-BM (CMS)
    Used to load single TIFF time-series TIFF GIWAXS images.
    """

    def __init__(self, metadata_keylist = []):
        """
        Description: Initialize instance with metadata key list. Default is an empty list.
        """
        # - Path Information
        self.basePath = None  # basePath used throughout the class methods to build additional paths.

        # - Metadata Attributes
        self.metadata_keylist = metadata_keylist # metadata keylist pertaining to your files
        self.attribute_dict = None  # attributes_dict

        # - Single Image Information
        self.singleimg_numpy = None  # single image numpy array
        self.singleimg_xarray = None  # single image xarray
        self.filePath = None # single image filepath
        
        # - Multi-/Series Image Information
        self.filePaths = None  # filePaths used in the loadMultiImage()
        self.multiimg_xarray = None  # the concatenated xarray

    def loadSingleImage(self, filePath):
        """
        Description: Loads a single xarray (DataArray) from a designated raw TIFF filePath.
        Handle Variables
            filePath : string
                Filepath passed to the loadMetaData method that is used to extract metadata relevant to the TIFF image.
        
        Method Variables
            singleimg_numpy : numpy array
                Numpy array of the information contained in the TIFF image found in filePath
            
            singleimg_xarray : xarray DataArray
                DataArray object containing both metadata and TIFF data mapped to x/y coordinates.
        """

        # Check that the path exists before continuing.
        if not pathlib.Path(filePath).is_file():
            raise ValueError(f"File {filePath} does not exist.")
        
        # Open the image from the filePath
        image = Image.open(filePath)

        # Create a numpy array from the image
        self.singleimg_numpy = np.array(image)

        # Run the loadMetaData method to construct the attribute dictionary for the filePath.
        self.attribute_dict = self.loadMetaData(filePath)

        # Convert the image numpy array into an xarray DataArray object.
        self.singleimg_xarray = xr.DataArray(data = self.singleimg_numpy, 
                                dims=['pix_y', 'pix_x'], # label the coordinate dimensions of the image
                                attrs=self.attribute_dict) # connect the extracted metadata as the attributes dictionary s
        
        # Map the dimension labels to the DataArray object.
        self.singleimg_xarray = self.singleimg_xarray.assign_coords({
            'pix_x': self.singleimg_xarray.pix_x.data,
            'pix_y': self.singleimg_xarray.pix_y.data
        })
        return self.singleimg_xarray
    
    def loadMetaData(self, filePath, delim = '_'):
        """
        Description: Uses metadata_keylist to generate attribute dictionary of metadata based on filename.
        Handle Variables
            filePath : string
                Filepath passed to the loadMetaData method that is used to extract metadata relevant to the TIFF image.
            delim : string
                String used as a delimiter in the filename. Defaults to an underscore '_' if no other delimiter is passed.
        
        Method Variables
            attribute_dict : dictionary
                Dictionary of metadata attributes created using the filename and metadata list passed during initialization.
            
        """
        self.attribute_dict = {} # Initialize the dictionary.

        filename = pathlib.Path(filePath).stem # strip the filename from the filePath
        metadata_list = filename.split(delim) # splits the filename based on the delimter passed to the loadMetaData method.

        # Check to make sure the metadata list is at least of appropriate length, if not, there
        # is a certain mismatch in the keylist and metadata able to be extracted from the filename.
        if len(metadata_list) != len(self.metadata_keylist):
            raise ValueError("Filename metadata items do not match with metadata keylist.")
        
        # Map the keylist onto the attribute dictionary with metadata info extracted from the filename.
        for i, metadata_item in enumerate(self.metadata_keylist):
            self.attribute_dict[metadata_item] = metadata_list[i]

        return self.attribute_dict
    
    def loadMultiImage(self, basePath, filter, time_start = 0):
        """
        Description: Load multiple raw TIFFs into an xarray (DataArray).
            basePath : string
                Outer directory corresponding to folderpath with all data for a given sample.
            filter : string
                Value to search all files in folder for (glob string), used to downselect relevant TIFFs.
            time_start : integer
                Timestamp to 
        """
        # Set the basePath attribute as the pathlib object.
        self.basePath = pathlib.Path(basePath)

        # Check that the basePath actually exists.
        if not self.basePath.is_dir():
            raise ValueError(f"Directory {basePath} does not exist.")
        
        # Use the glob filter to extract the relevant filepaths.
        self.filePaths = list(self.basePath.glob(f'*{filter}*'))

        # If there are no files corresponding to the glob filter, let the user know.
        if not self.filePaths:
            raise ValueError(f"No files found with filter {filter} in directory {basePath}.")
        
        data_rows = [] # define an empty data_rows list
        for filePath in self.filePaths:
            singleimg_xarray = self.loadSingleImage(filePath) # load the image with the loadSingleImage() method.
            singleimg_xarray = singleimg_xarray.assign_coords({'series_number': int(singleimg_xarray.attrs.get('series_number', 0))}) # get the series number from the attributes dictionary.
            singleimg_xarray = singleimg_xarray.expand_dims(dim={'series_number': 1}) # expand the xarray dimension
            data_rows.append(singleimg_xarray)

        # Create the multiple image/series xarray after looping through the filepaths.
        self.multiimg_xarray = xr.concat(data_rows, 'series_number')
        self.multiimg_xarray = self.multiimg_xarray.sortby('series_number')
        self.multiimg_xarray = self.multiimg_xarray.assign_coords({
            'series_number': self.multiimg_xarray.series_number.data,
            'time': ('series_number', 
                     self.multiimg_xarray.series_number.data*np.round(float(self.multiimg_xarray.attrs.get('exposure_time', '0')[:-1]),
                                                     1)+np.round(float(self.multiimg_xarray.attrs.get('exposure_time', '0')[:-1]),1)+time_start)
        })

        # Replace the series number with the actual time value as a float64.
        self.multiimg_xarray = self.multiimg_xarray.swap_dims({'series_number': 'time'})
        self.multiimg_xarray = self.multiimg_xarray.sortby('time')
        if 'series_number' in self.multiimg_xarray.attrs:
            del self.multiimg_xarray.attrs['series_number']
        return self.multiimg_xarray
