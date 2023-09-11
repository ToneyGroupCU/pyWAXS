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
import matplotlib.pyplot as plt

class WAXSSearch:
    def __init__(self):
        self.DoG = None  # Difference of Gaussians (DoG)
        self.maskedDoG = None  # Masked version of DoG

    # (MAIN) Find Peaks Using Difference of Gaussians: Ver. 1, Recenter -> Cluster
    def waxssearch_main(self, img_xr, sigma1=1.0, sigma2=2.0, threshold=0.006, clustering_method='DBSCAN', eps=1, min_samples=2, k=3, radius=5):
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
    
## -- METHODS FOR DISPLAYING OUTPUTS -- ##
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


## -- PeakSearch for an Xarray DataSet    
class PeakSearch2D:
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

# -- Add algorithms here for comparison of peak positions found in the WAXSSearch algorithm
class PeakPairing:
    def __init__(self) -> None:
        pass

# -- Add a class here to search 1D images (using Xarray DataSets)
class PeakSearch1D:
    def __init__(self) -> None:
        pass
