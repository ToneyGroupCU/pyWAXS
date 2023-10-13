import os, pathlib, tifffile, pyFAI, pygix, json, zarr, random, inspect
from PIL import Image
from typing import Union, Tuple
import matplotlib.pyplot as plt
from tifffile import TiffWriter
import xarray as xr
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from pathlib import Path
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
import pandas as pd
from itertools import combinations
from typing import Dict, List
import seaborn as sns

class WAXSSearch:
    def __init__(self, data):
        self.DoG = None  # Difference of Gaussians (DoG)
        self.maskedDoG = None  # Masked version of DoG
        self.data = data

        # Convert boolean attributes to integers
        bool_attrs = ['sinchi', 'qsqr']
        for attr in bool_attrs:
            if attr in self.data.attrs:
                self.data.attrs[attr] = int(self.data.attrs[attr])

        # # Check for and handle NaNs or Infinities in the data
        # if np.isnan(self.data.values).any() or np.isinf(self.data.values).any():
        #     print("Warning: Data contains NaNs or Infinities, setting them to zero.")
        #     self.data.values = np.nan_to_num(self.data.values, nan=0.0, posinf=0.0, neginf=0.0)

        # Get dimension names from original data
        dims = self.data.dims

        # Now, convert the data into a DataSet
        self.dataset = xr.Dataset({'intensity': (dims, self.data.values)})

        # Initialize 'peak_positions' DataArray with zeros
        peak_positions = xr.DataArray(np.zeros_like(self.data.values, dtype=int), coords=self.data.coords, dims=dims)
        self.dataset['peak_positions'] = peak_positions

    # (MAIN) Find Peaks Using Difference of Gaussians: Ver. 1, Recenter -> Cluster
    def waxssearch_main(self, sigma1=1.0, sigma2=2.0, threshold=0.006, clustering_method='DBSCAN',
                        eps=1, min_samples=2, k=3, radius=5, edge_percentage=5, stricter_threshold=0.01):
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
        - edge_percentage (float, default=5): The percentage of the minimum edge length to be considered as the edge zone.
        - stricter_threshold (float, default=0.01): A stricter threshold for edge peaks.

        Returns:
        - img_xr (xarray DataArray): The input array with peak information stored in its attrs attribute.
        """
        
        img_xr = self.data

        # Validate sigma values
        if sigma2 <= sigma1 or sigma1 <= 0 or sigma2 <= 0:
            raise ValueError("Both sigma1 and sigma2 must be positive numbers and sigma2 must be greater than sigma1.")

        # Convert edge_percentage to edge_distance in pixels
        edge_distance = int(np.ceil((edge_percentage / 100) * min(img_xr.shape)))

        # Step 1: Normalize Data
        img_normalized = self.normalize_data(img_xr)
        
        # Step 2: Initial Peak Identification
        initial_peaks = self.initial_peak_identification(img_normalized, sigma1, sigma2, threshold)
        
        # Step 3: Handle Edge Cases
        initial_peaks = self.handle_edge_cases(img_xr, initial_peaks, edge_percentage=edge_percentage)
        
        # Debugging information
        # print("Number of initial peaks:", np.count_nonzero(~np.isnan(initial_peaks)))

        # Step 4: Recenter Peaks
        valid_peak_coords = np.column_stack(np.where(~np.isnan(initial_peaks)))
        recentered_peak_coords = self.recenter_peaks(valid_peak_coords, k=k, radius=radius)

        # Create a new array for recentered peaks based on the original shape
        recentered_peaks = np.full(initial_peaks.shape, np.nan)
        for coord in recentered_peak_coords:
            recentered_peaks[coord[0], coord[1]] = self.DoG[coord[0], coord[1]]

        # print("Number of recentered peaks:", np.count_nonzero(~np.isnan(recentered_peaks)))
        
        # Step 5: Cluster Local Maxima
        peaks = self.cluster_local_maxima(recentered_peaks, method=clustering_method, eps=eps, min_samples=min_samples)
        
        # Step 6: Remove edge peaks
        edge_zone = self.identify_edge_zone(img_xr.values, edge_percentage=edge_percentage)

        # Evaluate edge peaks against original image values
        valid_edge_peaks = self.evaluate_edge_peaks(peaks, edge_zone, img_xr, stricter_threshold, edge_percentage)

        # Replace edge peaks in the final peak map
        peaks[edge_zone] = valid_edge_peaks[edge_zone]

        # print("Number of final peaks:", np.count_nonzero(~np.isnan(peaks)))

        # Initialize a new DataArray for peak positions with same dimensions and coordinates as the original data
        peak_positions = xr.DataArray(np.full_like(img_xr, np.nan), coords=img_xr.coords, dims=img_xr.dims)

        # Retrieve the names of the coordinates dynamically
        coord_names = list(img_xr.dims)

        # Fill the peak_positions DataArray
        y_coords, x_coords = np.where(~np.isnan(peaks))

        # Get actual coordinate values
        y_actual_coords = img_xr.coords[coord_names[0]].values
        x_actual_coords = img_xr.coords[coord_names[1]].values

        for y_idx, x_idx in zip(y_coords, x_coords):
            y_val = y_actual_coords[y_idx]
            x_val = x_actual_coords[x_idx]
            
            if y_val in img_xr.coords[coord_names[0]].values and x_val in img_xr.coords[coord_names[1]].values:
                peak_positions.loc[{coord_names[0]: y_val, coord_names[1]: x_val}] = 1  # Use dynamic coordinate names

        # Add peak search parameters as attributes to the DataArray
        peak_positions.attrs['threshold'] = threshold
        peak_positions.attrs['eps'] = eps
        peak_positions.attrs['min_samples'] = min_samples
        peak_positions.attrs['sigma1'] = sigma1
        peak_positions.attrs['sigma2'] = sigma2
        peak_positions.attrs['clustering_method'] = clustering_method
        peak_positions.attrs['k'] = k
        peak_positions.attrs['radius'] = radius
        peak_positions.attrs['edge_percentage'] = edge_percentage
        peak_positions.attrs['stricter_threshold'] = stricter_threshold

        # Add this new DataArray to the DataSet
        self.dataset['peak_positions'] = peak_positions.astype(int)

        # Store peak information in the attrs attribute of the original DataArray
        peaks_xr = xr.DataArray(peaks, coords=img_xr.coords, dims=img_xr.dims)
        img_xr.attrs['peaks'] = peaks_xr

        # Add this new DataArray to the DataSet
        self.dataset['peak_positions'] = peak_positions.astype(int)

        # Replace all NaN values in the DataSet with 0
        self.dataset = self.dataset.fillna(0)

        return self.dataset

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
    def handle_edge_cases(self, img_xr, initial_peaks, edge_percentage=1):
        """
        Handle edge cases by incorporating a padding mask.
        """
        # Create a mask for areas adjacent to NaN values
        distance = int(np.ceil((edge_percentage / 100) * min(img_xr.shape)))  # Convert edge_percentage to distance
        nan_adj_mask = self.expand_nan_zone(img_xr.fillna(0).values, distance=distance)  # Pass distance here

        # Create a mask for the edge of the image
        padding_mask = self.create_padding_mask(initial_peaks.shape, edge_percentage=edge_percentage)
        
        # Combine the two masks
        combined_mask = np.logical_or(nan_adj_mask, padding_mask)
        
        # Remove the peaks based on the combined mask
        initial_peaks[combined_mask] = np.nan
        return initial_peaks
    
    def create_padding_mask(self, shape, edge_percentage=1):
        """
        Create a padding mask for a 2D array of given shape.
        
        Parameters:
        - shape (tuple): Shape of the 2D array.
        - edge_percentage (float): Width of the padding as a percentage of the array dimensions.
        
        Returns:
        - np.ndarray: Padding mask with True in the padded region and False elsewhere.
        """
        edge_distance = np.ceil((edge_percentage / 100) * np.array(shape)).astype(int)
        mask = np.zeros(shape, dtype=bool)
        mask[:edge_distance[0], :] = True
        mask[-edge_distance[0]:, :] = True
        mask[:, :edge_distance[1]] = True
        mask[:, -edge_distance[1]:] = True
        return mask
    
    def expand_nan_zone(self, img, distance=10):
        nan_zone = np.isnan(img)
        for _ in range(distance):
            nan_zone = np.logical_or(nan_zone, np.roll(nan_zone, shift=1, axis=0))
            nan_zone = np.logical_or(nan_zone, np.roll(nan_zone, shift=-1, axis=0))
            nan_zone = np.logical_or(nan_zone, np.roll(nan_zone, shift=1, axis=1))
            nan_zone = np.logical_or(nan_zone, np.roll(nan_zone, shift=-1, axis=1))
        return nan_zone
    
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
    def evaluate_edge_peaks(self, peaks, edge_zone, img_xr, stricter_threshold, edge_percentage):
        """
        Evaluate peaks within the edge zone against the original img_xr.
        """
        # Convert edge_percentage to distance
        distance = int(np.ceil((edge_percentage / 100) * min(img_xr.shape)))  
        nan_adj_mask = self.expand_nan_zone(img_xr.fillna(0).values, distance=distance)
        
        peaks_in_edge_zone = np.where(edge_zone, peaks, np.nan)
        original_values = np.where(edge_zone, img_xr.values, np.nan)
        
        # Create a mask that includes the stricter threshold and adjacency to NaN values
        valid_mask = np.logical_and(original_values >= stricter_threshold, ~nan_adj_mask)
        
        # Update the peaks based on the valid mask
        valid_peaks = np.where(valid_mask, peaks_in_edge_zone, np.nan)
        return valid_peaks

    def identify_edge_zone(self, img, edge_percentage=5):
        """
        Identify a zone within `edge_percentage` from any NaN value or actual image edge.
        """
        img_shape = np.array(img.shape)
        edge_distance = np.ceil((edge_percentage / 100) * img_shape).astype(int)
        
        edge_mask = np.isnan(img)
        edge_zone = np.zeros(img.shape, dtype=bool)
        
        for axis, dist in enumerate(edge_distance):
            for shift in range(1, dist+1):
                edge_zone = np.logical_or(edge_zone, np.roll(edge_mask, shift=shift, axis=axis))
                edge_zone = np.logical_or(edge_zone, np.roll(edge_mask, shift=-shift, axis=axis))
        
        return edge_zone

## -- METHODS FOR DISPLAYING OUTPUTS -- ##
    #  Display Image Output (w/ peaks & DoG): Modified version of the display_image method to overlay scatter points for peak locations
    def display_image_with_peaks_and_DoG(self, dataset, title='Image with Peaks', cmap='turbo'):
        plt.close('all')
        plt.figure(figsize=(15, 7))

        DoG = self.DoG
        extent = None

        img = dataset['intensity']
        img_values = img.values
        peaks = dataset['peak_positions']
        coords_names = list(img.coords.keys())
        
        if len(coords_names) == 2:
            extent = [
                img.coords[coords_names[1]].min(),
                img.coords[coords_names[1]].max(),
                img.coords[coords_names[0]].min(),
                img.coords[coords_names[0]].max()
            ]
            ylabel, xlabel = coords_names

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
            peak_coords = np.column_stack(np.where(peaks.values == 1))
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
    def display_image_with_peaks(self, dataset, title='Image with Peaks', cmap='turbo'):
        plt.close('all')
        plt.figure(figsize=(15, 7))

        DoG = dataset.attrs.get('DoG', None)  # Retrieve DoG from attrs if available

        img = dataset['intensity']
        img_values = img.values
        peaks = dataset['peak_positions']
        coords_names = list(img.coords.keys())

        extent = [
            img.coords[coords_names[1]].min(),
            img.coords[coords_names[1]].max(),
            img.coords[coords_names[0]].min(),
            img.coords[coords_names[0]].max()
        ]
        ylabel, xlabel = coords_names

        vmin = np.nanpercentile(img_values, 10)
        vmax = np.nanpercentile(img_values, 99)

        plt.subplot(1, 2, 1)
        plt.imshow(np.flipud(img_values),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                aspect='auto')
        plt.colorbar()
        plt.title(f"{title} - Original")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if peaks is not None:
            peak_coords = np.column_stack(np.where(peaks.values == 1))
            peak_x_values = peaks.coords[coords_names[1]].values[peak_coords[:, 1]]
            peak_y_values = peaks.coords[coords_names[0]].values[peak_coords[:, 0]]
            plt.scatter(peak_x_values, peak_y_values, c='red', marker='o')

        plt.subplot(1, 2, 2)
        if DoG is not None:
            plt.imshow(np.flipud(DoG),
                    cmap=cmap,
                    extent=extent,
                    aspect='auto')
            plt.title(f"{title} - DoG")
        else:
            plt.title(f"{title} - No DoG Data")
        plt.colorbar()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.tight_layout()
        plt.show()

## -- METHODS FOR SAVING DATA OUTPUTS -- ## 
    def save_to_zarr(self, data_xr, file_path):
        # Serialize complex attributes to JSON strings
        serializable_attrs = {}
        for k, v in data_xr.attrs.items():
            if isinstance(v, xr.DataArray):
                serializable_attrs[k] = v.to_dict()
            elif isinstance(v, np.ndarray):
                serializable_attrs[k] = v.tolist()  # Convert ndarray to list
            elif isinstance(v, (int, float, str, list, dict)):
                serializable_attrs[k] = v
            # Skip methods or other non-serializable attributes
            elif callable(v):
                continue

        # Convert to JSON strings
        serializable_attrs = {k: json.dumps(v) for k, v in serializable_attrs.items()}

        # Create a new xarray DataArray for storing class attributes
        class_attrs_xr = xr.DataArray([0])  # Dummy DataArray
        class_attrs_dict = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items() if not callable(v)}
        class_attrs_xr.attrs['class_attrs'] = json.dumps(class_attrs_dict)

        # Replace the original attributes with the serialized version
        data_xr.attrs = serializable_attrs

        # Save to Zarr
        ds = xr.Dataset({'intensity': data_xr, 'class_attrs': class_attrs_xr})
        ds.to_zarr(file_path, mode='w')

    def save_to_netcdf(self, file_path, default_filename="output.nc"):
        """
        Save the DataSet to an HDF5 NETCDF file.

        Parameters:
        - file_path (str or pathlib.Path): The path where the HDF5 NETCDF file will be saved.
        """
        if isinstance(file_path, (str, Path)):
            path = Path(file_path)
            if path.is_dir():
                path = path / default_filename
            self.dataset.to_netcdf(path, engine='h5netcdf')
        else:
            raise ValueError("Invalid file path. Must be a string or a pathlib.Path object.")

    def load_from_netcdf(self, file_path):
        """
        Load the DataSet from an HDF5 NETCDF file.

        Parameters:
        - file_path (str or pathlib.Path): The path to the HDF5 NETCDF file to be loaded.

        Returns:
        - ds (xarray.Dataset): The loaded DataSet.
        """
        if isinstance(file_path, (str, Path)):
            self.dataset = xr.open_dataset(file_path, engine='h5netcdf')
        else:
            raise ValueError("Invalid file path. Must be a string or a pathlib.Path object.")

class SensitivityAnalysis:
    def __init__(self, simulate_function, df_columns: List[str]):
        self.simulate_function = simulate_function
        self.df_columns = df_columns
        self.results_df = pd.DataFrame(columns=df_columns)

        # Define constant and range parameters
        self.const_params = {
            'sigma1': 1, 'sigma2': 2, 'threshold': 0.006, 'clustering_method': 'DBSCAN',
            'eps': 1, 'min_samples': 2, 'k': 3, 'radius': 5, 'edge_percentage': 2, 'stricter_threshold': 5
        }
        self.param_ranges = {
            'sigma1': np.linspace(0.5, 1.5, 11), 'sigma2': np.linspace(1.5, 2.5, 11),
            'threshold': np.linspace(0.004, 0.008, 11), 'eps': np.linspace(0.5, 1.5, 11),
            'min_samples': np.arange(1, 6, 1), 'k': np.arange(2, 6, 1),
            'radius': np.linspace(3, 7, 9), 'edge_percentage': np.linspace(1, 5, 9),
            'stricter_threshold': np.linspace(3, 7, 9)
        }

    def run_individual_sensitivity_analysis(self):
        for param, values in self.param_ranges.items():
            num_peaks = []
            for value in values:
                test_params = self.const_params.copy()
                test_params[param] = value
                num_detected_peaks = self.simulate_function(**test_params)
                num_peaks.append(num_detected_peaks)
            new_df = pd.DataFrame({param: values, 'num_peaks': num_peaks})
            self.results_df = pd.concat([self.results_df, new_df], ignore_index=True)

    def plot_results(self):
        for param in self.param_ranges.keys():
            subset = self.results_df[[param, 'num_peaks']].dropna()
            plt.figure()
            plt.plot(subset[param], subset['num_peaks'], marker='o')
            plt.title(f"Sensitivity Analysis for {param}")
            plt.xlabel(param)
            plt.ylabel('Number of Detected Peaks')
            plt.grid(True)
            plt.show()

    def calculate_covariance(self):
        df = self.results_df.apply(pd.to_numeric, errors='ignore')
        cov_matrix = df.cov()
        cov_with_num_peaks = cov_matrix.loc['num_peaks'].drop('num_peaks')
        sorted_cov = cov_with_num_peaks.sort_values(ascending=False)
        return sorted_cov
    
    def save_to_netcdf(self, file_path: str):
        self.results_df.to_xarray().to_netcdf(file_path)
    
    def load_from_netcdf(self, file_path: str) -> pd.DataFrame:
        """
        Load a DataFrame from a NETCDF file.
        
        Parameters:
        - file_path (str): The file path to load the DataFrame from.
        
        Returns:
        - pd.DataFrame: The loaded DataFrame.
        """
        dataset = xr.open_dataset(file_path)
        # Replace 'correct_variable_name' with the actual variable name you're interested in
        self.results_df = dataset['correct_variable_name'].to_dataframe()
        return self.results_df

    def calculate_covariance(self) -> pd.DataFrame:
        """
        Calculate the covariance and correlation of each parameter with the output ('num_peaks').
        
        Returns:
        - pd.DataFrame: A DataFrame containing the covariance and correlation of each parameter with 'num_peaks'.
        """
        # Convert columns to appropriate data types
        df = self.results_df.apply(pd.to_numeric, errors='ignore')
        
        # Compute the covariance matrix
        cov_matrix = df.cov()
        
        # Compute the correlation matrix
        corr_matrix = df.corr()
        
        # Extract the 'num_peaks' row to find the covariance and correlation between 'num_peaks' and each parameter
        cov_with_num_peaks = cov_matrix.loc['num_peaks'].drop('num_peaks')
        corr_with_num_peaks = corr_matrix.loc['num_peaks'].drop('num_peaks')
        
        # Combine into a single DataFrame
        summary_df = pd.DataFrame({
            'Covariance': cov_with_num_peaks,
            'Correlation': corr_with_num_peaks
        })
        
        # Sort by absolute value of correlation for easier interpretation
        summary_df['Abs_Correlation'] = summary_df['Correlation'].abs()
        summary_df = summary_df.sort_values('Abs_Correlation', ascending=False).drop('Abs_Correlation', axis=1)
        
        return summary_df

'''
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
'''

'''
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