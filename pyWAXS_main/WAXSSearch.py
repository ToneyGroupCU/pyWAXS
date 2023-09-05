import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

class WAXSSearch:
    def __init__(self, data, coords=None):
        if isinstance(data, xr.DataArray):
            self.data = data
        elif isinstance(data, np.ndarray):
            if coords is None:
                raise ValueError("Coordinates must be provided for numpy arrays.")
            self.data = xr.DataArray(data, coords=coords)
        else:
            raise ValueError("Data must be an xarray DataArray or numpy ndarray.")
        
        self.layer_grouping = None  # Initialize to None
        self.layer_plot_data = None  # Initialize to None
        self.layer_data = None  # Initialize to None
        self.bin_centers = None

    # Method 1: Assess variance and bin layers
    def assess_variance(self, num_bins=1000):
        # Initialize to None for safety
        self.layer_grouping = None
        self.bin_centers = None

        # Step 1: Mask out invalid pixels
        if self.data is None:
            raise ValueError("Data is not available.")
        
        valid_data = self.data.where(self.data > 0)

        if valid_data.count() == 0:
            raise ValueError("No valid data points found.")
        
        # Step 2: Binning pixels based on intensity
        min_intensity = float(valid_data.min())
        max_intensity = float(valid_data.max())

        if np.isnan(min_intensity) or np.isnan(max_intensity):
            raise ValueError("Minimum or Maximum intensity is NaN.")

        bin_edges = np.linspace(min_intensity, max_intensity, num_bins+1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Digitize the data into bins
        digitized_data = np.digitize(valid_data, bin_edges) - 1

        # Step 3: Assign Group Numbers
        group_numbers = xr.DataArray(digitized_data, coords=valid_data.coords, dims=valid_data.dims)
        
        # Step 4: Store the results
        self.layer_grouping = group_numbers
        self.bin_centers = bin_centers

        return self.layer_grouping

    def plot_layer_variance(self):
        # Step 1: Check if layer_grouping and bin_centers are available
        if self.layer_grouping is None or self.bin_centers is None:
            raise ValueError("Run the assess_variance method first to populate layer_grouping and bin_centers.")
        
        # Step 2: Count the number of pixels in each bin
        bin_counts = self.layer_grouping.groupby(self.layer_grouping).count()
        
        # Convert to numpy array for plotting
        bin_counts_values = bin_counts.values
        bin_centers_values = self.bin_centers

        # Handle case where some bins might be missing
        full_bin_counts = np.zeros_like(bin_centers_values)
        for i, count in enumerate(bin_counts_values):
            full_bin_counts[i] = count

        # Step 3: Plotting
        plt.figure(figsize=(10, 6))
        plt.bar(bin_centers_values, full_bin_counts, width=bin_centers_values[1] - bin_centers_values[0])
        plt.xlabel("Intensity")
        plt.ylabel("Counts")
        plt.title("Layer Variance")
        plt.show()

    # Method 2: Select layers based on the 1D distribution
    def select_layers(self):
        # Your code here
        pass

    # Method 3: Layer Continuity Check and Masking
    def check_layer_continuity(self):
        # Your code here
        pass

    # Method 4: Monte Carlo Point Seeding
    def monte_carlo_seeding(self):
        # Your code here
        pass

    # Method 5: Cost Function
    def cost_function(self):
        # Your code here
        pass

    # Method 6: Optimizers
    def optimizer(self):
        # Your code here
        pass


'''
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


# -- Alternative 2D Peak Finder
    def detect_2D_peaks(self, threshold=0.5):
        # Finding peaks in image intensity
        peaks = find_peaks(self.corrected_tiff, threshold)
        self.peak_positions_pixel = np.array(peaks[0])

        # Getting coordinates with respect to the image
        self.peak_positions_coords = [self.pixel_to_coords(pixel) for pixel in self.peak_positions_pixel]
'''