import sys, re, os, time, psutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
import plotly.graph_objs as go
import ipywidgets as widgets
from IPython.display import display, clear_output

class WAXSComputeCrystal:
    def __init__(self, lattice_params):
        self.a, self.b, self.c, self.alpha, self.beta, self.gamma = lattice_params

        # - Compute Real Space Cartesian Vectors
        self.a1, self.a2, self.a3 = self.compute_realspacevecs()
        self.a1, self.a2, self.a3 = self.clean_data([self.a1, self.a2, self.a3])

        # - Compute the Volume of a Unit Cell
        self.volume = self.compute_volume()

        # - Compute Reciprocal Space Vectors
        self.b1, self.b2, self.b3 = self.compute_recipvecs()
        self.b1, self.b2, self.b3 = self.clean_data([self.b1, self.b2, self.b3])

    def initialize(self, Mhkl):
        self.Mhkl = Mhkl

        # - Computed (hkl) Coordinate Positions (Lattice Points)
        self.hkl_coords = self.compute_coords()

    # - Compute the real-space orthogonal basis vectors.
    def compute_realspacevecs(self):
        alpha = np.radians(self.alpha)
        beta = np.radians(self.beta)
        gamma = np.radians(self.gamma)

        a1 = np.array([self.a, 0, 0])
        a2 = np.array([self.b * np.cos(gamma), self.b * np.sin(gamma), 0])
        z_value = 1 - np.cos(beta)**2 - ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma))**2
        z_value = max(z_value, 0)
        a3 = np.array([self.c * np.cos(beta),
                       self.c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
                       self.c * np.sqrt(z_value)])

        return a1, a2, a3

    def compute_volume(self):
        return np.dot(self.a1, np.cross(self.a2, self.a3))

    def compute_recipvecs(self):
        b1 = 2 * np.pi * np.cross(self.a2, self.a3) / self.volume
        b2 = 2 * np.pi * np.cross(self.a3, self.a1) / self.volume
        b3 = 2 * np.pi * np.cross(self.a1, self.a2) / self.volume

        return b1, b2, b3

    def compute_coords(self):
        Mhkl = int(self.Mhkl)
        hkl_coords = []
        hkl_indices = []

        for h in range(Mhkl):
            for k in range(Mhkl):
                for l in range(Mhkl):
                    qx = h * self.b1[0] + k * self.b2[0] + l * self.b3[0]
                    qy = h * self.b1[1] + k * self.b2[1] + l * self.b3[1]
                    qz = h * self.b1[2] + k * self.b2[2] + l * self.b3[2]
                    hkl_coords.append((qx, qy, qz))  # Save only the (qx, qy, qz) coordinates
                    hkl_indices.append((h, k, l))  # Save the hkl indices separately

        self.hkl_indices = np.array(hkl_indices)  # Store the hkl indices as a new attribute
        return np.array(hkl_coords)

    def clean_data(self, data):
        return [np.where(np.isclose(vec, 0, atol=1e-10), 0, vec) for vec in data]

class GenCrystal(WAXSComputeCrystal):
    def __init__(self, lattice_params, Mhkl, sigma_r, sigma_theta, sigma_phi, m):
        super().__init__(lattice_params)
        self.initialize(Mhkl)
        self.sigma_r = sigma_r
        self.sigma_theta = sigma_theta
        self.sigma_phi = sigma_phi
        self.m = m
        self.lattice_params = lattice_params

class RandomCrystal(WAXSComputeCrystal):
    def __init__(self, lattice_params, Mhkl, sigma_r, sigma_theta, sigma_phi, m):
        self.crystal_systems = {
            'triclinic': self.triclinic,
            'monoclinic': self.monoclinic,
            'orthorhombic': self.orthorhombic,
            'tetragonal': self.tetragonal,
            'trigonal': self.trigonal,
            'hexagonal': self.hexagonal,
            'cubic': self.cubic
        }

        # Select a random crystal system
        system = np.random.choice(list(self.crystal_systems.keys()))

        # Generate lattice parameters for the selected system
        lattice_params = self.crystal_systems[system]()
        
        print(f"Selected Crystal System: {system}")

        super().__init__(lattice_params)
        self.initialize(Mhkl)
        self.sigma_r = sigma_r
        self.sigma_theta = sigma_theta
        self.sigma_phi = sigma_phi
        self.m = m
        self.lattice_params = lattice_params

    @staticmethod
    def triclinic():
        a, b, c = np.round(np.random.uniform(1, 10, 3), 2)
        alpha, beta, gamma = np.round(np.random.uniform(20, 160, 3), 2)
        return a, b, c, alpha, beta, gamma

    @staticmethod
    def monoclinic():
        a, b, c = np.round(np.random.uniform(1, 10, 3), 2)
        alpha, gamma = 90, 90
        beta = np.round(np.random.uniform(20, 160), 2)
        return a, b, c, alpha, beta, gamma

    @staticmethod
    def orthorhombic():
        a, b, c = np.round(np.random.uniform(1, 10, 3), 2)
        alpha, beta, gamma = 90, 90, 90
        return a, b, c, alpha, beta, gamma

    @staticmethod
    def tetragonal():
        a = np.round(np.random.uniform(1, 10), 2)
        c = np.round(np.random.uniform(1, 10), 2)
        alpha, beta, gamma = 90, 90, 90
        return a, a, c, alpha, beta, gamma

    @staticmethod
    def trigonal():
        a = np.round(np.random.uniform(1, 10), 2)
        alpha = np.round(np.random.uniform(20, 140), 2)
        return a, a, a, alpha, alpha, alpha

    @staticmethod
    def hexagonal():
        a = np.round(np.random.uniform(1, 10), 2)
        c = np.round(np.random.uniform(1, 10), 2)
        return a, a, c, 90, 90, 120

    @staticmethod
    def cubic():
        a = np.round(np.random.uniform(1, 10), 2)
        return a, a, a, 90, 90, 90

class PlotCrystal():
    def __init__(self, crystal, m):
        self.crystal = crystal
        self.m = m
        # self.crystal.generate_hkl_coords()  # generate hkl_coords
        self.create_pixel_space()

    # - Estimate the time order complexity for smearing a crystal structure.
    def estimate_time_complexity(self):
        """Estimate the time complexity based on system specs and current usage."""

        # Get the number of CPUs available
        num_cpus = os.cpu_count()

        # Get the CPU usage
        cpu_usage = psutil.cpu_percent()

        # Get the available memory
        avail_mem = psutil.virtual_memory().available

        # Estimate the time complexity based on the number of CPUs, CPU usage, and available memory
        # This is a very rough estimate and is unlikely to be accurate
        time_complexity = (self.Mhkl**3 * self.m**3) / (num_cpus * (1 - cpu_usage/100) * avail_mem)

        return time_complexity
    
    # - Generate Ewald Sphere Pixel Space
    def create_pixel_space(self):
        # Create the 3D pixel space with size m x m x m
        self.pixel_space = np.zeros((self.m, self.m, self.m))

    # - Convert Cartesian (qx, qy, qz) Coordinates to Spherical Coordinates
    def cart_to_sph(self, qx, qy, qz):
        qr = np.sqrt(qx**2 + qy**2 + qz**2)
        qtheta = np.arctan2(np.sqrt(qx**2 + qy**2), qz)
        qphi = np.arctan2(qy, qx)
        return qr, qtheta, qphi

    def plot_panel(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        qx_values = self.hkl_coords[:, 0]
        qy_values = self.hkl_coords[:, 1]
        qz_values = self.hkl_coords[:, 2]

        axs[0].scatter(qx_values, qz_values, color='blue')
        axs[1].scatter(qy_values, qz_values, color='green')
        axs[2].scatter(np.sqrt(qx_values**2 + qy_values**2), qz_values, color='red')

        axs[0].grid()
        axs[1].grid()
        axs[2].grid()

        axs[0].set_xlabel(r'$q_x \, (\AA^{-1})$')
        axs[0].set_ylabel(r'$q_z \, (\AA^{-1})$')
        axs[1].set_xlabel(r'$q_y \, (\AA^{-1})$')
        axs[1].set_ylabel(r'$q_z \, (\AA^{-1})$')
        axs[2].set_xlabel(r'$q_{xy} \, (\AA^{-1})$')
        axs[2].set_ylabel(r'$q_z \, (\AA^{-1})$')

        plt.tight_layout()
        plt.show()

    def plot_qx_qz(self):
        plt.figure()
        qx_values = self.hkl_coords[:, 0]
        qz_values = self.hkl_coords[:, 2]
        plt.scatter(qx_values, qz_values, color='blue')
        plt.grid()
        plt.xlabel(r'$q_x \, (\AA^{-1})$')
        plt.ylabel(r'$q_z \, (\AA^{-1})$')
        plt.show()

    def plot_qy_qz(self):
        plt.figure()
        qy_values = self.hkl_coords[:, 1]
        qz_values = self.hkl_coords[:, 2]
        plt.scatter(qy_values, qz_values, color='green')
        plt.grid()
        plt.xlabel(r'$q_y \, (\AA^{-1})$')
        plt.ylabel(r'$q_z \, (\AA^{-1})$')
        plt.show()

    def plot_qxy_qz(self):
        plt.figure()
        qx_values = self.hkl_coords[:, 0]
        qy_values = self.hkl_coords[:, 1]
        qz_values = self.hkl_coords[:, 2]
        plt.scatter(np.sqrt(qx_values**2 + qy_values**2), qz_values, color='red')
        plt.grid()
        plt.xlabel(r'$q_{xy} \, (\AA^{-1})$')
        plt.ylabel(r'$q_z \, (\AA^{-1})$')
        plt.show()

    def gen_gaussian(self, q_vec):
        # Define the variance for each direction in 3D
        variance = [self.sigma_r**2, self.sigma_theta**2, self.sigma_phi**2]
        
        # Convert Cartesian coordinates to spherical
        qr, qtheta, qphi = self.cart_to_sph(*q_vec)
        
        # Create a grid of coordinates spanning the pixel space
        grid = np.indices((self.m, self.m, self.m)).reshape(3,-1).T

        # Generate a Gaussian centered at each peak
        gauss = multivariate_normal.pdf(grid, mean=[qr, qtheta, qphi], cov=variance)

        return gauss.reshape((self.m, self.m, self.m))

    # - Gaussian Convolution Methods + Plotting
    def gaussian(self, x, mu, sigma):
        return norm.pdf(x, mu, sigma)

    def convoluted_gaussian(self, q_vec):
        qx, qy, qz = q_vec
        qr, qtheta, qphi = self.cart_to_sph(qx, qy, qz)
        gauss_r = self.gaussian(qr, qr, self.sigma_r)
        gauss_theta = self.gaussian(qtheta, qtheta, self.sigma_theta)
        gauss_phi = self.gaussian(qphi, qphi, self.sigma_phi)
        return gauss_r * gauss_theta * gauss_phi
    
    def smear_peaks(self):
        for hkl, q_vec in self.hkl_coords:
            gauss = self.gen_gaussian(q_vec)
            self.pixel_space += gauss

    def plot_qx_qz_convolution(self):
        plt.figure()
        for hkl, q_vec in self.hkl_coords:
            qx, qy, qz = q_vec
            intensity = self.convoluted_gaussian(q_vec)
            plt.scatter(qx, qz, c=intensity)
        plt.grid()
        plt.xlabel(r'$q_x \, (\AA^{-1})$')
        plt.ylabel(r'$q_z \, (\AA^{-1})$')
        plt.show()

    def plot_qy_qz_convolution(self):
        plt.figure()
        for hkl, q_vec in self.hkl_coords:
            qx, qy, qz = q_vec
            intensity = self.convoluted_gaussian(q_vec)
            plt.scatter(qy, qz, c=intensity)
        plt.grid()
        plt.xlabel(r'$q_y \, (\AA^{-1})$')
        plt.ylabel(r'$q_z \, (\AA^{-1})$')
        plt.show()

    def plot_qxy_qz_convolution(self):
        plt.figure()
        for hkl, q_vec in self.hkl_coords:
            qx, qy, qz = q_vec
            intensity = self.convoluted_gaussian(q_vec)
            plt.scatter(np.sqrt(qx**2 + qy**2), qz, c=intensity)
        plt.grid()
        plt.xlabel(r'$q_{xy} \, (\AA^{-1})$')
        plt.ylabel(r'$q_z \, (\AA^{-1})$')
        plt.show()

    def plot_panel_convolution(self):
        plt.figure(figsize=(15,5))
        
        plt.subplot(131)
        for hkl, q_vec in self.hkl_coords:
            qx, qy, qz = q_vec
            intensity = self.convoluted_gaussian(q_vec)
            plt.scatter(qx, qz, c=intensity)
        plt.grid()
        plt.xlabel(r'$q_x \, (\AA^{-1})$')
        plt.ylabel(r'$q_z \, (\AA^{-1})$')
        
        plt.subplot(132)
        for hkl, q_vec in self.hkl_coords:
            qx, qy, qz = q_vec
            intensity = self.convoluted_gaussian(q_vec)
            plt.scatter(qy, qz, c=intensity)
        plt.grid()
        plt.xlabel(r'$q_y \, (\AA^{-1})$')
        plt.ylabel(r'$q_z \, (\AA^{-1})$')

        plt.subplot(133)
        for hkl, q_vec in self.hkl_coords:
            qx, qy, qz = q_vec
            intensity = self.convoluted_gaussian(q_vec)
            plt.scatter(np.sqrt(qx**2 + qy**2), qz, c=intensity)
        plt.grid()
        plt.xlabel(r'$q_{xy} \, (\AA^{-1})$')
        plt.ylabel(r'$q_z \, (\AA^{-1})$')

        plt.tight_layout()
        plt.show()

    def plot_3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for hkl, q_vec in self.hkl_coords:
            qx, qy, qz = q_vec
            intensity = self.convoluted_gaussian(q_vec)
            ax.scatter(qx, qy, qz, c=intensity)

        ax.set_xlabel('Qx')
        ax.set_ylabel('Qy')
        ax.set_zlabel('Qz')
        plt.show()
    
    def plot_image(self, sigma):
        # Estimate the time complexity
        est_time_complexity = self.estimate_time_complexity()
        print(f"Estimated time complexity: {est_time_complexity}")

        # Ask the user to proceed
        proceed = input("Proceed ([y]/n)? ")
        if proceed.lower() != 'n':
            start_time = time.time()

            # Iterate through the points in the Cartesian coordinate system
            for hkl, q_vec in self.hkl_coords:
                gauss = self.gen_gaussian(q_vec)
                self.pixel_space += gauss

            # Define the threshold for plotting
            threshold = 0.5

            # Get the indices where the Gaussian is above the threshold
            ind = np.argwhere(self.pixel_space > threshold)

            # Plot the 3D pixel space
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(ind[:,0], ind[:,1], ind[:,2], c='r', marker='o')
            plt.show()
        
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Total computation time: {total_time} seconds")

    def plot_pixel_space(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        pixel_space_reshaped = self.pixel_space.reshape(self.m**3)
        x, y, z = np.indices((self.m, self.m, self.m)).reshape(3, -1)
        c = pixel_space_reshaped
        img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
        fig.colorbar(img)
        plt.show()
    
    def create_image_plane(self):
        self.image_plane = ImagePlane()  # instantiate the ImagePlane class
        try:
            self.image_plane.load_data(self.hkl_coords)  # load the data using the ImagePlane's load_data method
        except AttributeError:
            print("hkl_coords attribute not found. Make sure it is defined in child classes.")

class PlotGenCrystal(PlotCrystal):
    def __init__(self, lattice_params, Mhkl, sigma_r, sigma_theta, sigma_phi, m):
        self.crystal = GenCrystal(lattice_params, Mhkl, sigma_r, sigma_theta, sigma_phi, m)
        self.lattice_params = self.crystal.lattice_params
        self.m = m
        self.create_pixel_space()

        # Access attributes from GenCrystal (and thereby ComputeCrystal)
        self.a = self.crystal.a
        self.b = self.crystal.b
        self.c = self.crystal.c
        self.alpha = self.crystal.alpha
        self.beta = self.crystal.beta
        self.gamma = self.crystal.gamma

        self.a1 = self.crystal.a1
        self.a2 = self.crystal.a2
        self.a3 = self.crystal.a3

        self.volume = self.crystal.volume

        self.b1 = self.crystal.b1
        self.b2 = self.crystal.b2
        self.b3 = self.crystal.b3

        self.Mhkl = self.crystal.Mhkl

        self.hkl_coords = self.crystal.hkl_coords.copy()  # Assuming hkl_coords is a numpy array or a list
        self.hkl_indices = self.crystal.hkl_indices.copy()  # Access the hkl_indices from the GenCrystal object

    def create_image_plane(self):
        # Initialize ImagePlane with initial points
        self.image_plane = ImagePlane(np.array([]))
        # Load data
        self.image_plane.load_data(self.hkl_coords, self.a1, self.a2, self.a3)
        
class PlotRandCrystal(PlotCrystal):
    def __init__(self, lattice_params, Mhkl, sigma_r, sigma_theta, sigma_phi, m):
        self.lattice_params = None
        self.crystal = RandomCrystal(lattice_params, Mhkl, sigma_r, sigma_theta, sigma_phi, m)
        self.lattice_params = self.crystal.lattice_params

        # Access attributes from GenCrystal (and thereby ComputeCrystal)
        self.a = self.crystal.a
        self.b = self.crystal.b
        self.c = self.crystal.c
        self.alpha = self.crystal.alpha
        self.beta = self.crystal.beta
        self.gamma = self.crystal.gamma

        self.a1 = self.crystal.a1
        self.a2 = self.crystal.a2
        self.a3 = self.crystal.a3

        self.volume = self.crystal.volume

        self.b1 = self.crystal.b1
        self.b2 = self.crystal.b2
        self.b3 = self.crystal.b3

        self.Mhkl = self.crystal.Mhkl

        self.hkl_coords = self.crystal.hkl_coords.copy()  # Assuming hkl_coords is a numpy array or a list

class ImagePlane:
    def __init__(self, hkl_coords=None):
        self.points = hkl_coords
        self.fig_3d = None
        self.fig_2d = None
        self.scatter = None
        self.plane = None
        self.planes = []  # Store references to generated planes
        self.slider_a = None
        self.slider_b = None
        self.slider_c = None
        self.slider_d = None
        self.slider_thickness = None
        self.btn_level_parallel = None
        self.btn_level_perpendicular = None

    def update_plane(self, a, b, c, d):
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        X, Y = np.meshgrid(x, y)
        Z = (-a * X - b * Y - d) / c

        return go.Surface(x=X, y=Y, z=Z, opacity=0.5, showscale=False)

    def add_plane(self, a, b, c, d, offset):
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        X, Y = np.meshgrid(x, y)
        Z = (-a * X - b * Y - (d + offset)) / c

        plane = go.Surface(x=X, y=Y, z=Z, opacity=0.5, showscale=False)
        self.fig_3d.add_trace(plane)
        self.planes.append(plane)  # Store reference to the generated plane

    def delete_plane(self):
        plane = self.planes.pop()  # Remove reference to the last generated plane
        self.fig_3d.data.remove(plane)

    def update(self, change):
        a = self.slider_a.value
        b = self.slider_b.value
        c = self.slider_c.value
        d = self.slider_d.value
        plane_thickness = self.slider_thickness.value
        new_plane = self.update_plane(a, b, c, d)
        self.fig_3d.data[1].x = new_plane.x
        self.fig_3d.data[1].y = new_plane.y
        self.fig_3d.data[1].z = new_plane.z

        # Calculate intersection points
        if np.isclose(a, 0) and np.isclose(c, 0):  # This means the plane is parallel to the XZ plane
            intersection_points = self.points[np.abs(self.points[:, 1] - (-d/b)) <= plane_thickness]
        else:
            plane_distance = np.abs(a * self.points[:, 0] + b * self.points[:, 1] + c * self.points[:, 2] + d) / np.sqrt(
                a ** 2 + b ** 2 + c ** 2)
            intersection_points = self.points[plane_distance <= plane_thickness]

        # Calculate the thickness threshold from the origin plane
        origin_plane_distance = np.abs(a * self.points[:, 0] + b * self.points[:, 1] + c * self.points[:, 2] + d) / np.sqrt(
            a ** 2 + b ** 2 + c ** 2)
        origin_plane_thickness = np.max(origin_plane_distance)

        # Update the scatter plot with intersection points
        self.fig_2d.data[0].x = intersection_points[:, 0]
        self.fig_2d.data[0].y = intersection_points[:, 1]

        # Generate or delete visual plane objects based on the thickness
        existing_planes = len(self.planes)
        thickness_range = self.slider_thickness.max - self.slider_thickness.min
        thickness_increment = thickness_range * 0.1  # Generate new plane objects every 10% of the thickness slider range

        target_planes = min(
            existing_planes, int(np.ceil((origin_plane_thickness - self.slider_thickness.min) / thickness_increment))
        )

        if target_planes > existing_planes:  # Generate additional plane objects
            offset_factor = (self.slider_thickness.value - self.slider_thickness.min) / thickness_range
            for _ in range(target_planes - existing_planes):
                offset = offset_factor * origin_plane_thickness
                self.add_plane(a, b, c, d, offset)

        elif target_planes < existing_planes:  # Delete excess plane objects
            for _ in range(existing_planes - target_planes):
                self.delete_plane()

    def level_parallel(self, _):
        # Level the plane parallel to the xy-plane about its center
        self.slider_a.value = 0
        self.slider_b.value = 0
        self.slider_c.value = 1
        self.slider_d.value = -np.mean(self.points[:, 2])

    def level_perpendicular(self, _):
        # Level the plane parallel to the xz-plane about its center
        self.slider_a.value = 0
        self.slider_b.value = 1
        self.slider_c.value = 0
        self.slider_d.value = -np.mean(self.points[:, 1])

    def load_data(self, hkl_coords, a1, a2, a3):
        hkl_coords = np.asarray(hkl_coords)
        a1 = np.asarray(a1)
        a2 = np.asarray(a2)
        a3 = np.asarray(a3)

        # Check if the shapes are compatible
        if hkl_coords.shape[1] != 3 or a1.shape != (3,) or a2.shape != (3,) or a3.shape != (3,):
            raise ValueError("Invalid shapes of input arrays")

        # Convert hkl coordinates to Cartesian coordinates
        self.points = hkl_coords[:, 0][:, np.newaxis] * a1 + hkl_coords[:, 1][:, np.newaxis] * a2 + hkl_coords[:, 2][:, np.newaxis] * a3

        # Rescale the plane to match the data range
        self.scale_plane()

        # Calculate the range of the data
        data_range = np.max(self.points, axis=0) - np.min(self.points, axis=0)

        # Create a 3D scatter plot
        self.scatter = go.Scatter3d(x=self.points[:, 0], y=self.points[:, 1], z=self.points[:, 2], mode='markers',
                                    marker=dict(size=5))

        # Initial plane parameters
        a, b, c, d = 1, -1, 1, 0

        self.plane = self.update_plane(a, b, c, d)

        # Create the initial 3D plot
        self.fig_3d = go.FigureWidget(data=[self.scatter, self.plane])
        self.fig_3d.layout.title = "3D Plot with Plane"
        self.fig_3d.layout.width = 800
        self.fig_3d.layout.height = 600

        # Create the 2D scatter plot for intersection points
        self.fig_2d = go.FigureWidget(data=[go.Scatter(x=[], y=[], mode='markers')])
        self.fig_2d.layout.title = "Intersection Points"
        self.fig_2d.layout.xaxis.title = 'X'
        self.fig_2d.layout.yaxis.title = 'Y'

        # Create sliders for plane parameters
        self.slider_a = widgets.FloatSlider(min=-1, max=1, step=0.01, value=a, description='a (X coefficient)')
        self.slider_b = widgets.FloatSlider(min=-1, max=1, step=0.01, value=b, description='b (Y coefficient)')
        self.slider_c = widgets.FloatSlider(min=-1, max=1, step=0.01, value=c, description='c (Z coefficient)')
        self.slider_d = widgets.FloatSlider(min=-1, max=1, step=0.01, value=d, description='d (Constant)')

        # Update the thickness slider range
        self.slider_thickness = widgets.FloatSlider(min=0.01, max=np.max(data_range), step=0.01, value=0.05, description='Thickness')

        # Create spring-action buttons for leveling the plane
        self.btn_level_parallel = widgets.Button(description="Level Parallel to XY-Plane")
        self.btn_level_perpendicular = widgets.Button(description="Level Perpendicular to XY-Plane")

        # Add event handlers to the buttons
        self.btn_level_parallel.on_click(self.level_parallel)
        self.btn_level_perpendicular.on_click(self.level_perpendicular)

        # Add the observer to the sliders
        self.slider_a.observe(self.update, names='value')
        self.slider_b.observe(self.update, names='value')
        self.slider_c.observe(self.update, names='value')
        self.slider_d.observe(self.update, names='value')
        self.slider_thickness.observe(self.update, names='value')

        # Display the interactive plot and sliders
        display(widgets.HBox([self.fig_3d, self.fig_2d]))
        display(widgets.VBox([widgets.Label('Plane Parameters:'), self.slider_a, self.slider_b, self.slider_c, self.slider_d,
                              self.slider_thickness, self.btn_level_parallel, self.btn_level_perpendicular]))

    def scale_plane(self):
        min_vals = np.min(self.points, axis=0)
        max_vals = np.max(self.points, axis=0)
        self.points = (self.points - min_vals) / (max_vals - min_vals)

    def level_parallel(self, _):
        # Level the plane parallel to the xy-plane about its center
        self.slider_a.value = 0
        self.slider_b.value = 0
        self.slider_c.value = 1
        self.slider_d.value = -np.mean(self.points[:, 2])

    def level_perpendicular(self, _):
        # Level the plane parallel to the xz-plane about its center
        self.slider_a.value = 0
        self.slider_b.value = 1
        self.slider_c.value = 0
        self.slider_d.value = -np.mean(self.points[:, 1])
        