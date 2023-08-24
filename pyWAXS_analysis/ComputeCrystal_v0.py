import sys, re, os, time, psutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist

class ComputeCrystal:
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

        for h in range(Mhkl):
            for k in range(Mhkl):
                for l in range(Mhkl):
                    qx = h * self.b1[0] + k * self.b2[0] + l * self.b3[0]
                    qy = h * self.b1[1] + k * self.b2[1] + l * self.b3[1]
                    qz = h * self.b1[2] + k * self.b2[2] + l * self.b3[2]
                    hkl_coords.append(((h, k, l), (qx, qy, qz)))

        return np.array(hkl_coords)

    def clean_data(self, data):
        return [np.where(np.isclose(vec, 0, atol=1e-10), 0, vec) for vec in data]

class GenCrystal(ComputeCrystal):
    def __init__(self, lattice_params, Mhkl, sigma_r, sigma_theta, sigma_phi, m):
        super().__init__(lattice_params)
        self.initialize(Mhkl)
        self.sigma_r = sigma_r
        self.sigma_theta = sigma_theta
        self.sigma_phi = sigma_phi
        self.m = m
        self.lattice_params = lattice_params

class RandomCrystal(ComputeCrystal):
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

    # - Plot Generated Points in a 3-Panel
    def plot_panel(self):
        plt.figure(figsize=(15,5))

        plt.subplot(131)
        for hkl, q_vec in self.hkl_coords:
            qx, qy, qz = q_vec
            plt.scatter(qx, qz, color = 'blue')
        plt.grid()
        plt.xlabel(r'$q_x \, (\AA^{-1})$')
        plt.ylabel(r'$q_z \, (\AA^{-1})$')

        plt.subplot(132)
        for hkl, q_vec in self.hkl_coords:
            qx, qy, qz = q_vec
            plt.scatter(qy, qz, color = 'green')
        plt.grid()
        plt.xlabel(r'$q_y \, (\AA^{-1})$')
        plt.ylabel(r'$q_z \, (\AA^{-1})$')

        plt.subplot(133)
        for hkl, q_vec in self.hkl_coords:
            qx, qy, qz = q_vec
            plt.scatter(np.sqrt(qx**2 + qy**2), qz, color = 'red')
        plt.grid()
        plt.xlabel(r'$q_{xy} \, (\AA^{-1})$')
        plt.ylabel(r'$q_z \, (\AA^{-1})$')

        plt.tight_layout()
        plt.show()

    def plot_qx_qz(self):
        plt.figure()
        for hkl, q_vec in self.hkl_coords:
            qx, qy, qz = q_vec
            plt.scatter(qx, qz, color='blue')
        plt.grid()
        plt.xlabel(r'$q_x \, (\AA^{-1})$')
        plt.ylabel(r'$q_z \, (\AA^{-1})$')
        plt.show()

    def plot_qy_qz(self):
        plt.figure()
        for hkl, q_vec in self.hkl_coords:
            qx, qy, qz = q_vec
            plt.scatter(qy, qz, color = 'green')
        plt.grid()
        plt.xlabel(r'$q_y \, (\AA^{-1})$')
        plt.ylabel(r'$q_z \, (\AA^{-1})$')
        plt.show()

    def plot_qxy_qz(self):
        plt.figure()
        for hkl, q_vec in self.hkl_coords:
            qx, qy, qz = q_vec
            plt.scatter(np.sqrt(qx**2 + qy**2), qz, color = 'red')
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

class PlotGenCrystal(PlotCrystal):
    def __init__(self, lattice_params, Mhkl, sigma_r, sigma_theta, sigma_phi, m):
        self.crystal = GenCrystal(lattice_params, Mhkl, sigma_r, sigma_theta, sigma_phi, m)
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
    def __init__(self, plot_crystal, plane_normal=np.array([0,0,1]), plane_point=np.array([0,0,0]), plane_thickness=0.05):
        self.plot_crystal = plot_crystal
        self.plane_normal = plane_normal
        self.plane_point = plane_point
        self.plane_thickness = plane_thickness
        self.plane_d = -plane_point.dot(plane_normal)

    def get_intersection_points(self):
        """
        Get the points in the plot crystal that intersect with the plane
        """

        # Calculate the distance of each point in the pixel space from the plane
        distances = np.abs((self.plane_normal[0] * self.plot_crystal.pixel_space[:,:,0] + 
                            self.plane_normal[1] * self.plot_crystal.pixel_space[:,:,1] + 
                            self.plane_normal[2] * self.plot_crystal.pixel_space[:,:,2] + 
                            self.plane_d) / 
                           np.linalg.norm(self.plane_normal))

        # Get the points in the pixel space where the distance is less than the plane thickness
        intersection_points = np.where(distances <= self.plane_thickness)

        return intersection_points

    def plot_intersection(self):
        """
        Plot the intersection of the plane and the plot crystal
        """

        intersection_points = self.get_intersection_points()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(intersection_points[0], intersection_points[1], intersection_points[2], c='r', marker='o')
        plt.show()