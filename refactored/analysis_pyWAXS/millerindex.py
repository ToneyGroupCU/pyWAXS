import math
import itertools
import numpy as np
import pandas as pd
import hvplot.pandas
import holoviews as hv
import panel as pn
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from IPython.core.display import display, HTML
import plotly.io as pio
import webbrowser
import tempfile
import sys, os, subprocess

display(HTML("<style>.container { width:100% !important; }</style>"))
hv.extension('bokeh')

class MillerIndex:
    def __init__(self, params):
        self.a, self.b, self.c, self.alpha, self.beta, self.gamma, self.Mhkl = params
        self.params = params

    def crystal_system(self):
        if self.a == self.b == self.c and self.alpha == self.beta == self.gamma == 90:
            return self.cubic
        elif self.a == self.b != self.c and self.alpha == self.beta == 90 and self.gamma == 120:
            return self.hexagonal
        elif self.a == self.b != self.c and self.alpha == self.beta == self.gamma == 90:
            return self.tetragonal
        elif self.a != self.b != self.c and self.alpha == self.beta == self.gamma == 90:
            return self.orthorhombic
        elif self.a == self.b != self.c and self.alpha == self.beta != self.gamma and all(x == 90 for x in [self.alpha, self.beta]):
            return self.monoclinic
        elif self.a != self.b != self.c and self.alpha != self.beta != self.gamma and all(x == 90 for x in [self.alpha, self.beta, self.gamma]):
            return self.triclinic
        else:
            return self.rhombohedral

    def determine_crystal_system(self):
        if self.a == self.b == self.c and self.alpha == self.beta == self.gamma != 90:
            return self.rhombohedral
        elif self.a == self.b != self.c and self.alpha == self.beta == 90 and self.gamma == 120:
            return self.hexagonal
        elif self.a == self.b == self.c and self.alpha == self.beta == self.gamma == 90:
            return self.cubic
        elif self.a != self.b != self.c and self.alpha == self.beta == self.gamma == 90:
            return self.orthorhombic
        elif self.a == self.b != self.c and self.alpha == self.beta == self.gamma == 90:
            return self.tetragonal
        elif self.a != self.b != self.c and self.alpha == self.gamma == 90 and self.beta != 90:
            return self.monoclinic
        else:
            return self.triclinic

    def triclinic(self):
        return self.a, self.b, self.c, self.alpha, self.beta, self.gamma

    def monoclinic(self):
        return self.a, self.b, self.c, 90, self.beta, 90

    def orthorhombic(self):
        return self.a, self.b, self.c, 90, 90, 90

    def tetragonal(self):
        return self.a, self.a, self.c, 90, 90, 90

    def rhombohedral(self):
        return self.a, self.a, self.a, self.alpha, self.alpha, self.alpha

    def hexagonal(self):
        return self.a, self.a, self.c, 90, 90, 120

    def cubic(self):
        return self.a, self.a, self.a, 90, 90, 90

    def calculate_reciprocal_lattice_vectors(self, a, b, c, alpha, beta, gamma):
        # Convert angles to radians
        alpha_rad = math.radians(alpha)
        beta_rad = math.radians(beta)
        gamma_rad = math.radians(gamma)

        # Calculate unit cell volume V
        V = a * b * c * math.sqrt(1 - math.cos(alpha_rad)**2 - math.cos(beta_rad)**2 - math.cos(gamma_rad)**2 + 2 * math.cos(alpha_rad) * math.cos(beta_rad) * math.cos(gamma_rad))

        # Calculate reciprocal lattice vectors
        a_star = [(2 * math.pi / V) * (b * c * math.sin(alpha_rad)), 0, 0]
        b_star = [0, (2 * math.pi / V) * (a * c * math.sin(beta_rad)), 0]
        c_star = [0, 0, (2 * math.pi / V) * (a * b * math.sin(gamma_rad))]

        return a_star, b_star, c_star
    
    def calculate_q_xyz(self, h, k, l, a_star, b_star, c_star):
        q_x = h * a_star[0] + k * b_star[0] + l * c_star[0]
        q_y = h * a_star[1] + k * b_star[1] + l * c_star[1]
        q_z = h * a_star[2] + k * b_star[2] + l * c_star[2]
        return q_x, q_y, q_z

    def reciprocal_coordinates(self):
        crystal_system_func = self.crystal_system()
        a, b, c, alpha, beta, gamma = crystal_system_func()
        a_star, b_star, c_star = self.calculate_reciprocal_lattice_vectors(a, b, c, alpha, beta, gamma)
        
        hkl_combinations = list(itertools.product(range(-self.Mhkl, self.Mhkl + 1), repeat=3))
        reciprocal_coords = []

        for h, k, l in hkl_combinations:
            if (h, k, l) != (0, 0, 0):
                d_hkl = self.calculate_d_hkl(h, k, l, a, b, c, alpha, beta, gamma)
                q_x, q_y, q_z = self.calculate_q_xyz(h, k, l, a_star, b_star, c_star)
                reciprocal_coords.append({'h': h, 'k': k, 'l': l, 'd_hkl': d_hkl, 'q_x': q_x, 'q_y': q_y, 'q_z': q_z})

        return reciprocal_coords

    def calculate_d_hkl(self, h, k, l, a, b, c, alpha, beta, gamma):
        sin_alpha = np.sin(np.radians(alpha))
        sin_beta = np.sin(np.radians(beta))
        sin_gamma = np.sin(np.radians(gamma))

        if self.crystal_system() == self.cubic:
            return 1 / ((h**2 + k**2 + l**2) / a**2)**0.5
        elif self.crystal_system() == self.hexagonal:
            return 1 / ((4/3 * (h**2 + h*k + k**2) / a**2 + l**2 / c**2)**0.5)
        elif self.crystal_system() == self.tetragonal:
            return 1 / ((h**2 + k**2) / a**2 + l**2 / c**2)**0.5
        elif self.crystal_system() == self.orthorhombic:
            return 1 / ((h**2 / a**2 + k**2 / b**2 + l**2 / c**2)**0.5)
        elif self.crystal_system() == self.monoclinic:
            cos_gamma = np.cos(np.radians(gamma))
            return 1 / ((h**2 / a**2 + k**2 / b**2 + l**2 / c**2 - 2 * h * k * cos_gamma / (a * b))**0.5)
        elif self.crystal_system() == self.triclinic:
            cos_alpha = np.cos(np.radians(alpha))
            cos_beta = np.cos(np.radians(beta))
            cos_gamma = np.cos(np.radians(gamma))
            V = self.calculate_unit_cell_volume(a, b, c, alpha, beta, gamma)
            return 1 / (((h**2 * b**2 * c**2 * sin_alpha**2 + k**2 * a**2 * c**2 * sin_beta**2 + l**2 * a**2 * b**2 * sin_gamma**2 +
                        2 * h * k * a * b * c**2 * cos_alpha * cos_beta + 2 * h * l * a**2 * b * c * cos_alpha * cos_gamma +
                        2 * k * l * a * b**2 * c * cos_beta * cos_gamma) / (a**2 * b**2 * c**2 * (1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma)))**0.5)
        else:
            cos_alpha = np.cos(np.radians(alpha))
            return 1 / ((h**2 + k**2 + h * k * (1 - cos_alpha) / cos_alpha) / a**2 + l**2 / c**2)**0.5

    def plot_reciprocal_coordinates(self, reciprocal_coords):
        # Create a DataFrame from the reciprocal coordinates
        df = pd.DataFrame(reciprocal_coords)
        df['q_xy'] = np.sqrt(df['q_x'] ** 2 + df['q_y'] ** 2)
        df['hkl'] = df[['h', 'k', 'l']].apply(lambda x: f"({x['h']}{x['k']}{x['l']})", axis=1)

        # Create interactive plots using Hvplot
        plot1 = df.hvplot.scatter(x='q_x', y='q_y', hover_cols=['hkl'], title='q_x vs q_y')
        # plot2 = df.hvplot.scatter(x='q_x', y='q_z', hover_cols=['hkl'], title='q_x vs q_z')
        plot3 = df.hvplot.scatter(x='q_xy', y='q_z', hover_cols=['hkl'], title='q_xy vs q_z')

        # Combine the plots and show them in the browser
        # combined_plot = (plot1 + plot2 + plot3).cols(3)
        combined_plot = (plot1 + plot3).cols(2)
        return combined_plot

class MillerIndices:
    def __init__(self, a, b, c, alpha, beta, gamma, Mhkl):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.Mhkl = Mhkl

        self.a1, self.a2, self.a3 = self.compute_cartesian_vectors()
        self.V = self.compute_volume()
        self.a_star, self.b_star, self.c_star = self.compute_reciprocal_vectors()
        self.points = self.get_coordinates()
        self.rotated_points = None

    def compute_cartesian_vectors(self):
        alpha = np.radians(self.alpha)
        beta = np.radians(self.beta)
        gamma = np.radians(self.gamma)

        a1 = np.array([self.a, 0, 0])
        a2 = np.array([self.b * np.cos(gamma), self.b * np.sin(gamma), 0])
        a3 = np.array([self.c * np.cos(beta),
                       self.c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
                       self.c * np.sqrt(1 - np.cos(beta)**2 - ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma))**2)])

        return a1, a2, a3

    def compute_volume(self):
        return np.dot(self.a1, np.cross(self.a2, self.a3))

    def compute_reciprocal_vectors(self):
        a_star = 2 * np.pi * np.cross(self.a2, self.a3) / self.V
        b_star = 2 * np.pi * np.cross(self.a3, self.a1) / self.V
        c_star = 2 * np.pi * np.cross(self.a1, self.a2) / self.V

        return a_star, b_star, c_star

    def get_coordinates(self):
        Mhkl = int(self.Mhkl)
        coordinates = []

        for h in range(Mhkl):
            for k in range(Mhkl):
                for l in range(Mhkl):
                    qx = h * self.a_star[0] + k * self.b_star[0] + l * self.c_star[0]
                    qy = h * self.a_star[1] + k * self.b_star[1] + l * self.c_star[1]
                    qz = h * self.a_star[2] + k * self.b_star[2] + l * self.c_star[2]
                    coordinates.append((qx, qy, qz))

        return np.array(coordinates)
    
    def generate_lattice_points(self):
        points = []
        for h in range(self.Mhkl + 1):
            for k in range(self.Mhkl + 1):
                for l in range(self.Mhkl + 1):
                    points.append([h, k, l])
        return np.array(points)

    @staticmethod
    def cartesian_to_spherical(x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return theta, phi, r

    @staticmethod
    def rotate_theta(theta, phi, r, d_theta):
        theta_rotated = theta + d_theta
        x_rotated = r * np.sin(theta_rotated) * np.cos(phi)
        y_rotated = r * np.sin(theta_rotated) * np.sin(phi)
        z_rotated = r * np.cos(theta_rotated)
        return x_rotated, y_rotated, z_rotated

    @staticmethod
    def rotate_phi(theta, phi, r, d_phi):
        phi_rotated = phi + d_phi
        x_rotated = r * np.sin(theta) * np.cos(phi_rotated)
        y_rotated = r * np.sin(theta) * np.sin(phi_rotated)
        z_rotated = r * np.cos(theta)
        return x_rotated, y_rotated, z_rotated

    def rotate_x(self, thetax):
        thetax = np.radians(thetax)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(thetax), -np.sin(thetax)],
                       [0, np.sin(thetax), np.cos(thetax)]])
        rotated_coordinates = self.get_coordinates().dot(Rx.T)
        q_xy = np.sqrt(rotated_coordinates[:, 0] ** 2 + rotated_coordinates[:, 1] ** 2)
        return rotated_coordinates[:, 0], rotated_coordinates[:, 1], rotated_coordinates[:, 2], q_xy

    def rotate_y(self, thetay):
        thetay = np.radians(thetay)
        Ry = np.array([[np.cos(thetay), 0, np.sin(thetay)],
                       [0, 1, 0],
                       [-np.sin(thetay), 0, np.cos(thetay)]])
        rotated_coordinates = self.get_coordinates().dot(Ry.T)
        q_xy = np.sqrt(rotated_coordinates[:, 0] ** 2 + rotated_coordinates[:, 1] ** 2)
        return rotated_coordinates[:, 0], rotated_coordinates[:, 1], rotated_coordinates[:, 2], q_xy

    def cart_to_spherical(self, points):
        spherical_points = []
        for point in points:
            r = np.linalg.norm(point)
            theta = np.arccos(point[2] / r)
            phi = np.arctan2(point[1], point[0])
            spherical_points.append(np.array([r, theta, phi]))
        return np.array(spherical_points)

    def spherical_to_cart(self, spherical_point):
        r, theta, phi = spherical_point
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([x, y, z])
    
    def apply_gaussian_rotations(self, n, theta_mean, phi_mean, sigma_theta, sigma_phi):
        self.rotated_points = []

        spherical_points = self.cart_to_spherical(self.points)

        for point in spherical_points:
            if np.all(point == 0):  # Exclude the (0, 0, 0) point
                continue

            for _ in range(n):
                theta_offset = np.random.normal(theta_mean, sigma_theta)
                phi_offset = np.random.normal(phi_mean, sigma_phi)

                rotated_spherical = point.copy()
                rotated_spherical[1] += theta_offset
                rotated_spherical[2] += phi_offset

                rotated_cartesian = self.spherical_to_cart(rotated_spherical)

                if rotated_cartesian[0] >= 0 and rotated_cartesian[1] >= 0 and rotated_cartesian[2] >= 0:  # Positive quadrant only
                    qxy = np.sqrt(rotated_cartesian[0]**2 + rotated_cartesian[1]**2)
                    self.rotated_points.append(np.hstack([rotated_cartesian, qxy]))

        self.rotated_points = np.array(self.rotated_points)

    def generate_plots_mpl(self):
        if self.rotated_points is None:
            raise ValueError("No rotated points found. Please call apply_gaussian_rotations first.")

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        axs[0].scatter(self.rotated_points[:, 0], self.rotated_points[:, 1], label="q_x vs q_y")
        axs[1].scatter(self.rotated_points[:, 0], self.rotated_points[:, 2], label="q_x vs q_z")
        axs[2].scatter(self.rotated_points[:, 1], self.rotated_points[:, 2], label="q_y vs q_z")

        axs[0].set_xlabel('q_x')
        axs[0].set_ylabel('q_y')
        axs[0].legend()

        axs[1].set_xlabel('q_x')
        axs[1].set_ylabel('q_z')
        axs[1].legend()

        axs[2].set_xlabel('q_y')
        axs[2].set_ylabel('q_z')
        axs[2].legend()

        plt.show()

    def generate_2d_plot(self, x, y, x_label, y_label, color):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=3, color=color), hoverinfo='x+y'))
        fig.update_layout(xaxis_title=x_label, yaxis_title=y_label, xaxis_title_font=dict(size=14), yaxis_title_font=dict(size=14), width=400, height=400)
        return fig

    def generate_3d_plot(self):
        # Calculate radial distances for color scaling
        radial_distances = np.sqrt(self.rotated_points[:, 0]**2 + self.rotated_points[:, 1]**2 + self.rotated_points[:, 2]**2)
        colors = radial_distances / np.max(radial_distances)

        # Create the 3D plot
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(x=self.rotated_points[:, 0],
                                   y=self.rotated_points[:, 1],
                                   z=self.rotated_points[:, 2],
                                   mode='markers',
                                   marker=dict(size=2, color=colors, colorscale='turbo', showscale=False),
                                   hoverinfo='x+y+z'))

        fig.update_layout(scene=dict(xaxis_title='q_x',
                                      yaxis_title='q_y',
                                      zaxis_title='q_z',
                                      xaxis_title_font=dict(size=14),
                                      yaxis_title_font=dict(size=14),
                                      zaxis_title_font=dict(size=14)),
                          width=800, height=800)

        return fig
    
    def generate_plots(self):
        # fig_2d = []
        # fig_2d.append(self.generate_2d_plot(self.rotated_points[:, 0], self.rotated_points[:, 1], 'q_x', 'q_y', 'red'))
        # fig_2d.append(self.generate_2d_plot(self.rotated_points[:, 0], self.rotated_points[:, 2], 'q_x', 'q_z', 'blue'))
        # fig_2d.append(self.generate_2d_plot(np.sqrt(self.rotated_points[:, 0]**2 + self.rotated_points[:, 1]**2),
        #                                     self.rotated_points[:, 2], 'q_xy', 'q_z', 'green'))

        # Create 3D plot
        fig_3d = self.generate_3d_plot()

        # Combine 2D and 3D plots in a single layout
        final_fig = make_subplots(rows=2, cols=3, row_heights=[0.3, 0.5], vertical_spacing=0.15,
                                specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                                        [{"type": "scatter3d", "colspan": 3}, None, None]],
                                subplot_titles=("q_x vs q_y", "q_x vs q_z", "q_xy vs q_z", "3D Plot"))

        # for idx, fig in enumerate(fig_2d):
        #     final_fig.add_trace(fig.data[0], row=1, col=idx + 1)
        #     final_fig.update_xaxes(title_text=fig.layout.xaxis.title.text, row=1, col=idx + 1)
        #     final_fig.update_yaxes(title_text=fig.layout.yaxis.title.text, row=1, col=idx + 1)

        final_fig.add_trace(fig_3d.data[0], row=2, col=1)

        final_fig.update_layout(scene=dict(xaxis=dict(title='q_x', title_font=dict(size=14)),
                                        yaxis=dict(title='q_y', title_font=dict(size=14)),
                                        zaxis=dict(title='q_z', title_font=dict(size=14))))

        final_fig.update_layout(title="Rotated Points Plots", height=800, width=800)
        # final_fig.show()
    
        # Convert the figure to an HTML string
        html_string = pio.to_html(final_fig, full_html=True)

        # Save the HTML string to a temporary file and open it in a browser
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
            f.write(html_string)
            
            # Open the file in Chrome
            try:
                chrome_path = None

                # Locate the Chrome browser based on the operating system
                if sys.platform.startswith('win'):
                    chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'
                elif sys.platform.startswith('darwin'):
                    chrome_path = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
                elif sys.platform.startswith('linux'):
                    chrome_path = '/usr/bin/google-chrome'

                if chrome_path:
                    subprocess.Popen([chrome_path, f.name])
                else:
                    subprocess.Popen(["start", f.name], shell=True)

            except Exception as e:
                print(f"Error opening browser: {e}")
                print("Opening in the default browser instead.")
                subprocess.Popen(["start", f.name], shell=True)

# ---- Generate plots example
# lattice = MillerIndices(a=3, b=10, c=2, alpha=90, beta=90, gamma=120, Mhkl=6)
# lattice.apply_gaussian_rotations(n=100, theta_mean=0, phi_mean=0, sigma_theta=.1, sigma_phi=0.001)
# lattice.generate_plots()

'''
# class MillerIndices:
#     def __init__(self, a, b, c, alpha, beta, gamma, Mhkl):
#         self.a = a
#         self.b = b
#         self.c = c
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.Mhkl = Mhkl

#         self.a1, self.a2, self.a3 = self.compute_cartesian_vectors()
#         self.V = self.compute_volume()
#         self.a_star, self.b_star, self.c_star = self.compute_reciprocal_vectors()

#     def compute_cartesian_vectors(self):
#         alpha = np.radians(self.alpha)
#         beta = np.radians(self.beta)
#         gamma = np.radians(self.gamma)

#         a1 = np.array([self.a, 0, 0])
#         a2 = np.array([self.b * np.cos(gamma), self.b * np.sin(gamma), 0])
#         a3 = np.array([self.c * np.cos(beta),
#                        self.c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
#                        self.c * np.sqrt(1 - np.cos(beta)**2 - ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma))**2)])

#         return a1, a2, a3

#     def compute_volume(self):
#         return np.dot(self.a1, np.cross(self.a2, self.a3))

#     def compute_reciprocal_vectors(self):
#         a_star = 2 * np.pi * np.cross(self.a2, self.a3) / self.V
#         b_star = 2 * np.pi * np.cross(self.a3, self.a1) / self.V
#         c_star = 2 * np.pi * np.cross(self.a1, self.a2) / self.V

#         return a_star, b_star, c_star

#     def get_coordinates(self):
#         Mhkl = int(self.Mhkl)
#         coordinates = []

#         for h in range(Mhkl):
#             for k in range(Mhkl):
#                 for l in range(Mhkl):
#                     qx = h * self.a_star[0] + k * self.b_star[0] + l * self.c_star[0]
#                     qy = h * self.a_star[1] + k * self.b_star[1] + l * self.c_star[1]
#                     qz = h * self.a_star[2] + k * self.b_star[2] + l * self.c_star[2]
#                     coordinates.append((qx, qy, qz))

#         return np.array(coordinates)
    
#     def generate_plots1(self):
#         Mhkl = int(self.Mhkl)
#         coordinates = []

#         for h in range(Mhkl):
#             for k in range(Mhkl):
#                 for l in range(Mhkl):
#                     qx = h * self.a_star[0] + k * self.b_star[0] + l * self.c_star[0]
#                     qy = h * self.a_star[1] + k * self.b_star[1] + l * self.c_star[1]
#                     qz = h * self.a_star[2] + k * self.b_star[2] + l * self.c_star[2]
#                     coordinates.append((qx, qy, qz))

#         coordinates = np.array(coordinates)
#         qxy = np.sqrt(coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2)

#         fig, axs = plt.subplots(1, 3, figsize=(15, 5))

#         # Plot qx vs qy
#         axs[0].scatter(coordinates[:, 0], coordinates[:, 1])
#         axs[0].set_xlabel("qx")
#         axs[0].set_ylabel("qy")
#         axs[0].set_title("Plot of qx vs qy")
#         axs[0].grid(True)

#         # Plot qx vs qz
#         axs[1].scatter(coordinates[:, 0], coordinates[:, 2])
#         axs[1].set_xlabel("qx")
#         axs[1].set_ylabel("qz")
#         axs[1].set_title("Plot of qx vs qz")
#         axs[1].grid(True)

#         # Plot qxy vs qz
#         axs[2].scatter(qxy, coordinates[:, 2])
#         axs[2].set_xlabel("qxy")
#         axs[2].set_ylabel("qz")
#         axs[2].set_title("Plot of qxy vs qz")
#         axs[2].grid(True)

#         plt.show()

#     def generate_plots2(self):
#         Mhkl = int(self.Mhkl)
#         coordinates = []
#         hkl_labels = []

#         for h in range(Mhkl):
#             for k in range(Mhkl):
#                 for l in range(Mhkl):
#                     qx = h * self.a_star[0] + k * self.b_star[0] + l * self.c_star[0]
#                     qy = h * self.a_star[1] + k * self.b_star[1] + l * self.c_star[1]
#                     qz = h * self.a_star[2] + k * self.b_star[2] + l * self.c_star[2]
#                     coordinates.append((qx, qy, qz))
#                     hkl_labels.append(f"({h}, {k}, {l})")

#         coordinates = np.array(coordinates)
#         qxy = np.sqrt(coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2)

#         # Create a subplot with 1 row and 3 columns
#         fig = make_subplots(rows=1, cols=3, subplot_titles=("Plot of qx vs qy", "Plot of qx vs qz", "Plot of qxy vs qz"))

#         # Plot qx vs qy
#         fig.add_trace(go.Scatter(x=coordinates[:, 0], y=coordinates[:, 1], text=hkl_labels, mode="markers+text", marker=dict(size=8), textposition="top center", name="qx vs qy"), row=1, col=1)

#         # Plot qx vs qz
#         fig.add_trace(go.Scatter(x=coordinates[:, 0], y=coordinates[:, 2], text=hkl_labels, mode="markers+text", marker=dict(size=8), textposition="top center", name="qx vs qz"), row=1, col=2)

#         # Plot qxy vs qz
#         fig.add_trace(go.Scatter(x=qxy, y=coordinates[:, 2], text=hkl_labels, mode="markers+text", marker=dict(size=8), textposition="top center", name="qxy vs qz"), row=1, col=3)

#         # Update xaxis and yaxis labels
#         fig.update_xaxes(title_text="qx", row=1, col=1)
#         fig.update_yaxes(title_text="qy", row=1, col=1)
#         fig.update_xaxes(title_text="qx", row=1, col=2)
#         fig.update_yaxes(title_text="qz", row=1, col=2)
#         fig.update_xaxes(title_text="qxy", row=1, col=3)
#         fig.update_yaxes(title_text="qz", row=1, col=3)

#         fig.show()

#     def generate_plots(self):
#         Mhkl = int(self.Mhkl)
#         coordinates = []
#         hkl_labels = []

#         for h in range(Mhkl):
#             for k in range(Mhkl):
#                 for l in range(Mhkl):
#                     qx = h * self.a_star[0] + k * self.b_star[0] + l * self.c_star[0]
#                     qy = h * self.a_star[1] + k * self.b_star[1] + l * self.c_star[1]
#                     qz = h * self.a_star[2] + k * self.b_star[2] + l * self.c_star[2]
#                     coordinates.append((qx, qy, qz))
#                     hkl_labels.append(f"({h}, {k}, {l})")

#         coordinates = np.array(coordinates)
#         qxy = np.sqrt(coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2)

#         # Create a subplot with 1 row and 3 columns
#         fig = make_subplots(rows=1, cols=3, subplot_titles=("Plot of qx vs qy", "Plot of qx vs qz", "Plot of qxy vs qz"))

#         # Plot qx vs qy
#         fig.add_trace(go.Scatter(x=coordinates[:, 0], y=coordinates[:, 1], text=hkl_labels, mode="markers", marker=dict(size=8), hovertemplate="(%{x:.2f}, %{y:.2f})<br>%{text}"), row=1, col=1)

#         # Plot qx vs qz
#         fig.add_trace(go.Scatter(x=coordinates[:, 0], y=coordinates[:, 2], text=hkl_labels, mode="markers", marker=dict(size=8), hovertemplate="(%{x:.2f}, %{y:.2f})<br>%{text}"), row=1, col=2)

#         # Plot qxy vs qz
#         fig.add_trace(go.Scatter(x=qxy, y=coordinates[:, 2], text=hkl_labels, mode="markers", marker=dict(size=8), hovertemplate="(%{x:.2f}, %{y:.2f})<br>%{text}"), row=1, col=3)

#         # Update xaxis and yaxis labels
#         fig.update_xaxes(title_text="qx", row=1, col=1)
#         fig.update_yaxes(title_text="qy", row=1, col=1)
#         fig.update_xaxes(title_text="qx", row=1, col=2)
#         fig.update_yaxes(title_text="qz", row=1, col=2)
#         fig.update_xaxes(title_text="qxy", row=1, col=3)
#         fig.update_yaxes(title_text="qz", row=1, col=3)

#         fig.show()

#     def rotate_x(self, thetax):
#         thetax = np.radians(thetax)
#         Rx = np.array([[1, 0, 0],
#                        [0, np.cos(thetax), -np.sin(thetax)],
#                        [0, np.sin(thetax), np.cos(thetax)]])
#         rotated_coordinates = self.get_coordinates().dot(Rx.T)
#         q_xy = np.sqrt(rotated_coordinates[:, 0] ** 2 + rotated_coordinates[:, 1] ** 2)
#         return rotated_coordinates[:, 0], rotated_coordinates[:, 1], rotated_coordinates[:, 2], q_xy

#     def rotate_y(self, thetay):
#         thetay = np.radians(thetay)
#         Ry = np.array([[np.cos(thetay), 0, np.sin(thetay)],
#                        [0, 1, 0],
#                        [-np.sin(thetay), 0, np.cos(thetay)]])
#         rotated_coordinates = self.get_coordinates().dot(Ry.T)
#         q_xy = np.sqrt(rotated_coordinates[:, 0] ** 2 + rotated_coordinates[:, 1] ** 2)
#         return rotated_coordinates[:, 0], rotated_coordinates[:, 1], rotated_coordinates[:, 2], q_xy

# # Example usage:
# # a, b, c = 3, 10, 5
# # alpha, beta, gamma = 90, 90, 90
# # Mhkl = 3

# # miller_indices = MillerIndices(a, b, c, alpha, beta, gamma, Mhkl)
# # miller_indices.generate_plots()

# # Commented out example
# # a, b, c = 1, 1, 12
# # alpha, beta, gamma = 90, 90, 120
# # Mhkl = 3

# # miller_indices2 = MillerIndices(a, b, c, alpha, beta, gamma, Mhkl)
# # miller_indices2.generate_plots()

def calculate_d_hkl(self, h, k, l, a, b, c, alpha, beta, gamma):
    # Convert angles to radians
    alpha_rad = math.radians(alpha)
    beta_rad = math.radians(beta)
    gamma_rad = math.radians(gamma)

    # Calculate unit cell volume V
    V = a * b * c * math.sqrt(1 - math.cos(alpha_rad)**2 - math.cos(beta_rad)**2 - math.cos(gamma_rad)**2 + 2 * math.cos(alpha_rad) * math.cos(beta_rad) * math.cos(gamma_rad))

    # Calculate d-spacing
    d_hkl = V / (h * a * math.sin(alpha_rad) + k * b * math.sin(beta_rad) + l * c * math.sin(gamma_rad))

    return d_hkl
    '''