from scipy.optimize import least_squares
import numpy as np
import AFF as AFF
import glob2 as glob
import os
import re
import pygix
import pyFAI
import numpy as np
from scipy.optimize import least_squares
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QLineEdit, QVBoxLayout, QWidget, QPushButton, QSpinBox, QHBoxLayout
import sys
import numpy as np
import plotly.graph_objs as go
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QLineEdit, QVBoxLayout, QWidget, QPushButton, QSpinBox, QHBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
import sys
import csv
import numpy as np
import plotly.graph_objs as go
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QLineEdit, QVBoxLayout, QWidget, QPushButton, QSpinBox, QHBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
import sys
from PyQt5.QtWidgets import QApplication
# from simuWAXS import simuWAXS
# from mainwindow import MainWindow  # Import the MainWindow class from your View code file

class simuWAXS:
    def __init__(self):
        self.a1 = None
        self.a2 = None
        self.a3 = None
        self.positions = None

    @staticmethod
    def stripparentpath(poscar_path):
        parent_path = os.path.dirname(os.path.abspath(poscar_path))
        return parent_path

    @staticmethod
    def findfilepath(parent_folder, fileStr):
        pathStr = parent_folder + fileStr
        filepath = sorted(glob.glob(pathStr))
        return filepath

    @staticmethod
    def readPOSCAR(address):
        file = open(address)
        ind = 0
        for x in file:
            ind += 1
            if ind == 3:
                x = x.lstrip()
                x = x.rstrip()
                x = x.split()
                x = np.array(x)
                p = x.astype(float)
                a1 = p
            if ind == 4:
                x = x.lstrip()
                x = x.rstrip()
                x = x.split()
                x = np.array(x)
                p = x.astype(float)
                a2 = p
            if ind == 5:
                x = x.lstrip()
                x = x.rstrip()
                x = x.split()
                x = np.array(x)
                p = x.astype(float)
                a3 = p
            if ind == 6:
                x = x.lstrip()
                x = x.rstrip()
                x = x.split()
                iii = 0
                pp = np.zeros(len(x))
                for p in x:
                    pp[iii] = AFF.atom_dict[p]
                    iii += 1
                x = np.array(pp)
                z = x.astype(int)
            if ind == 7:
                x = x.lstrip()
                x = x.rstrip()
                x = x.split()
                x = np.array(x)
                z1 = x.astype(int)
                temp = np.sum(z1)
                position = np.zeros((temp, 4))
            if ind > 8:
                x = x.lstrip()
                x = x.rstrip()
                x = x.split("         ")
                x = np.array(x)
                p = x.astype(float)
                position[ind - 9, 1] = p[0]
                position[ind - 9, 2] = p[1]
                position[ind - 9, 3] = p[2]
        file.close()

        ind = 0
        iii = 0
        for ii in z1:
            position[iii:iii + ii + 1, 0] = z[ind]
            iii = iii + ii
            ind = ind + 1
        return a1, a2, a3, position

    @staticmethod
    def Bragg_Peaks(a1, a2, a3, positions, theta_x, theta_y, hkl_dimension):
        M = [a1, a2, a3]
        M = np.asarray(M)
        # Rotation Matrix respect to X axis, rotation angle = theta_x
        Rx=np.array([[1,0,0],[0,np.cos(theta_x),-np.sin(theta_x)],[0,np.sin(theta_x),np.cos(theta_x)]])
        # Rotation Matrix respect to Y axis, rotation angle = theta_y
        Ry=np.array([[np.cos(theta_y),0,-np.sin(theta_y)],[0,1,0],[np.sin(theta_y),0,np.cos(theta_y)]])

        # Rotation of the sample
        M=np.matmul(M, Rx)
        M=np.matmul(M, Ry)
        
        # New lattice parameter
        aa1=M[0,:]
        aa2=M[1,:]
        aa3=M[2,:]

        # reciprocal lattice
        volume=np.matmul(aa3,np.cross(aa1,aa2))
        b1=2*np.pi*np.cross(aa2,aa3)/volume
        b2=2*np.pi*np.cross(aa3,aa1)/volume
        b3=2*np.pi*np.cross(aa1,aa2)/volume

        # grid for Miller index
        i =np.linspace(-hkl_dimension,hkl_dimension,2*hkl_dimension+1)
        H,K,L=np.meshgrid(i,i,i)
        
        # The position of Bragg peaks in reciprocal space
        G1=H*b1[0]+K*b2[0]+L*b3[0]
        G2=H*b1[1]+K*b2[1]+L*b3[1]
        G3=H*b1[2]+K*b2[2]+L*b3[2]
        
        ss=np.size(positions)/4
        ss=int(ss)
        
        # load atomic form factor table
        AF=AFF.AFF()
        
        # calculate the atomic form factor
        ii =np.linspace(0,ss-1,ss)
        ii =ii.astype(int)
        q2=G1*G1+G2*G2+G3*G3
        F= 0
        for j in ii:
            x = np.searchsorted(AF[:,0],positions[j,0])
            fq= 0
            # first formula at http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
            fq=fq+AF[x,1]*np.exp(-AF[x,2]*q2/16/pow(np.pi,2))
            fq=fq+AF[x,3]*np.exp(-AF[x,4]*q2/16/pow(np.pi,2))
            fq=fq+AF[x,5]*np.exp(-AF[x,6]*q2/16/pow(np.pi,2))
            fq=fq+AF[x,7]*np.exp(-AF[x,8]*q2/16/pow(np.pi,2))
            fq=fq+AF[x,9]
            # position of atom in real space cartesian coordinate(angstrom)
            RR=positions[j,1]*aa1+positions[j,2]*aa2+positions[j,3]*aa3
            F=F+fq*np.exp(1j*(G1*RR[0]+G2*RR[1]+G3*RR[2]))
        F=np.abs(F)
        F=pow(F,2)
        BPeaks=np.concatenate((G1,G2,G3,F), axis= 0)

        Mqxy = pow(G1*G1+G2*G2,0.5)
        Mqz = pow(G3*G3,0.5)
        FMiller = F

        return BPeaks, Mqxy, Mqz, FMiller

    @staticmethod
    def loadPOSCAR(poscar_folder, fileStr, BPeakParams):
        
        theta_x, theta_y, hkl_dimension = BPeakParams # unpack input params

        poscar_path = simuWAXS.findfilepath(poscar_folder, fileStr)[0] # build the .vasp path
        print("Loaded POSCAR file from path: " + poscar_path)

        a1, a2, a3, positions = simuWAXS.readPOSCAR(poscar_path) # extract relevant metadata by reading the POSCAR
        BPeaks, Mqxy, Mqz, FMiller = simuWAXS.Bragg_Peaks(a1, a2, a3, positions, theta_x, theta_y, hkl_dimension)

        return BPeaks, Mqxy, Mqz, FMiller

class simuWAXSExtended(simuWAXS):

    def __init__(self):
        super().__init__()

    def extract_ROI(self, data, top_left, bottom_right):
        x1, y1 = top_left
        x2, y2 = bottom_right
        return data[y1:y2, x1:x2]

    def compare_ROI(self, real_data, simulated_data, real_roi, sim_roi):
        real_data_roi = self.extract_ROI(real_data, *real_roi)
        simulated_data_roi = self.extract_ROI(simulated_data, *sim_roi)
        return np.linalg.norm(real_data_roi - simulated_data_roi)

    def monte_carlo_peak_finder(self, real_data, simulated_data, real_roi, sim_roi, n_iter=1000):
        best_diff = float("inf")
        best_sim_roi = None

        for _ in range(n_iter):
            random_sim_roi = (
                (np.random.randint(0, simulated_data.shape[1] - (sim_roi[1][0] - sim_roi[0][0])),
                 np.random.randint(0, simulated_data.shape[0] - (sim_roi[1][1] - sim_roi[0][1]))),
                sim_roi[1]
            )
            diff = self.compare_ROI(real_data, simulated_data, real_roi, random_sim_roi)
            if diff < best_diff:
                best_diff = diff
                best_sim_roi = random_sim_roi

        return best_sim_roi

    def modify_lattice_constants(self, a1, a2, a3, delta_a1, delta_a2, delta_a3):
        new_a1 = a1 + delta_a1
        new_a2 = a2 + delta_a2
        new_a3 = a3 + delta_a3
        return new_a1, new_a2, new_a3

    def intensitymap (poscar_folder, fileStr, BPeakParams, crystParams, imgParams):
        # unpack variables
        theta_x, theta_y, hkl_dimension = BPeakParams
        sigma_theta, sigma_phi, sigma_r = crystParams
        resolutionx, qxymax, qzmax, qzmin = imgParams

        # map the image space based on input parameters
        resolutionz = int(resolutionx/qxymax*qzmax)
        gridx, gridz = np.meshgrid(np.linspace(-qxymax,qxymax,resolutionx),np.linspace(0,qzmax,resolutionz))
        
        # BPeaks = loadPOSCAR(poscar_folder, fileStr, BPeakParams)
        BPeaks, Mqxy, Mqz, FMiller = loadPOSCAR(poscar_folder, fileStr, BPeakParams)

        t1_start = 0
        t1_stop = 0
        t1_start = time.process_time()

        iMiller = hkl_dimension*2+1
        G1 = BPeaks[0:iMiller,:,:]+np.finfo(float).eps
        G2 = BPeaks[iMiller:2*iMiller,:,:]+np.finfo(float).eps
        G3 = BPeaks[2*iMiller:3*iMiller,:,:]+np.finfo(float).eps
        F = BPeaks[3*iMiller:4*iMiller,:,:]
        
        Eye=np.ones((iMiller,iMiller,iMiller))
        # The positions(r0,theta0,phi0) of Bragg peaks in spherical coordinates.
        theta0=np.pi/2-np.arctan(G3/np.sqrt(pow(G2,2)+pow(G1,2)))
        phi0=np.ones((iMiller,iMiller,iMiller))
        i =np.arange(iMiller)
        for k1 in i:
            for k2 in i:
                for k3 in i:
                    if G1[k1,k2,k3]>0:
                        phi0[k1,k2,k3]=np.arcsin(G2[k1,k2,k3]/np.sqrt(pow(G2[k1,k2,k3],2)+pow(G1[k1,k2,k3],2)))
                    else:
                        phi0[k1,k2,k3]=np.pi+np.arcsin(G2[k1,k2,k3]/np.sqrt(pow(G2[k1,k2,k3],2)+pow(G1[k1,k2,k3],2)))
                    if abs(G2[k1,k2,k3])<0.2:
                        if abs(G1[k1,k2,k3])<0.2:
                            phi0[k1,k2,k3]= 0
                        
        r0=np.sqrt(pow(G1,2)+pow(G2,2)+pow(G3,2))

        # The positions(r,theta,phi) of image plane in spherical coordinates.
        ix,iy=gridx.shape
        I0=np.ones((ix,iy))
        ix=np.arange(ix)
        iy=np.arange(iy)
        for x in ix:
            for y in iy:
                theta=np.pi/2-np.arctan(gridz[x,y]/abs(gridx[x,y]))
                r=np.sqrt(pow(gridx[x,y],2)+pow(gridz[x,y],2))
                if gridx[x,y]>0:
                    phi = 0
                else:
                    phi =np.pi
                phi =phi*Eye
                phid =abs(phi-phi0)
                phid =abs(abs(phid-np.pi)-np.pi)
                I1=np.exp(-0.5*pow(theta*Eye-theta0,2)/sigma_theta/sigma_theta)
                I2=np.exp(-0.5*phid*phid/sigma_phi/sigma_phi)
                I3=np.exp(-0.5*pow(r*Eye-r0,2)/sigma_r/sigma_r)
                Intensity=I1*I2*I3*F
                I0[x,y]=np.sum(Intensity)
        
        # t1_stop = time.process_time()
        # print('CPU Time: ')
        # print(t1_stop - t1_start,' s')

        return I0, BPeaks, Mqxy, Mqz, FMiller

    def residual(self, params, real_data, simulated_data, real_roi, sim_roi):
        a1, a2, a3, delta_a1, delta_a2, delta_a3 = params
        new_a1, new_a2, new_a3 = self.modify_lattice_constants(a1, a2, a3, delta_a1, delta_a2, delta_a3)
        BPeaks, Mqxy, Mqz, FMiller = self.Bragg_Peaks(new_a1, new_a2, new_a3, self.positions, theta_x, theta_y, hkl_dimension)
        new_simulated_data, BPeaks, Mqxy, Mqz, FMiller = self.intensitymap (poscar_folder, fileStr, BPeakParams, crystParams, imgParams)
        # You should implement a method to generate a simulated_data array from the output of Bragg_Peaks
        # new_simulated_data = generate_simulated_data(BPeaks, Mqxy, Mqz, FMiller)
        
        return self.compare_ROI(real_data, new_simulated_data, real_roi, sim_roi)

    def refine_lattice_constants(self, real_data, simulated_data, real_roi, sim_roi, initial_params, bounds):
        result = least_squares(self.residual, initial_params, bounds=bounds, args=(real_data, simulated_data, real_roi, sim_roi))
        return result.x

    def extract_metadata(self, filename, keylist):
            metadata = {}
            for key in keylist:
                pattern = r'{}_([^-_]+)'.format(key)
                match = re.search(pattern, filename)
                if match:
                    metadata[key] = match.group(1)
            return metadata

    def load_poni(self, poni_file):
        ai = pyFAI.load(poni_file)
        return ai

    def process_poni(self, ai, real_data):
        gix = pygix.Transform(ai)
        q_space, intensity = gix.q_space(real_data)
        return q_space, intensity

    def store_processed_data(self, q_space, intensity):
        self.q_space = q_space
        self.intensity = intensity

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("SimuWAXS ViewModel")

        layout = QVBoxLayout()

        self.keylist_label = QLabel("Keylist (comma-separated):")
        layout.addWidget(self.keylist_label)

        self.keylist_entry = QLineEdit()
        layout.addWidget(self.keylist_entry)

        self.select_file_button = QPushButton("Select Data File")
        self.select_file_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_file_button)

        self.file_label = QLabel("No file selected")
        layout.addWidget(self.file_label)

        h_layout = QHBoxLayout()

        self.theta_x_label = QLabel("Theta_x:")
        h_layout.addWidget(self.theta_x_label)

        self.theta_x_spinbox = QSpinBox()
        h_layout.addWidget(self.theta_x_spinbox)

        self.theta_y_label = QLabel("Theta_y:")
        h_layout.addWidget(self.theta_y_label)

        self.theta_y_spinbox = QSpinBox()
        h_layout.addWidget(self.theta_y_spinbox)

        self.hkl_dimension_label = QLabel("HKL Dimension:")
        h_layout.addWidget(self.hkl_dimension_label)

        self.hkl_dimension_spinbox = QSpinBox()
        h_layout.addWidget(self.hkl_dimension_spinbox)

        layout.addLayout(h_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.show_heatmap_button = QPushButton("Show Heatmaps")
        self.show_heatmap_button.clicked.connect(self.show_heatmaps)
        layout.addWidget(self.show_heatmap_button)

        self.show_scatterplots_button = QPushButton("Show Scatterplots")
        self.show_scatterplots_button.clicked.connect(self.show_scatterplots)
        layout.addWidget(self.show_scatterplots_button)

        self.web_view = QWebEngineView()
        layout.addWidget(self.web_view)

        self.save_results_button = QPushButton("Save Results")
        self.save_results_button.clicked.connect(self.save_results)
        layout.addWidget(self.save_results_button)

    def select_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName()
        if file_path:
            self.file_label.setText("Selected file: " + file_path)

    def show_heatmaps(self):
        real_data = np.random.rand(10, 10)  # Replace with actual real data
        simulated_data = np.random.rand(10, 10)  # Replace with actual simulated data

        fig = go.Figure()

        fig.add_trace(go.Heatmap(z=real_data, name="Real Data"))
        fig.add_trace(go.Heatmap(z=simulated_data, name="Simulated Data", xaxis="x2", yaxis="y2"))

        fig.update_layout(
            xaxis=dict(domain=[0, 0.45]),
            xaxis2=dict(domain=[0.55, 1]),
            yaxis=dict(automargin=True),
            yaxis2=dict(automargin=True, anchor="x2"),
            hovermode="closest"
        )

        self.web_view.setHtml(fig.to_html(include_plotlyjs="cdn"))

    def show_scatterplots(self):
            real_data = np.random.rand(10, 2)  # Replace with actual real data peak centers
            simulated_data = np.random.rand(10, 2)  # Replace with actual simulated data peak centers

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=real_data[:, 0], y=real_data[:, 1], mode="markers", name="Real Data"))
            fig.add_trace(go.Scatter(x=simulated_data[:, 0], y=simulated_data[:, 1], mode="markers", name="Simulated Data", xaxis="x2", yaxis="y2"))

            fig.update_layout(
                xaxis=dict(domain=[0, 0.45]),
                xaxis2=dict(domain=[0.55, 1]),
                yaxis=dict(automargin=True),
                yaxis2=dict(automargin=True, anchor="x2"),
                hovermode="closest"
            )

            self.web_view.setHtml(fig.to_html(include_plotlyjs="cdn"))
    
    def save_results(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "CSV Files (*.csv);;All Files (*)", options=options)
        
        if file_name:
            if not file_name.endswith('.csv'):
                file_name += '.csv'
            
            refined_lattice_constants = [10.5, 11.3, 12.4]  # Replace with actual refined lattice constants
            texturing_parameters = [0.2, 0.3, 0.4]  # Replace with actual texturing parameters

            with open(file_name, mode='w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(["Refined Lattice Constants"])
                csv_writer.writerow(["a", "b", "c"])
                csv_writer.writerow(refined_lattice_constants)
                csv_writer.writerow([])
                csv_writer.writerow(["Texturing Parameters"])
                csv_writer.writerow(["t1", "t2", "t3"])
                csv_writer.writerow(texturing_parameters)
            
            print(f"Results saved to {file_name}")


app = QApplication(sys.argv)
main_window = MainWindow()
main_window.show()
sys.exit(app.exec_())

class Controller:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.main_window = MainWindow()
        self.simuWAXS_instance = simuWAXS()

        # Connect signals and slots
        self.main_window.file_button.clicked.connect(self.load_real_data)
        self.main_window.run_simulation_button.clicked.connect(self.run_simulation)
        self.main_window.save_results_button.clicked.connect(self.save_results)
        self.main_window.error_signal.connect(self.display_error_message)

    def load_real_data(self):
        # Get user inputs for file dialog and keylist
        real_data_file = self.main_window.get_real_data_file()
        keylist = self.main_window.get_keylist()

        # Pass the inputs to the simuWAXS_instance
        self.simuWAXS_instance.load_real_data(real_data_file, keylist)

    def run_simulation(self):
        # Get user inputs for simulation parameters
        BPeakParams = self.main_window.get_simulation_parameters()

        # Pass the inputs to the simuWAXS_instance
        self.simuWAXS_instance.loadPOSCAR(*BPeakParams)

        # Run the simulation
        BPeaks, Mqxy, Mqz, FMiller = self.simuWAXS_instance.run_simulation()

        # Update the plots in the main window
        self.main_window.update_plots(BPeaks, Mqxy, Mqz, FMiller)

    def save_results(self):
        # Get user inputs for saving the results
        save_file = self.main_window.get_save_file()

        # Save the results to the specified file
        self.simuWAXS_instance.save_results(save_file)

    def display_error_message(self, error_msg):
        self.main_window.show_error_message(error_msg)

    def run(self):
        self.main_window.show()
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    controller = Controller()
    controller.run()

# This Controller class initializes the MainWindow and simuWAXS instances, 
# and connects the signals emitted by the MainWin