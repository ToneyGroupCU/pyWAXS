# ----------------------------------------------------------------------------------------- #
# -------- PyFAI -------- #
import pyFAI
import pyFAI.gui
import pyFAI.detectors
import pyFAI.calibrant
# -------- PyGIX -------- #
import pygix
# import pygix.plotting as gixsplt
# -------- Standard Libraries -------- #
import math, fabio, silx, os, re, time, csv, io, pylatex, lmfit, psutil, cv2, sys, gc, dask
from dask import delayed, compute
import dask.array as da
import numpy as np
import pandas as pd
import glob2 as glob
from IPython.display import clear_output
from PIL import Image
from pathlib import Path
from lmfit import Model
# from zipfile import ZipFile
# --------- SciPy ----------- #
import scipy as sp
from scipy import signal
from scipy.optimize import curve_fit
from scipy.special import wofz
from scipy.signal import fftconvolve
from scipy.signal import argrelextrema
from scipy.signal import find_peaks, peak_prominences, peak_widths
# -------- Matplotlib -------- #
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import matplotlib as mpl
from matplotlib import cm
import matplotlib.tri as tri
from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.pyplot import figure
# -------- PyQt5 -------- #
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QDialog, QLineEdit, QPushButton, QFileDialog, QRadioButton, QLabel, QDialogButtonBox
from PyQt5 import QtWidgets
# ----------------------------------------------------------------------------------------- #

class GIXSDataReduction:
    def __init__(self, wcard, keylist=None, maskdata=None, correctSolidAngle=True, polarization_factor=None, dark=None, flat=None):
        self.wcard = wcard # wildcard string for glob
        self.pyg = pygix.Transform()
        self.poni_file = None
        self.mask_file = None
        self.tiff_dict = None
        self.geometry_dict = None
        self.keylist = keylist if keylist is not None else ['solutionnum', 'composition', 'filtopt','molarity', 'purgerate', 'sub', 'solvol', 'sampnum', 'clocktime', 'xpos', 'thpos', 'exptime', 'scanid', 'framenum', 'det.ext']
        self.dask_tiff_stack = None
        self.transformed_stack = None
        self.integrated_stack = None
        self.maskdata = maskdata
        self.correctSolidAngle = correctSolidAngle
        self.polarization_factor = polarization_factor
        self.dark = dark
        self.flat = flat

    def open_folder_dialog(self, message, extension):
        app = QApplication([])
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(None, message, "", f"All Files (*);;{extension} Files (*.{extension})", options=options)
        if file_path == '':
            return None
        if not file_path.lower().endswith(f'.{extension}'):
            print(f"Invalid file format. Please select a '{extension}' file.")
            return None
        app.exit()

        # Check if a file was selected
        if file_path:
            return file_path
        else:
            return None

    def select_folder_dialog(self):
        folder_dialog = QtWidgets.QFileDialog()
        folder_path = folder_dialog.getExistingDirectory(
            None, "Select Folder", "", QtWidgets.QFileDialog.ShowDirsOnly
        )
        return folder_path
    
    def open_tiff_dialog(self):
        return self.open_folder_dialog("Select TIFF (.tiff) Data File", 'tiff')

    def open_poni_dialog(self):
        self.poni_file = self.open_folder_dialog("Select PONI (.poni) File", 'poni')

    def open_mask_dialog(self):
        mask_file = self.open_folder_dialog("Select Mask (.edf) File", 'edf')

        # Check if a file was selected
        if mask_file is not None:
            self.mask_file = mask_file

    def extract_tiff(self, tiff_path):
        filename = os.path.split(tiff_path)[0] + '.tiff'
        data = fabio.open(filename).data
        return data

    def get_geometry(self):
        # Create a PyQt5 application
        app = QApplication(sys.argv)

        # Create a QDialog for the popup window
        dialog = QDialog()
        dialog.setWindowTitle("Variable Input")
        layout = QVBoxLayout(dialog)

        # Create labels and line edits for each variable
        variable_labels = {
            "Incident Angle (deg.)": "incident_angle",
            "Wobble (deg.)": "incident_angle_wobble",
            "SDD (mm)": "sample_detector_distance",
            "Exposure Time (s)": "exp_time",
            "Frames (#)": "num_frames",
            "Rotation (#)": "sample_rotation",
            "Polarization Factor (-1 to 1)": "polarization_factor"
        }

        line_edits = {}

        for label_text, variable_name in variable_labels.items():
            label = QLabel(label_text)
            line_edit = QLineEdit()
            layout.addWidget(label)
            layout.addWidget(line_edit)

            # Store the line edit reference in the line_edits dictionary
            line_edits[variable_name] = line_edit

        # Create the OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)

        # Handle button clicks
        def handle_button_click(button):
            if button == button_box.button(QDialogButtonBox.Ok):
                dialog.accept()
            else:
                dialog.reject()

        button_box.clicked.connect(handle_button_click)

        # Execute the dialog and retrieve the entered values
        if dialog.exec_() == QDialog.Accepted:
            geometry_dict = {}
            for variable_name, line_edit in line_edits.items():
                geometry_dict[variable_name] = line_edit.text()

            # Convert the variable types
            geometry_dict["incident_angle"] = float(geometry_dict["incident_angle"])
            geometry_dict["incident_angle_wobble"] = float(geometry_dict["incident_angle_wobble"])
            geometry_dict["sample_detector_distance"] = float(geometry_dict["sample_detector_distance"])
            geometry_dict["exp_time"] = float(geometry_dict["exp_time"])
            geometry_dict["num_frames"] = int(geometry_dict["num_frames"])
            geometry_dict["sample_rotation"] = int(geometry_dict["sample_rotation"])
            geometry_dict["polarization_factor"] = float(geometry_dict["polarization_factor"])

            return geometry_dict

        # Close the application
        app.quit()

        return None

    def mine_tiff_md(self, data_path, keylist, delimiter="_"):
        # tiff_dict = {} # dictionary corresponding to the imported TIFF that contains the metadata 
        dataname = os.path.splitext(data_path) [0] + '.tiff'
        base_dataname = os.path.basename(dataname) # gets the base of the dataname from the datapath
        minedValList = base_dataname.split(delimiter) # split the base_dataname using the input delimiter - default set to "_"

        tiff_dict = dict(zip(keylist, minedValList)) # pair the mined values with the input keylist

        tiff_dict['samplename'] = base_dataname # store the basename
        tiff_dict['path'] = data_path # store the datapath

        clocktime = tiff_dict['clocktime']
        clocktime = re.findall("\d+\.\d+", clocktime)
        clocktime = clocktime[0]
        tiff_dict['clocktime'] = clocktime

        xpos = tiff_dict['xpos']
        xpos = re.findall("\d+\.\d+", xpos)
        xpos = xpos[0]
        tiff_dict['xpos'] = xpos

        thpos = tiff_dict['thpos']
        thpos = re.findall("\d+\.\d+", thpos)
        thpos = thpos[0]
        tiff_dict['thpos'] = thpos

        exptime = tiff_dict['exptime']
        exptime = re.findall("\d+\.\d+", exptime)
        exptime = exptime[0]
        tiff_dict['exptime'] = exptime

        return tiff_dict

    def create_detector(self):
        # - Create the detector object.
        pyg = self.pyg

        # - Load the PONI file into the transform object
        pyg.load(self.poni_file)

        # - Set the sample orientation
        sample_orientation = self.geometry_dict['sample_rotation']
        if sample_orientation is None:
            sample_orientation = 3

        pyg.sample_orientation = sample_orientation

        # - Set the incident_angle
        incident_angle = self.geometry_dict['incident_angle']

        if incident_angle is None or np.isnan(incident_angle) or incident_angle <= 0:
            incident_angle = self.tiff_dict['thpos']
            self.geometry_dict['incident_angle'] = incident_angle

        self.pyg.incident_angle = float(incident_angle)

        # - Extract the mask data from the mask_file
        maskdata = fabio.open(self.mask_file).data

        self.pyg = pyg
        self.maskdata = maskdata

    # @delayed
    def load_tiff(self, filepath):
        tiff_data = self.extract_tiff(filepath)
        tiff_md = self.mine_tiff_md(filepath, keylist=['solutionnum', 'composition', 'filtopt','molarity', 'purgerate', 'sub', 'solvol', 'sampnum', 'clocktime', 'xpos', 'thpos', 'exptime', 'scanid', 'framenum', 'det.ext'])
        return tiff_data

    def import_tiffs(self, folder_path):
        # folder_path = self.open_tiff_dialog()
        file_list = glob.glob(os.path.join(folder_path, self.wcard + '.tiff'))
        tiff_arrays = [self.load_tiff(filepath) for filepath in file_list]
        tiff_arrays = compute(*tiff_arrays)  # Trigger computation
        tiff_dask_arrays = [da.from_array(arr, chunks=(1000, 1000)) for arr in tiff_arrays]  # Convert to Dask arrays
        self.dask_tiff_stack = da.stack(tiff_dask_arrays)  # Stack the Dask arrays and store to self.dask_tiff_stack

    
    def gixs_1D_azi_int(self, caked_data, qr, chi, chilims = [-90, 90], qlims=[0, np.inf]):
        chimin, chimax = chilims
        qmin, qmax = qlims
        
        if qmax <= 0:
            qmax = np.max(qr)

        if qmin >= qmax:
            qmin = 0 

        int2D_nparray = np.zeros_like(caked_data)

        for row in range(caked_data.shape[0]):
            for col in range(caked_data.shape[1]):
                if qmin <= qr[col] <= qmax and chimin <= chi[row] <= chimax:
                    int2D_nparray[row, col] = caked_data[row, col]
                if caked_data[row, col] < 1:
                    int2D_nparray[row, col] = np.nan
        
        int2D_data = np.nanmean(int2D_nparray, axis = 0)

        qr = qr[int2D_data != 0]
        int2D_data = int2D_data[int2D_data != 0]

        int1D_data = np.stack([qr, int2D_data], axis = 1)
        int1D_data = np.transpose(int1D_data)

        qr_int1D = int1D_data[0]
        chi_int1D = int1D_data[1]

        return qr_int1D, chi_int1D

    def integrate_images(self):
        reshaped_stack = self.transformed_stack.reshape(-1, self.transformed_stack.shape[-2], self.transformed_stack.shape[-1])
        reshaped_qr = self.qr_stack.reshape(-1, self.qr_stack.shape[-2], self.qr_stack.shape[-1])
        reshaped_chi = self.chi_stack.reshape(-1, self.chi_stack.shape[-2], self.chi_stack.shape[-1])

        self.integrated_stack = da.map_blocks(self.gixs_1D_azi_int, reshaped_stack, reshaped_qr, reshaped_chi,
                                            dtype=reshaped_stack.dtype)

        self.integrated_stack = self.integrated_stack.compute()

    def gixs_2D_caked(self, data):
        caked_data, qr, chi = self.pyg.transform_image(data, process='polar',
                                                method = 'bbox',
                                                unit='q_A^-1',
                                                mask=self.maskdata, 
                                                correctSolidAngle = self.correctSolidAngle, 
                                                polarization_factor=self.polarization_factor, 
                                                dark=self.dark, 
                                                flat=self.flat)
        return caked_data, qr, chi

    def transform_images(self):
        # Update the method to use self.dask_tiff_stack instead of self.stack
        caked_stack, qr_stack, chi_stack = da.map_blocks(self.gixs_2D_caked, self.dask_tiff_stack, 
                                                        dtype=self.dask_tiff_stack.dtype).compute()
        self.transformed_stack = caked_stack
        self.qr_stack = qr_stack
        self.chi_stack = chi_stack

    def gixs_2D_recip(self, data):
        recip_data, qxy, qz = self.pyg.transform_reciprocal(data, # Convert detector image to q-space (sample reciprocal)
                                                method = 'bbox',
                                                unit='q_A^-1',
                                                mask=self.maskdata, 
                                                correctSolidAngle = self.correctSolidAngle, 
                                                polarization_factor=self.polarization_factor, 
                                                dark=self.dark, 
                                                flat=self.flat)
        return recip_data, qxy, qz

    def transform_images_recip(self):
        # New method to use gixs_2D_recip for transformation
        recip_stack, qxy_stack, qz_stack = da.map_blocks(self.gixs_2D_recip, self.dask_tiff_stack, 
                                                         dtype=self.dask_tiff_stack.dtype).compute()
        self.transformed_stack_recip = recip_stack
        self.qxy_stack = qxy_stack
        self.qz_stack = qz_stack

    def generate_heatmap(self):
        # Check if self.integrated_stack and self.geometry_dict are not None
        if self.integrated_stack is None or self.geometry_dict is None:
            print("Please make sure to run the transformation and integration first.")
            return

        # Generate the timestamp array
        exposure_time = self.geometry_dict.get('exp_time', 1)
        total_images = self.integrated_stack.shape[0]  # Assuming integrated_stack is a 2D numpy array
        timestamps = np.arange(0, total_images*exposure_time, exposure_time)

        # Plot the heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        c = ax.pcolormesh(timestamps, np.arange(self.integrated_stack.shape[1]), self.integrated_stack, cmap='viridis')
        fig.colorbar(c, ax=ax)
        ax.set_title('2D Heatmap of Integrated Image Data')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('q (Å^-1)')
        plt.show()

@property
def tiff_dict(self):
    if self._tiff_dict is None:
        raise ValueError("Tiff metadata has not been loaded yet.")
    return self._tiff_dict

@property
def geometry_dict(self):
    if self._geometry_dict is None:
        raise ValueError("Geometry data has not been loaded yet.")
    return self._geometry_dict

# -- Get Image Correction Information (Secondary)
class GIXSImageCorrections(QDialog):
    def __init__(self, parent=None):
        super(GIXSImageCorrections, self).__init__(parent)

        # Initialize dictionary
        self.gixsimgcorrections = {
            'correctSolidAngle': False,
            'polarization_factor': float('nan'),
            'dark_path': 'None',
            'flatfield_path': 'None'
        }

        # Set up GUI elements
        self.setWindowTitle("GIXS Image Corrections")

        self.layout = QVBoxLayout()

        self.label_correctSolidAngle = QLabel("Solid Angle Correction")
        self.correctSolidAngle = QRadioButton()
        self.layout.addWidget(self.label_correctSolidAngle)
        self.layout.addWidget(self.correctSolidAngle)

        self.label_polarization_factor = QLabel("Polarization Factor")
        self.polarization_factor = QLineEdit()
        self.layout.addWidget(self.label_polarization_factor)
        self.layout.addWidget(self.polarization_factor)

        self.label_dark_path = QLabel("Dark Detector Image")
        self.dark_path = QLineEdit()
        self.select_dark_path = QPushButton("File Select")
        self.select_dark_path.clicked.connect(self.update_dark_path_dialog)
        self.layout.addWidget(self.label_dark_path)
        self.layout.addWidget(self.dark_path)
        self.layout.addWidget(self.select_dark_path)

        self.label_flatfield_path = QLabel("Flatfield Detector Image")
        self.flatfield_path = QLineEdit()
        self.select_flatfield_path = QPushButton("File Select")
        self.select_flatfield_path.clicked.connect(self.update_flatfield_path_dialog)
        self.layout.addWidget(self.label_flatfield_path)
        self.layout.addWidget(self.flatfield_path)
        self.layout.addWidget(self.select_flatfield_path)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_and_close)
        self.layout.addWidget(self.apply_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.close)
        self.layout.addWidget(self.cancel_button)

        self.setLayout(self.layout)

    def update_correctSolidAngle(self):
        self.gixsimgcorrections['correctSolidAngle'] = self.correctSolidAngle.isChecked()

    def update_polarization_factor(self):
        text = self.polarization_factor.text()
        try:
            value = float(text)
            if -1.0 <= value <= 1.0:
                self.gixsimgcorrections['polarization_factor'] = value
        except ValueError:
            self.gixsimgcorrections['polarization_factor'] = float('nan')

    def update_dark_path_dialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if fname:
            self.dark_path.setText(fname)
            self.gixsimgcorrections['dark_path'] = fname
        else:
            self.gixsimgcorrections['dark_path'] = 'None'

    def update_flatfield_path_dialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if fname:
            self.flatfield_path.setText(fname)
            self.gixsimgcorrections['flatfield_path'] = fname
        else:
            self.gixsimgcorrections['flatfield_path'] = 'None'

    def apply_and_close(self):
        self.update_correctSolidAngle()
        self.update_polarization_factor()
        self.close()