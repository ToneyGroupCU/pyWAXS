# -- Import Model/View Classes to the Controller() Class
from model_main import Model
from view_main import View
# -- General Packages -- #
import os, sys, re, fabio
import glob2 as glob
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from zipfile import ZipFile
# -- PyQt5 Packages -- #
from PyQt5 import QtWidgets, QtGui, QtCore
import PyQt5.QtWidgets as qtw
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QComboBox, QLabel, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QPlainTextEdit
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import pyqtgraph as pyqt
from pyqtgraph import PlotDataItem, PlotWidget
from plotly.graph_objs import FigureWidget
# -- Matplotlib Packages -- #
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# -- PyFAI Packages -- #
import pyFAI
from pyFAI.detectors import Detector
# -- pygix package -- #
# import pygix as pg

class Controller:
    def __init__(self, model=None, view=None):
        self.model = model if model is not None else Model()
        self.view = view if view is not None else View()
        # self.view.load_image_signal.connect(self.load_image)
        self.connect_signals()

    def load_image(self, file_path):
        tiff_data = self.model.load_image(file_path) 
        self.view.display_image(tiff_data) 
        self.view.update_file_label(file_path) 

    def display_image(self, file_name):
        # we don't need to load the image again here
        image_data = self.model.get_current_image() 
        self.view.display_image(image_data)
        self.view.update_file_label(file_name)
        self.view.update_calibration_widget(image_data)

    def connect_signals(self):
        # Connect all of the signals from the View() to the Controller() - call in the __init__ dunder method for the Controller().
        # Signal 1: Connect on_load_image_button_clicked from View to load_image slot in Controller
        self.view.load_image_signal.connect(self.load_image)

    # -------------------------------------------------------------------------------------------------------- #
    # -- SKELETON BELOW -- #
    # -------------------- #
    
    # Define the remaining methods here as discussed previously
    def load_poni_file(self, filepath):
        # Loads the PONI file and sends it to the model
        pass

    def load_mask_file(self, filepath):
        # Loads the Mask file and sends it to the model
        pass

    def load_calibrant_data(self, filepath):
        # Loads the calibrant data and sends it to the model
        pass

    def generate_poni_file(self, user_input):
        # Interacts with the model to generate the PONI file
        pass

    def generate_mask_file(self, user_input):
        # Interacts with the model to generate the Mask file
        pass

    def save_detector_params(self, user_input):
        # Sends the detector parameters to the model
        pass

# -------------------------------------------------------------------------------------------------------- #
# -- CODE SNIPPETS BELOW -- #
# -------------------------------------------------------------------------------------------------------- #
# class Controller:
#     def __init__(self, model=None, view=None):
#         self.model = model if model is not None else Model()
#         self.view = view if view is not None else View()
#         self.view.load_image_signal.connect(self.load_image)

#     # @pyqtSlot(str)
#     def load_image(self, file_path):
#         tiff_data = self.model.load_image(file_path) # get the image file reference from the Model()
#         # self.view.display_tiff_data(tiff_data) # send the TIFF data to the View()
#         self.view.display_image(tiff_data) # tell the View() to update with the newly received TIFF data
#         self.view.update_file_label(file_path) # Update the file label in the View() to reflect the file name of the TIFF data.
        
#         # Create the file path label to display the filename
#         # self.view.file_label.setText(f'Loaded file: {file_path}')

#     def display_image(self, file_name):
#         image_data = self.model.load_image(file_name) # Controller() --> Model() --> WAXS_SIM() --> load_image(): converts .tiff to numpy array
#         # image_data = self.model.get_current_image() 
#         # image_data = self.model.
#         self.view.display_image(image_data)
#         self.view.update_file_label(file_name)
#         self.view.update_calibration_widget(image_data)