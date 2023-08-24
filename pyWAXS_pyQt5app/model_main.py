# -- Import Model() Subclasses -- #
from waxscorr_class import WAXS_CORR
from waxssing_class import WAXS_SING
# from waxssim_class import WAXS_SIM
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

class Model:
    def __init__(self):
        self.waxs_sing = WAXS_SING()

    def load_image(self, file_name):
        self.waxs_sing.load_image(file_name)
        return self.waxs_sing.get_image_data()

# -------------------------------------------------------------------------------------------------------- #
# -- SKELETON BELOW -- #
# -------------------- #

# def apply_corrections(self, WAXS_CORR_instance):
#     self.WAXS_data.apply_corrections(WAXS_CORR_instance)

# def integrate_azimuthal_1D(self, roi):
#     self.WAXS_data.integrate_azimuthal_1D(roi)

# def integrate_boxcut_1d(self, roi):
#     self.WAXS_data.integrate_boxcut_1d(roi)

# def generate_pole_figure(self):
#     self.WAXS_data.plot_pole_figures()

# -------------------------------------------------------------------------------------------------------- #
# -- CODE SNIPPETS BELOW -- #
# -------------------------------------------------------------------------------------------------------- #
# Define additional methods as needed based on what actions you'll be performing on the WAXS_CORR() and WAXS_SING() instances.
# class Model():
#     def __init__(self):
#         self.waxs_corr = WAXS_CORR()
#         self.waxs_tiffdata = WAXS_SING()
    
#     def load_image(self, file_name):
#         self.waxs_tiffdata.load_image(file_name)
#         return self.waxs_tiffdata.image_data
        # return self.waxs_tiffdata.get_current_image()

    # def get_current_image(self):
    #     tiff_data = self.waxs_tiffdata.get_current_image()
    #     return tiff_data

