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

class View(qtw.QMainWindow):
    # Define the emitters for signals to be passed from UI to Controller()
    load_image_signal = pyqtSignal(str)
    current_image_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.init_ui()
        # ------------------------------ #
        # -- MAIN LAYOUT: Create main layout as QV vertical box layout.
        self.layout = qtw.QVBoxLayout()

        # -- MAIN LAYOUT: Create primary tabs using QTabWidget()
        self.primary_tabs = QtWidgets.QTabWidget()
        self.layout.addWidget(self.primary_tabs)

        # ------------------------------ #
        # -- "Single Image Processing" Panel: Add tabs to the primary tab.
        self.primary_tabs.addTab(self.init_single_image_processing_tab(), "Single Image Processing")
        
        # When load image button is clicked, emit signal with file path
        self.load_image_button.clicked.connect(self.on_load_image_button_clicked)

    def init_ui(self):
        # ------------------------------ #
        # -- MAIN LAYOUT: Set up main window properties
        self.setWindowTitle("WAXS Analysis")
        self.setGeometry(100, 100, 1600, 1200)

        # -- MAIN LAYOUT: Create central widget and layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # -- MAIN LAYOUT: Create primary tabs
        self.primary_tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.primary_tabs)

        # -- MAIN LAYOUT: Add tabs to primary tabs
        self.primary_tabs.addTab(self.init_calibration_tab(), "Calibration")
        self.primary_tabs.addTab(self.init_single_image_processing_tab(), "Single Image Processing")
        self.primary_tabs.addTab(self.init_single_image_analysis_tab(), "Single Image Analysis")
        self.primary_tabs.addTab(self.init_multi_image_processing_tab(), "Multi-Image Processing")
        self.primary_tabs.addTab(self.init_waxs_sim_tab(), "WAXS Sim")
        self.primary_tabs.addTab(self.init_waxs_orientation_analysis_tab(), "WAXS Orientation Analysis")
        self.primary_tabs.addTab(self.init_waxs_phase_analysis_tab(), "WAXS Phase Analysis")
        self.primary_tabs.addTab(self.init_settings_tab(), "Settings")
        # self.setCentralWidget(self.main_tabs)

        self.show()

    # ------------------------------------------------------------------------------------------------------------------------------------------------ #
    # -- GENERAL METHODS -- #
    # -------------------------------------------------------------- #
    # General Method 2: Method used the manipulate the color map of an imported TIFF image.
    def change_color_map(self):
        current_color_map = self.color_map_combo.currentText()
        color_map = pyqt.colormap.get(current_color_map)
        self.plot_widget.setColorMap(color_map)
    # -------------------------------------------------------------- #

    # -------------------------------------------------------------- #
    # General Method 3: 
    def update_file_label(self, file_path):
        self.file_label.setText(f"Loaded File: {file_path}")
    # -------------------------------------------------------------- #

    # -------------------------------------------------------------- #
    # General Method 5: 
    def display_image(self, image_data):
        self.tiff_data = image_data
        self.plot_widget.setImage(image_data.T)
    # ------------------------------------------------------------------------------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------------------------------------------------------------------------------ #
    # -- (Primary Tab Initialization) Method 1: Initialize "Single Image Processing" tab.
    # -------------------------------------------------------------- #
    def init_single_image_processing_tab(self):
        # Create Single Image Processing primary tab
        single_image_processing_tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(single_image_processing_tab)

        # Create secondary tabs
        secondary_tabs = QtWidgets.QTabWidget()
        layout.addWidget(secondary_tabs)

        # Add tabs to secondary tabs
        secondary_tabs.addTab(self.init_load_image_tab(), "Load Image")
        secondary_tabs.addTab(self.init_apply_corrections_tab(), "Apply Corrections")
        secondary_tabs.addTab(self.init_azimuthal_1d_integration_tab(), "Azimuthal 1D Integration")
        secondary_tabs.addTab(self.init_boxcut_1d_integration_tab(), "Boxcut 1D Integration")
        secondary_tabs.addTab(self.init_pole_figure_generation_tab(), "Pole Figure Generation")

        single_image_processing_tab.setLayout(layout)

        return single_image_processing_tab
    # -------------------------------------------------------------- #

    # -------------------------------------------------------------- #
    # -- (Secondary Tab Initialization) Method 1: Initialize "Load Image" tab.
    def init_load_image_tab(self):
        # -------------------- #
        # DEFINE THE LAYOUT
        load_image_tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(load_image_tab)

        # -------------------- #
        # DEFINE THE WIDGETS
        # - Widget 1: Create the Load TIFF Push Button widget - #
        self.load_image_button = QtWidgets.QPushButton("Load TIFF")
        self.load_image_button.clicked.connect(self.on_load_image_button_clicked)

        # - Widget 2: Create the file label widget - #
        self.file_label = QtWidgets.QLabel('No file selected.')
        self.file_label.setWordWrap(True)

        # - Widget 3: Create the color map label dropdown widget - #
        # Add color map label
        self.color_map_label = QtWidgets.QLabel('Color Map')
        layout.addWidget(self.color_map_label)

        # Add color map combo box
        self.color_map_combo = QtWidgets.QComboBox()
        self.color_map_combo.addItem('viridis')
        self.color_map_combo.addItem('plasma')
        self.color_map_combo.addItem('inferno')
        self.color_map_combo.addItem('magma')
        self.color_map_combo.addItem('cividis')
        self.color_map_combo.currentIndexChanged.connect(self.change_color_map)

        # - Widget 4: Create the plot widget
        self.plot_widget = pyqt.ImageView()

        # -------------------- #
        # ADD WIDGETS TO THE LAYOUT AFTER DEFINING THE LAYOUT
        layout.addWidget(self.load_image_button) # Widget 1
        layout.addWidget(self.file_label) # Widget 2
        layout.addWidget(self.color_map_combo) # Widget 3
        layout.addWidget(self.plot_widget) # Widget 4

        return load_image_tab

    # "Load Image" Method 1A: Define the slot for the on_load_image_button_clicked() method.
    @pyqtSlot()
    def on_load_image_button_clicked(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load TIFF", "", "TIFF Files (*.tif *.tiff);;All Files (*)")
        if file_path:
            self.load_image_signal.emit(file_path)
    # -------------------------------------------------------------- #

    # -------------------------------------------------------------- #
    # -- (Secondary Tab Initialization) Method 2: Initialize "Apply Corrections" tab.
    def init_apply_corrections_tab(self):
        apply_corrections_tab = QtWidgets.QWidget()

        # Add buttons, LineEdits, and PlotItem (from pyqtgraph) here
        # Define layout
        # Add 'plotly' interactive 2D plot window
        # Add 'Select PONI' and 'Select Mask' buttons
        # Add output display field for filenames
        # Add grid of input fields and radio buttons for correction factors

        return apply_corrections_tab
    
    # -------------------------------------------------------------- #
    # -- (Secondary Tab Initialization) Method 3: Initialize "1D Azimuthal Integrations" tab.
    def init_azimuthal_1d_integration_tab(self):
        azimuthal_1d_integration_tab = QtWidgets.QWidget()

        # Add PlotItems (from pyqtgraph), LineEdits, a ROI (from pyqtgraph), and a button here
        
        # Define layout
        # Add 'plotly' interactive 2D plot window
        # Add widget for selecting regions of interest
        # Add second output plot for 1D integration
        # Add buttons to save 1D integration

        return azimuthal_1d_integration_tab
    
    # -------------------------------------------------------------- #
    # -- (Secondary Tab Initialization) Method 4: Initialize "Boxcut 1D Integrations" tab. 
    def init_boxcut_1d_integration_tab(self):
        boxcut_1d_integration_tab = QtWidgets.QWidget()

        # Add PlotItems (from pyqtgraph), LineEdits, a ROI (from pyqtgraph), and a button here
        # Define layout
        # Add 'plotly' interactive 2D plot window
        # Add widget for selecting regions of interest
        # Add second output plot for 1D integration
        # Add buttons to save 1D integration

        return boxcut_1d_integration_tab
    
    # -------------------------------------------------------------- #
    # -- (Secondary Tab Initialization) Method 5: Initialize "Pole Figure Generation" tab. 
    def init_pole_figure_generation_tab(self):
        pole_figure_generation_tab = QtWidgets.QWidget()

        # Add PlotItems (from pyqtgraph), LineEdits, a ROI (from pyqtgraph), and a button here
        # Define layout
        # Add 'plotly' interactive 2D plot window
        # Add widget for selecting regions of interest
        # Add second output plot for 1D integration
        # Add buttons to save 1D integration
        return pole_figure_generation_tab

    # ------------------------------------------------------------------------------------------------------------------------------------------------ #
    
    # ------------------------------------------------------------------------------------------------------------------------------------------------ #    
    # -- (Primary Tab Initialization) Method 2: Initialize "Calibration" tab.
    def init_calibration_tab(self):
        tab = QtWidgets.QWidget()
        # set up the secondary tabs for the "Calibration" tab
        # ...
        return tab

    # ------------------------------------------------------------------------------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------------------------------------------------------------------------------ #
    # -- (Primary Tab Initialization) Method 3: Initialize "Single Image Analysis" tab.
    def init_single_image_analysis_tab(self):
        tab = QtWidgets.QWidget()
        # set up the secondary tabs for the "Single Image Analysis" tab
        # ...
        return tab

    # ------------------------------------------------------------------------------------------------------------------------------------------------ #
    
    # ------------------------------------------------------------------------------------------------------------------------------------------------ #
    # -- (Primary Tab Initialization) Method 3: Initialize "Multi-Image Processing" tab.
    def init_multi_image_processing_tab(self):
        tab = QtWidgets.QWidget()
        # set up the secondary tabs for the "Multi-Image Processing" tab
        # ...
        return tab

    # ------------------------------------------------------------------------------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------------------------------------------------------------------------------ #
    # -- (Primary Tab Initialization) Method 4: Initialize "WAXS Simulation" tab.
    def init_waxs_sim_tab(self):
        tab = QtWidgets.QWidget()
        # set up the secondary tabs for the "WAXS Sim" tab
        # ...
        return tab

    # ------------------------------------------------------------------------------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------------------------------------------------------------------------------ #
    # -- (Primary Tab Initialization) Method 5: Initialize "Orientation Analysis" tab.
    def init_waxs_orientation_analysis_tab(self):
        tab = QtWidgets.QWidget()
        # set up the secondary tabs for the "WAXS Orientation Analysis" tab
        # ...
        return tab

    # ------------------------------------------------------------------------------------------------------------------------------------------------ #
    
    # ------------------------------------------------------------------------------------------------------------------------------------------------ #
    # -- (Primary Tab Initialization) Method 6: Initialize "WAXS Phase Analysis" tab.
    def init_waxs_phase_analysis_tab(self):
        tab = QtWidgets.QWidget()
        # set up the secondary tabs for the "WAXS Phase Analysis" tab
        # ...
        return tab
    
    # ------------------------------------------------------------------------------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------------------------------------------------------------------------------ #
    # -- (Primary Tab Initialization) Method 7: Initialize "Settings" tab.
    def init_settings_tab(self):
        tab = QtWidgets.QWidget()
        # set up the secondary tabs for the "Settings" tab
        # ...
        return tab
    # ------------------------------------------------------------------------------------------------------------------------------------------------ #


# -------------------------------------------------------------------------------------------------------- #
# -- CODE SNIPPETS BELOW -- #
# -------------------------------------------------------------------------------------------------------- #

# Add additional methods to create secondary panels (create_load_image_panel(), create_apply_corrections_panel(), etc.) here...
# '''
    # class SecondaryTab:
    #     def __init__(self):
    #         # Set up shared layout, widgets, etc.
    #         self.setup_layout()
    #         self.add_common_widgets()

    #     def setup_layout(self):
    #         # Set up common layout here
    #         pass

    #     def add_common_widgets(self):
    #         # Add common widgets here
    #         pass

    # class SpecificSecondaryTab(SecondaryTab):
    #     def __init__(self):
    #         super().__init__()
    #         # Add unique elements for this specific tab
    #         self.add_unique_widgets()

    #     def add_unique_widgets(self):
    #         # Add unique widgets for this tab
    #         pass
    
    # # Initialize all secondary tabs
    # self.secondary_tabs = {
    #     "Tab1": SpecificSecondaryTab(),
    #     # "Tab2": AnotherSpecificSecondaryTab(),
    #     # etc.
    # }
# '''
    # General Method 1: Displays TIFF data with a transpose applied to the input data.
    # def display_tiff_data(self, tiff_data):
    #     self.plot_widget.setImage(tiff_data.T)
    # -------------------------------------------------------------- #


    # -------------------------------------------------------------- #
    # General Method 4: 
    # def get_current_image(self):
    #     self.current_image_requested.emit()
    # -------------------------------------------------------------- #