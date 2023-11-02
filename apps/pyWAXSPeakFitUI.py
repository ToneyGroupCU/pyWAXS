from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, QWidget, QFrame, QVBoxLayout, QHBoxLayout, QAction, QFileDialog, QToolBar, QPushButton, QSizePolicy, QSlider, QMessageBox, QTableWidget, QTableWidgetItem, QLabel, QLineEdit, QGridLayout, QAbstractItemView
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QWindow

import sys, os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.text import Text
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.gridspec import GridSpec

from scipy.interpolate import griddata
import lmfit
from lmfit.lineshapes import gaussian2d, lorentzian
from lmfit.models import Model

script_dir = os.path.dirname(os.path.abspath(__file__)) # Get the directory of the current script
parent_dir = os.path.dirname(script_dir) # Get the parent directory
# main_dir = os.path.join(parent_dir, 'main') # Construct the path to the /main/ directory
main_dir = os.path.join(parent_dir, 'pywaxs') # Construct the path to the /main/ directory
sys.path.append(main_dir) # Add the /main/ directory to sys.path

class XrayApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ds = None  # Initialize dataset to None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('X-ray Data Analysis')
        self.setGeometry(100, 100, 1600, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Left frame and layout (50%)
        self.left_frame = QFrame()
        self.left_layout = QVBoxLayout()
        self.left_frame.setLayout(self.left_layout)
        self.layout.addWidget(self.left_frame, stretch=1)

        # Main frame for X-ray plot (canvas1)
        self.main_frame = QFrame()
        self.main_layout = QVBoxLayout()
        self.main_frame.setLayout(self.main_layout)
        self.left_layout.addWidget(self.main_frame)

        self.fig1, self.ax1 = plt.subplots()
        self.ax1.axis('off')
        self.canvas1 = MyCanvas(self.fig1, main_app=self, add_subplot=True)
        self.main_layout.addWidget(self.canvas1)

        self.nav1 = MyNavigationToolbar(self.canvas1, self)
        self.main_layout.insertWidget(0, self.nav1)

        # Right frame and layout (50%)
        self.output_frame = QFrame()
        self.output_layout = QVBoxLayout()
        self.layout.addWidget(self.output_frame, stretch=1)
        self.output_frame.setLayout(self.output_layout)

        self.fig2 = plt.figure(figsize=(8, 4))
        gs = GridSpec(2, 2, width_ratios=[3, 2])
        self.ax2 = self.fig2.add_subplot(gs[0, 0])
        self.ax3 = self.fig2.add_subplot(gs[1, 0])
        self.ax4 = self.fig2.add_subplot(gs[0, 1])
        self.ax5 = self.fig2.add_subplot(gs[1, 1])
        self.ax3.axis('off')

        axes_dict = {'ax2': self.ax2, 'ax3': self.ax3, 'ax4': self.ax4, 'ax5': self.ax5}
        self.canvas2 = MyCanvas(self.fig2, main_app=self, add_subplot=False, axes=axes_dict)
        self.nav2 = GaussianFitToolbar(self.canvas2, self)
        self.nav2.setFixedWidth(600)
        self.canvas2.ax2.set_title('Active ROI')

        self.output_layout.addWidget(self.nav2)
        self.output_layout.addWidget(self.canvas2, stretch=2)  # 2/3 of the vertical space

        # Stat boxes (1/3 of the vertical space)
        self.widget3 = QWidget()
        self.grid3 = QGridLayout()
        self.widget3.setLayout(self.grid3)
        self.add_stat_box('Amplitude', 1, 0)
        self.add_stat_box('Center X', 2, 0)
        self.add_stat_box('Center Y', 3, 0)
        self.add_stat_box('Chi-square', 1, 1)
        self.add_stat_box('Reduced Chi-square', 2, 1)
        self.add_stat_box('R-squared', 3, 1)
        self.add_stat_box('Sigma X', 1, 2)
        self.add_stat_box('Sigma Y', 2, 2)
        self.add_stat_box('FWHM X', 3, 2)
        self.add_stat_box('FWHM Y', 4, 2)

        self.output_layout.addWidget(self.widget3, stretch=1)

        # Initialize the table with additional columns for fit statistics
        self.roi_table = QTableWidget()
        self.header_labels = ['ROI #', 'q_xy (min)', 'q_xy (max)', 'q_z (min)', 'q_z (max)', 
                            'Amplitude', 'Center X', 'Center Y', 'Sigma X', 'Sigma Y',
                            'Chi-square', 'Reduced Chi-square', 'R-squared', 'FWHM X', 'FWHM Y']
        self.roi_table.setColumnCount(len(self.header_labels))
        self.roi_table.setHorizontalHeaderLabels(self.header_labels)
        self.roi_table.setWordWrap(True)
        self.roi_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.main_layout.addWidget(self.roi_table)

        self.roi_selector = None
        self.coords = None

        # Add table to the layout
        self.roi_table.cellClicked.connect(self.on_table_click)
   
    def load_project(self):
        # Check if a project is already loaded
        if hasattr(self, 'ds') and self.ds is not None:
            reply = QMessageBox.question(self, 'Save Current Project', 
                                        'Would you like to save the current project before loading a new one?', 
                                        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, 
                                        QMessageBox.Cancel)

            if reply == QMessageBox.Yes:
                options = QFileDialog.Options()
                filepath, _ = QFileDialog.getSaveFileName(self, "Save Current Project Data", "", "NetCDF Files (*.nc);;All Files (*)", options=options)
                if filepath:
                    if not filepath.endswith('.nc'):
                        filepath += '.nc'
                    try:
                        # Assume you have a method called export_data that takes care of saving
                        self.canvas1.export_data(filepath)
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"An error occurred while saving: {e}", QMessageBox.Ok)
                        return  # If save fails, don't proceed to load a new project

            elif reply == QMessageBox.Cancel:
                return  # Don't proceed to load a new project

            # Clear the current dataset only if we are sure that a new project will be loaded
            # self.ds = None

        # Now you can call your loadData function here
        self.loadData()

    def loadData(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "Load Data", "", "NetCDF Files (*.nc);;All Files (*)", options=options)
        if file:
            try:

                self.canvas1.init_plot() # initialize canvas1
                # self.canvas2.init_plot()
                self.ds = None
                self.ds = xr.open_dataset(file, engine='h5netcdf')

                try:
                    if self.ds['intensity'] is None:  #or self.ds['peak_positions'] is None:
                            raise ValueError("Intensity or peak positions are None.")
                    
                    # Extract coordinate names
                    coord_names_intensity = list(self.ds['intensity'].dims)
                    coord_names_peak = list(self.ds['peak_positions'].dims)
                    
                    self.coords = {
                        'xlabel': coord_names_intensity[1] if len(coord_names_intensity) > 1 else None,  
                        'ylabel': coord_names_intensity[0] if len(coord_names_intensity) > 0 else None,
                    }

                    # Assuming `plot_data` is a method in your canvas object
                    self.canvas1.plot_data(self.ds['intensity'], self.ds['peak_positions'], self.coords)
                    self.roi_selector = self.canvas1.roi_selector
                except Exception as e:
                    QMessageBox.critical(self.window, "Error", f"An error occurred while plotting: {e}", QMessageBox.Ok)
        
                if self.ds:
                    max_intensity = self.ds['intensity'].max()
                    max_intensity = int(max_intensity.values)
                    # self.slider_vmin.setMaximum(max_intensity)
                    # self.slider_vmax.setMaximum(max_intensity)
            
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while loading the project: {e}", QMessageBox.Ok)

        print("Loaded dataset:", self.ds)
        print("Intensity dimensions:", self.ds['intensity'].dims)
        print("Peak_positions dimensions:", self.ds['peak_positions'].dims)

    def update_roi_table(self):
        print("Updating ROI table")  # Debugging print

        # print(f"ID of ROISelector in XrayApp: {id(self.roi_selector)}")
        # print(f"ID of rectangles in XrayApp: {id(self.roi_selector.rectangles)}")
        # print(f"ID of ROISelector in canvas1: {id(self.canvas1.roi_selector)}")
        # print(f"ID of rectangles in canvas1: {id(self.canvas1.roi_selector.rectangles)}")

        self.roi_table.setRowCount(0)  # Clear the table
        self.roi_selector = self.canvas1.roi_selector

        for i, rect in enumerate(self.roi_selector.rectangles):
            self.roi_table.insertRow(i)
            self.roi_table.setItem(i, 0, QTableWidgetItem(str(i)))
            
            # Retrieve the stored coordinates of the rectangle
            coords = rect.get('coords', {})
            x1, y1, x2, y2 = coords.get('x1', 0), coords.get('y1', 0), coords.get('x2', 0), coords.get('y2', 0)
            print(f"update_roi_table: (x1, y1):{x1:3.2f}, {y1:3.2f}) --> (*x2, y2):{x2:3.2f}, {y2:3.2f})")

            self.roi_table.setItem(i, 1, QTableWidgetItem(f"{x1:.2f}"))
            self.roi_table.setItem(i, 2, QTableWidgetItem(f"{x2:.2f}"))
            self.roi_table.setItem(i, 3, QTableWidgetItem(f"{y1:.2f}"))
            self.roi_table.setItem(i, 4, QTableWidgetItem(f"{y2:.2f}"))

    def on_table_click(self, row, column):
        # Reset the edge color of previously active rectangle, if any
        if self.canvas1.roi_selector.active_rect:
            self.canvas1.roi_selector.active_rect['main'].set_edgecolor('black')

        # Get the new active rectangle based on the clicked row in the table
        new_active_rect = self.canvas1.roi_selector.rectangles[row]

        # Set its edge color to blue to highlight it
        new_active_rect['main'].set_edgecolor('blue')

        # Update the active rectangle in the roi_selector
        self.canvas1.roi_selector.active_rect = new_active_rect

        # Redraw the canvas to reflect the changes
        self.canvas1.draw()

        # *** Modified part starts here ***
        # Check if the active rectangle exists before updating the sliced data array
        if self.canvas1.roi_selector.active_rect is not None:
            self.update_sliced_dataarray()
        else:
            self.canvas2.ax2.clear()
            if self.canvas2.colorbar is not None:
                ax = self.canvas2.colorbar.ax  # Get the axis associated with the colorbar
                if ax is not None:
                    self.canvas2.colorbar.remove()
                self.canvas2.colorbar = None  # Reset to None either way
            self.canvas2.draw()
        # *** Modified part ends here ***

    def update_sliced_dataarray(self):
        """Updates the sliced dataarray based on the coordinates of the selected ROI."""
        if self.canvas1.roi_selector.active_rect is None:
            self.canvas2.ax2.clear()
            if self.canvas2.colorbar is not None:
                ax = self.canvas2.colorbar.ax  # Get the axis associated with the colorbar
                if ax is not None:
                    self.canvas2.colorbar.remove()
                self.canvas2.colorbar = None  # Reset to None either way
            self.canvas2.draw()
            return
        # Retrieve the stored coordinates of the rectangle
        coords = self.canvas1.roi_selector.active_rect.get('coords', {})
        x1, y1, x2, y2 = coords.get('x1', 0), coords.get('y1', 0), coords.get('x2', 0), coords.get('y2', 0)
        
        # Map pixel positions to actual dataset coordinates
        dim1 = 'q_xy'
        dim2 = 'q_z'
        
        # - Uncomment if using pixel dimensions
        # q_xy_min = self.ds[dim1][int(np.floor(x1))].values
        # q_xy_max = self.ds[dim1][int(np.ceil(x2))].values
        # q_z_min = self.ds[dim2][int(np.floor(y1))].values
        # q_z_max = self.ds[dim2][int(np.ceil(y2))].values

        # - Use for q-space coordinates, not pixel dimensions.
        q_xy_min = min(x1, x2)
        q_xy_max = max(x1, x2)
        q_z_min = min(y1, y2)
        q_z_max = max(y1, y2)

        # Find the nearest actual dataset coordinates for each boundary
        q_xy_min_nearest = self.ds[dim1].sel({dim1: q_xy_min}, method='nearest').values
        q_xy_max_nearest = self.ds[dim1].sel({dim1: q_xy_max}, method='nearest').values
        q_z_min_nearest = self.ds[dim2].sel({dim2: q_z_min}, method='nearest').values
        q_z_max_nearest = self.ds[dim2].sel({dim2: q_z_max}, method='nearest').values

        # Use these nearest values to slice the data array
        self.sliced_ds = self.ds.sel(
            q_z=slice(min(q_z_min_nearest, q_z_max_nearest), max(q_z_min_nearest, q_z_max_nearest)), 
            q_xy=slice(min(q_xy_min_nearest, q_xy_max_nearest), max(q_xy_min_nearest, q_xy_max_nearest))
        )

        # Update ax2
        self.canvas2.ax2.clear()

        # Get extents based on actual dataset coordinates
        extent = [q_xy_min, q_xy_max, q_z_min, q_z_max]

        # Calculate vmin and vmax
        img_values = self.ds['intensity'].values
        vmin = np.nanpercentile(img_values, 10)
        vmax = np.nanpercentile(img_values, 99)

        im = self.canvas2.ax2.imshow(
            self.sliced_ds['intensity'].values, 
            cmap='turbo',
            origin='lower',
            extent=extent,
            aspect='auto',
            vmin=vmin, vmax=vmax
        )

        if self.canvas2.colorbar is not None:
            ax = self.canvas2.colorbar.ax  # Get the axis associated with the colorbar
            if ax is not None:
                self.canvas2.colorbar.remove()
            else:
                self.canvas2.colorbar = None  # Reset to None if ax is None

        if self.coords:
            xlabel = self.coords.get('xlabel', None)
            ylabel = self.coords.get('ylabel', None)
            if xlabel:
                self.canvas2.ax2.set_xlabel(xlabel)
            if ylabel:
                self.canvas2.ax2.set_ylabel(ylabel)

        self.canvas2.colorbar = self.canvas2.fig.colorbar(im, ax=self.canvas2.ax2)
        self.canvas2.draw()

        # *** Modified part starts here ***
        if self.canvas2.colorbar is not None:
            ax = self.canvas2.colorbar.ax  # Get the axis associated with the colorbar
            if ax is not None:
                self.canvas2.colorbar.remove()
            self.canvas2.colorbar = None  # Reset to None either way
        
        self.canvas2.ax2.set_title('Active ROI')

    def delete_active_roi(self):
        if self.canvas1.roi_selector.active_rect:
            self.canvas1.roi_selector.active_rect['main'].remove()
            self.canvas1.roi_selector.active_rect['close_box'].remove()
            self.canvas1.roi_selector.active_rect = None
            self.canvas2.ax2.clear()
            self.canvas2.draw()

    def add_stat_box(self, label, row, col):
        lbl = QLabel(label)
        le = QLineEdit(self)
        le.setReadOnly(True)
        setattr(self, f"stat_{label.replace(' ', '_').lower()}", le)  # Save as attribute
        self.grid3.addWidget(lbl, 2*row-1, col)
        self.grid3.addWidget(le, 2*row, col)

    def update_stat_boxes(self, fit_stats_dict):
        for key, value in fit_stats_dict.items():
            stat_box_name = f"stat_{key.lower()}"
            stat_box = getattr(self, stat_box_name, None)
            if stat_box is not None:
                stat_box.setText(str(value))

    def perform_gaussian_fit(self):
        fit_instance = Gaussian2DFit(self.sliced_ds)
        fit_instance.construct_model()
        fit_instance.perform_fit()
        
        # Print detailed fit report
        lmfit.report_fit(fit_instance.result)
        
        fit_instance.plot_fit(self.canvas2.axes['ax4'], self.canvas2.axes['ax5'])
        
        # Get the fit statistics
        fit_stats_dict = fit_instance.fit_statistics()
        
        # Update stat boxes
        self.update_stat_boxes(fit_stats_dict)

        # Identify the active ROI row
        # active_row = self.roi_selector  # Assuming roi_selector contains the index of the active ROI
        active_row = self.roi_table.currentRow()
        # Check if there are existing values and prompt user for overwrite
        if self.roi_table.item(active_row, 5) is not None:  # Assuming the first fit parameter is in column 5
            reply = QMessageBox.question(self, 'Overwrite Existing Values', 
                                        "Do you want to overwrite existing fit values for this ROI?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        # Update the table with new fit parameters
        for col, header in enumerate(self.header_labels[5:]):  # Starting from 'Amplitude'
            if header.replace(' ', '_').replace('-', '_').lower() in fit_stats_dict:
                new_item = QTableWidgetItem(str(fit_stats_dict[header.replace(' ', '_').replace('-', '_').lower()]))
                self.roi_table.setItem(active_row, col+5, new_item)  # col+5 because we start from 'Amplitude' which is at index 5
                
    # def perform_gaussian_fit(self):
    #     fit_instance = Gaussian2DFit(self.sliced_ds)
    #     fit_instance.construct_model()
    #     fit_instance.perform_fit()
        
    #     # Print detailed fit report
    #     lmfit.report_fit(fit_instance.result)
        
    #     fit_instance.plot_fit(self.canvas2.axes['ax4'], self.canvas2.axes['ax5'])
        
    #     # Get the fit statistics
    #     stats = fit_instance.fit_statistics()
    #     # self.display_fit_statistics(stats)

    #     # Get fit statistics
    #     fit_stats_dict = fit_instance.fit_statistics()
        
    #     # Update stat boxes
    #     self.update_stat_boxes(fit_stats_dict)

        # # Find the active row in the table (assuming it's selected)
        # active_row = self.roi_table.currentRow()

        # if active_row != -1:  # A row is selected
        #     # Check for existing fit values in the active row
        #     existing_value = self.roi_table.item(active_row, 5)  # Check one of the fit-related columns
        #     if existing_value and existing_value.text():  
        #         # Prompt user for overwriting
        #         msg = QMessageBox()
        #         msg.setIcon(QMessageBox.Warning)
        #         msg.setText("Do you want to overwrite existing fit values for the active ROI?")
        #         msg.setWindowTitle("Overwrite Confirmation")
        #         msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        #         retval = msg.exec_()
        #         if retval != QMessageBox.Yes:
        #             return  # Do not overwrite, return early

        #     # Update the table with new fit statistics
        #     for col_idx, label in enumerate(self.header_labels[5:], start=5):
        #         item_value = fit_stats_dict.get(label.replace(' ', '_').lower(), "")
        #         self.roi_table.setItem(active_row, col_idx, QTableWidgetItem(str(item_value)))

        
    def display_fit_statistics(self, stats):
        # Assuming you have QLineEdit widgets stored in a dictionary self.stat_boxes
        if stats is not None:
            self.stat_boxes['Amplitude'].setText(str(stats['amplitude']))
            self.stat_boxes['Center X'].setText(str(stats['centerx']))
            self.stat_boxes['Center Y'].setText(str(stats['centery']))
            self.stat_boxes['Chi-square'].setText(str(stats['chisqr']))
            self.stat_boxes['Reduced Chi-square'].setText(str(stats['redchi']))
            self.stat_boxes['R-squared'].setText(str(stats['rsquared']))
            self.stat_boxes['Sigma X'].setText(str(stats['sigmax']))
            self.stat_boxes['Sigma Y'].setText(str(stats['sigmay']))
            self.stat_boxes['FWHM X'].setText(str(stats['fwhmx']))
            self.stat_boxes['FWHM Y'].setText(str(stats['fwhmy']))

class MyCanvas(FigureCanvas):
    def __init__(self, fig, main_app=None, add_subplot=True, axes=None, subplot_dims=None):
        super(MyCanvas, self).__init__(fig)
        self.main_app = main_app
        self.fig = fig

        if add_subplot:
            self.ax = self.fig.add_subplot(111)
        elif axes is not None:
            if isinstance(axes, list):
                self.axes = axes
                # Dynamically set named attributes for axes
                for i, ax in enumerate(axes, start=2):
                    setattr(self, f'ax{i}', ax)
            elif isinstance(axes, dict):
                self.axes = axes
                for key, ax in axes.items():
                    setattr(self, key, ax)
            else:
                raise ValueError("Invalid type for axes. Should be either list or dictionary.")
        else:
            self.ax = None  # No axes provided

        if subplot_dims:
            self.subplot_dims = subplot_dims

        self.intensity = None
        self.colorbar = None

        self.roi_selector = None  # Initialize roi_selector to None, will be set later
        self.rectangles = []  # Initialize rectangles

        self.mpl_connect('key_press_event', self.on_key)
        
    def init_plot(self):
        '''Add an init_plot method to initialize a blank plot with no data. This can be called to reset the plot before adding new data.'''
        self.ax.clear()
        
        # Clear or reset other attributes
        self.peak_positions = None
        self.intensity = None
        self.scatter = None

        # self.roi_selector.rectangles = []  # Clear the existing rectangles in ROISelector
        
        # Remove the colorbar if it exists
        if self.colorbar:
            self.colorbar.remove()
            self.colorbar = None
        
        self.draw()
        # self.roi_selector = ROISelector(self.ax, self.main_app)
        # self.roi_selector.activate()
        
        # Initialize ROISelector only if not already initialized
        # if self.roi_selector is None:
        #     self.roi_selector = ROISelector(self.ax, self.main_app)

        # self.roi_selector.activate()

    def plot_data(self, intensity, peak_positions, coords=None):
        ''' 
        plot_data:
            Purpose:
            Plot the 2D intensity data.

            Implementation:
            Clears the existing axis.
            Plots the 2D heatmap using imshow.
            Adds or updates the colorbar.
        '''
        # self.init_plot()  # Reset the plot
            # Validation
        if intensity is None:
            print("Error: Intensity is None.")
            return
        if peak_positions is None:
            print("Error: Peak Positions are None.")
            return
        if not isinstance(intensity, xr.DataArray):
            print("Error: Intensity is not an xarray DataArray.")
            return
        if not isinstance(peak_positions, xr.DataArray):
            print("Error: Peak Positions is not an xarray DataArray.")
            return

        # Get extents based on xarray coordinates
        extent = [
            intensity.coords[intensity.dims[1]].min(),
            intensity.coords[intensity.dims[1]].max(),
            intensity.coords[intensity.dims[0]].min(),
            intensity.coords[intensity.dims[0]].max(),
        ]
        
        # Calculate vmin and vmax
        img_values = intensity.values
        vmin = np.nanpercentile(img_values, 10)
        vmax = np.nanpercentile(img_values, 99)
        
        # Change color map and set contrast
        im = self.ax.imshow(intensity.values, cmap='turbo', 
                            origin='lower', 
                            extent=extent, 
                            aspect='auto', 
                            vmin=vmin, vmax=vmax)
        
        # Add or update the colorbar
        if self.colorbar is not None:
            self.colorbar.remove()
        self.colorbar = self.fig.colorbar(im, ax=self.ax)
        
        # If coordinates for labels are provided, set them
        if coords:
            xlabel = coords.get('xlabel', None)
            ylabel = coords.get('ylabel', None)
            if xlabel:
                self.ax.set_xlabel(xlabel)
            if ylabel:
                self.ax.set_ylabel(ylabel)

        # Find the coordinates where peak_positions is 1
        y, x = np.where(peak_positions.values == 1)
        
        # Convert to actual coordinate values
        y_vals = peak_positions.coords[peak_positions.dims[0]].values[y]
        x_vals = peak_positions.coords[peak_positions.dims[1]].values[x]
        
        # Initialize facecolors
        num_points = len(x_vals)
        # initial_colors = np.full((num_points, 4), [1, 0, 0, 1])  # Initial color is red for all points
        
        # self.scatter = self.ax.scatter(x_vals, y_vals, facecolors=initial_colors)
        # self.facecolors = initial_colors

        # Update MyCanvas Intensity & Peak Positions variables.
        self.intensity = intensity
        self.peak_positions = peak_positions
        
        # If coordinates for labels are provided, set them
        if coords:
            xlabel = coords.get('xlabel', None)
            ylabel = coords.get('ylabel', None)
            if xlabel:
                self.ax.set_xlabel(xlabel)
            if ylabel:
                self.ax.set_ylabel(ylabel)

        self.draw()
        
        # if self.roi_selector is None:
        self.roi_selector = ROISelector(self.ax, self.main_app)
        
        # Use the existing ROISelector
        if self.roi_selector is not None:
            self.roi_selector.activate()        

    def on_key(self, event):
        if event.key in ['R', 'r']:
            self.roi_selector.activate()  # Assuming roi_selector is accessible as an attribute
        elif event.key in ['Q', 'q']:
            self.roi_selector.deactivate()

    def setup_rectangle_selector(self):
        ''' 
        setup_rectangle_selector:
            Purpose:
            Sets up the RectangleSelector for the plot.

            Implementation:
            Initializes a RectangleSelector with specific properties and binds it to the line_select_callback.

            Considerations:
            Call this method to initialize the RectangleSelector before using it.

            Attributes:
            RS (RectangleSelector): Initialized and set up for use.
        '''

        self.RS = RectangleSelector(self.ax, self.line_select_callback,
                                    useblit=True, button=[1, 3],
                                    minspanx=5, minspany=5,
                                    spancoords='pixels',
                                    interactive=True)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_rect_pick)

class MyNavigationToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent, coordinates=True):
        super(MyNavigationToolbar, self).__init__(canvas, parent, coordinates)
        self.window = parent  # Assuming the parent window is passed as `parent`
        
        script_dir = os.path.dirname(os.path.realpath(__file__))
        
        # Create custom buttons
        custom_buttons = [
            {'name': 'Load Data', 'icon': os.path.join(script_dir, 'icons/loaddata_icon.png'), 'function': self.window.load_project},
            {'name': 'Toggle ROI Selector', 'icon': os.path.join(script_dir, 'icons/roi_icon.png'), 'function': self.toggle_roi_selector, 'toggle': True}  # Add your own icon path for ROI
        ]
        
        for button in custom_buttons:
            action = QAction(QIcon(button['icon']), '', self)
            action.setToolTip(button['name'])
            if button.get('toggle', False):
                action.setCheckable(True)
            action.triggered.connect(button['function'])
            self.addAction(action)

        # Move custom buttons to the front
        for i in range(len(custom_buttons)):
            action = self.actions()[-1]
            self.removeAction(action)
            self.insertAction(self.actions()[0], action)

    # Toggle function for ROI Selector
    def toggle_roi_selector(self, checked):
        print("toggle_roi_selector called")  # Debugging print
        if self.window.canvas1.roi_selector is None:
            print("roi_selector is None")
        else:
            print("roi_selector is an instance of ROISelector")
        if checked:
            self.window.canvas1.roi_selector.activate()
            self.window.canvas1.draw()
        else:
            self.window.canvas1.roi_selector.deactivate()

class ROISelector:
    def __init__(self, ax, window):
        self.ax = ax
        self.window = window
        print(f"Initializing ROI Selector with ax: {self.ax}, {type(self.ax)}")

        self.rectangles = []  # List to store all rectangles
        self.rectangle_selector = RectangleSelector(self.ax, self.on_select,
                                                    useblit=True, button=[1, 3], 
                                                    minspanx=5, minspany=5,
                                                    spancoords='pixels', interactive=True)
        self.rectangle_selector.set_active(False)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.closing_action = False
        self.active_rect = None

        self.close_box_visible = False  # Add this line
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)  # Add this line

    def activate(self):
        print("ROISelector activate called")  # Debugging print
        self.rectangle_selector.set_active(True)
        print(f"RectangleSelector active state: {self.rectangle_selector.active}")

    def deactivate(self):
        print("ROISelector deactivate called")  # Debugging print
        self.rectangle_selector.set_active(False)
        print(f"RectangleSelector active state: {self.rectangle_selector.active}")

    def on_select(self, eclick, erelease):
        print(f"Window reference: {self.window}")  # Debugging print
        print(f"ROISelector in on_select: {id(self)}, Rectangles: {id(self.rectangles)}")

        # Check if a close action is happening
        if self.closing_action:
            self.closing_action = False
            return
        
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print(f"on_select: (x1, y1):{x1:3.2f}, {y1:3.2f}) --> (*x2, y2):{x2:3.2f}, {y2:3.2f})")

        # Create the main rectangle
        new_rect = Rectangle((min(x1, x2), min(y1, y2)), np.abs(x1 - x2), np.abs(y1 - y2),
                            edgecolor='black', facecolor='none', linewidth=1.5)

        # Create the close box
        close_box_size = 0.15
        close_box_center_x = min(x1, x2) - 0.6 * close_box_size
        close_box_center_y = max(y1, y2) + 0.6 * close_box_size
        close_box = Rectangle((close_box_center_x - close_box_size / 2, close_box_center_y - close_box_size / 2), 
                        close_box_size, close_box_size, 
                        edgecolor='black', facecolor='gray')  # Solid gray fill

        # Store the rectangle and its actual data coordinates in a dictionary
        self.rectangles.append({
            'main': new_rect,
            'close_box': close_box,
            'coords': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        })

        # Add a Line2D to represent the 'X' inside the close box
        x = close_box_center_x
        y = close_box_center_y
        half_size = close_box_size / 2
        x_line = [x - half_size, x + half_size]
        y_line = [y - half_size, y + half_size]
        close_x1 = self.ax.add_line(Line2D(x_line, y_line, color='black', linewidth=1))

        x_line = [x + half_size, x - half_size]
        close_x2 = self.ax.add_line(Line2D(x_line, y_line, color='black', linewidth=1))

        self.ax.add_patch(new_rect)
        self.ax.add_patch(close_box)
        self.ax.figure.canvas.draw()
        self.window.update_roi_table()

        # Update the stored rectangle dictionary to include the 'X'
        self.rectangles[-1].update({
            'close_x1': close_x1,
            'close_x2': close_x2,
        })

    def on_click(self, event):
        was_active = self.rectangle_selector.active  # Store the previous state
        self.rectangle_selector.set_active(False)  # Temporarily deactivate

        clicked_any = False  # Flag to know if any rectangle was clicked

        for rect in self.rectangles:
            main_rect = rect['main']
            close_box = rect['close_box']
            close_x1 = rect.get('close_x1', None)
            close_x2 = rect.get('close_x2', None)

            # Remove rectangle if the close box is clicked
            if close_box.contains(event)[0]:
                self.closing_action = True  # Set the flag
                main_rect.remove()
                close_box.remove()
                self.rectangles.remove(rect)
                self.ax.figure.canvas.draw()
                clicked_any = True

                # Remove the 'X' as well
                if close_x1:
                    close_x1.remove()
                if close_x2:
                    close_x2.remove()

                break  # Only remove the first rectangle that matches

            # Toggle active rectangle if the main rectangle is clicked
            elif main_rect.contains(event)[0]:
                if self.active_rect:  # Reset the previous active rectangle
                    self.active_rect['main'].set_edgecolor('black')
                
                self.active_rect = rect  # Set the new active rectangle
                main_rect.set_edgecolor('blue')  # Highlight the active rectangle
                
                # Coordinates of the selected ROI
                x1, y1, x2, y2 = main_rect.get_extents().get_points().flatten()
                print(f"on_click: (x1, y1):{x1:3.2f}, {y1:3.2f}) --> (*x2, y2):{x2:3.2f}, {y2:3.2f})")
                self.window.update_sliced_dataarray(x1, y1, x2, y2)
                
                self.ax.figure.canvas.draw()
                clicked_any = True
                break  # Only activate the first rectangle that matches

        self.rectangle_selector.set_active(was_active)  # Restore the previous state

        if not clicked_any:
            # Reset the previous active rectangle if none was clicked
            if self.active_rect:
                self.active_rect['main'].set_edgecolor('black')
            self.active_rect = None
        
        self.window.update_roi_table()

    def update_close_box_visibility(self, visible):
        """Updates the visibility of the close box."""
        for rect in self.rectangles:
            close_box = rect['close_box']
            close_box.set_visible(visible)
        self.ax.figure.canvas.draw()

    def on_mouse_move(self, event):
        for rect in self.rectangles:
            close_box = rect['close_box']
            close_x1 = rect.get('close_x1', None)
            close_x2 = rect.get('close_x2', None)

            if close_box.contains(event)[0]:
                # Show the close box and 'X'
                close_box.set_visible(True)
                if close_x1:
                    close_x1.set_visible(True)
                if close_x2:
                    close_x2.set_visible(True)
            else:
                # Hide the close box and 'X'
                close_box.set_visible(False)
                if close_x1:
                    close_x1.set_visible(False)
                if close_x2:
                    close_x2.set_visible(False)
        
        self.ax.figure.canvas.draw()

class Gaussian2DFit:
    # Adapted From: https://lmfit.github.io/lmfit-py/examples/example_two_dimensional_peak.html
    def __init__(self, data_array: xr.DataArray):
        self.data_array = data_array
        self.result = None

    @staticmethod
    def gaussian2d_rotated(x, y, amplitude, centerx, centery, sigmax, sigmay, rotation):
        xp = (x - centerx) * np.cos(rotation) - (y - centery) * np.sin(rotation)
        yp = (x - centerx) * np.sin(rotation) + (y - centery) * np.cos(rotation)
        g = amplitude * np.exp(-((xp/sigmax)**2 + (yp/sigmay)**2) / 2.)
        return g
    
    def construct_model(self):
        # self.model = lmfit.models.Gaussian2dModel()
        self.model = Model(self.gaussian2d_rotated, independent_vars=['x', 'y'])

    def perform_fit(self):
        x_vals = self.data_array.coords['q_xy'].values
        y_vals = self.data_array.coords['q_z'].values
        x, y = np.meshgrid(x_vals, y_vals)
        z = self.data_array['intensity'].values

        params = self.model.make_params(amplitude=np.max(z), centerx=np.mean(x_vals), centery=np.mean(y_vals),
                                        sigmax=np.std(x_vals), sigmay=np.std(y_vals), rotation=0)
        params['rotation'].set(value = .1, min=0, max=np.pi)
        params['sigmax'].set(min=0)
        params['sigmay'].set(min=0)

        error = np.sqrt(z + 1)
        self.result = self.model.fit(z.ravel(), x=x.ravel(), y=y.ravel(), params=params, weights=1/np.sqrt(error.ravel()))

    def calculate_residuals(self):
        if self.result is None:
            return None

        x, y = np.meshgrid(self.data_array.coords['q_xy'], self.data_array.coords['q_z'])
        fit_data = self.model.func(x.flatten(), y.flatten(), **self.result.best_values)
        return self.data_array['intensity'].values.flatten() - fit_data

    def fit_statistics(self):
        if self.result is None:
            return None

        # Create an empty dictionary to store fit statistics
        fit_stats_dict = {}

        # Chi-square
        fit_stats_dict['chi_square'] = self.result.chisqr

        # Akaike Information Criterion
        fit_stats_dict['aic'] = self.result.aic

        # Bayesian Information Criterion
        fit_stats_dict['bic'] = self.result.bic

        # Parameters and their uncertainties
        for param_name, param in self.result.params.items():
            fit_stats_dict[f"{param_name}_value"] = param.value
            fit_stats_dict[f"{param_name}_stderr"] = param.stderr

        return fit_stats_dict

    def plot_fit(self, ax_fit, ax_residual):
        if self.result is None:
            return

        # Clear the previous plots
        ax_fit.clear()
        ax_residual.clear()

        x, y = np.meshgrid(self.data_array.coords['q_xy'], self.data_array.coords['q_z'])

        # Generate grid for plotting
        x_plot, y_plot = np.meshgrid(
            np.linspace(x.min(), x.max(), 100),
            np.linspace(y.min(), y.max(), 100)
        )

        # Fit plot
        fit_data = self.model.func(x_plot, y_plot, **self.result.best_values)
        ax_fit.pcolor(x_plot, y_plot, fit_data, shading='auto')
        ax_fit.set_title('Fit')

        # Residual plot
        residuals = self.calculate_residuals()
        Z_residual = griddata((x.flatten(), y.flatten()), residuals.flatten(), (x_plot, y_plot), method='linear', fill_value=0)
        ax_residual.pcolor(x_plot, y_plot, Z_residual, shading='auto')
        ax_residual.set_title('Residuals')

        # Update canvas
        ax_fit.figure.canvas.draw_idle()
        ax_residual.figure.canvas.draw_idle()

class GaussianFitToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent, coordinates=True):
        super(GaussianFitToolbar, self).__init__(canvas, parent, coordinates)
        
        # Assuming the parent window is passed as `parent`
        self.window = parent  
        script_dir = os.path.dirname(os.path.realpath(__file__))

        # Create custom buttons
        custom_buttons = [
            {'name': '2D Gaussian Fit', 'icon': os.path.join(script_dir, 'icons/gaussian_icon.png'), 'function': self.window.perform_gaussian_fit},
            # {'name': 'Toggle ROI Selector', 'icon': os.path.join(script_dir, 'icons/roi_icon.png'), 'function': self.toggle_roi_selector, 'toggle': True}  # Add your own icon path for ROI
        ]
        
        for button in custom_buttons:
            action = QAction(QIcon(button['icon']), '', self)
            action.setToolTip(button['name'])
            if button.get('toggle', False):
                action.setCheckable(True)
            action.triggered.connect(button['function'])
            self.addAction(action)

        # Move custom buttons to the front
        for i in range(len(custom_buttons)):
            action = self.actions()[-1]
            self.removeAction(action)
            self.insertAction(self.actions()[0], action)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = XrayApp()
    ex.show()
    sys.exit(app.exec_())


    '''
        # self.setWindowTitle('X-ray Data Analysis')
        # self.setGeometry(100, 100, 1600, 800)

        # self.central_widget = QWidget()
        # self.setCentralWidget(self.central_widget)
        # self.layout = QHBoxLayout()
        # self.central_widget.setLayout(self.layout)

        # # Left part for canvas1 (50% of window)
        # self.left_frame = QFrame()
        # self.left_layout = QVBoxLayout()
        # self.left_frame.setLayout(self.left_layout)
        # self.layout.addWidget(self.left_frame, stretch=1)

        # # Main canvas1 setup
        # self.fig1, self.ax1 = plt.subplots()
        # self.ax1.axis('off')
        # self.canvas1 = MyCanvas(self.fig1, main_app=self, add_subplot=True)
        # self.nav1 = MyNavigationToolbar(self.canvas1, self)
        # self.main_layout = QVBoxLayout(self.left_frame)
        # self.main_layout.addWidget(self.nav1)
        # self.main_layout.addWidget(self.canvas1)

        # # ROI table
        # self.roi_table = QTableWidget()
        # self.roi_table.setColumnCount(5)
        # self.roi_table.setHorizontalHeaderLabels(['ROI #', 'q_xy (min)', 'q_xy (max)', 'q_z (min)', 'q_z (max)'])
        # self.main_layout.addWidget(self.roi_table)
        
        # # Right part for canvas2 and stats (50% of window)
        # self.right_frame = QFrame()
        # self.right_layout = QVBoxLayout()
        # self.right_frame.setLayout(self.right_layout)
        # self.layout.addWidget(self.right_frame, stretch=1)
        
        # # Canvas2 setup
        # self.fig2, self.axs = plt.subplots(2, 2, figsize=(8, 4))
        # self.ax2, self.ax3, self.ax4, self.ax5 = self.axs.flatten()
        # self.ax3.axis('off')
        # self.canvas2 = MyCanvas(self.fig2, main_app=self, add_subplot=False)
        # self.nav2 = MyNavigationToolbar(self.canvas2, self)
        # self.nav2.setFixedWidth(600)
        # self.right_layout.addWidget(self.nav2)
        # self.right_layout.addWidget(self.canvas2, stretch=2)

        # # Stats setup
        # self.stats_widget = QWidget()
        # self.stats_layout = QGridLayout()
        # self.stats_widget.setLayout(self.stats_layout)
        # self.add_stat_box('Amplitude', 1, 0)
        # self.add_stat_box('Center X', 2, 0)
        # self.add_stat_box('Center Y', 3, 0)
        # self.add_stat_box('Chi-square', 1, 1)
        # self.add_stat_box('Reduced Chi-square', 2, 1)
        # self.add_stat_box('R-squared', 3, 1)
        # self.add_stat_box('Sigma X', 1, 2)
        # self.add_stat_box('Sigma Y', 2, 2)
        # self.add_stat_box('FWHM X', 3, 2)
        # self.add_stat_box('FWHM Y', 4, 2)
        # self.right_layout.addWidget(self.stats_widget, stretch=1)

    
    def initUI(self):
        self.setWindowTitle('X-ray Data Analysis')
        self.setGeometry(100, 100, 1600, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Create a frame for the left part (plot + sliders)
        self.left_frame = QFrame()
        self.left_layout = QVBoxLayout()
        self.left_frame.setLayout(self.left_layout)
        self.layout.addWidget(self.left_frame)

        # Create main frame for X-ray plot
        self.main_frame = QFrame()
        self.main_layout = QVBoxLayout()
        self.main_frame.setLayout(self.main_layout)
        self.left_layout.addWidget(self.main_frame)

        # Create matplotlib figure and attach it to main frame
        self.fig1, self.ax1 = plt.subplots()
        self.ax1.axis('off')

        self.canvas1 = MyCanvas(self.fig1, main_app=self, add_subplot=True)
        # self.canvas1 = MyCanvas(self.fig1, main_app=self, add_subplot=True)
        self.main_layout.addWidget(self.canvas1)

        self.nav1 = MyNavigationToolbar(self.canvas1, self)
        self.main_layout.insertWidget(0, self.nav1)

        # Initialize the output frame and layout
        self.output_frame = QFrame()
        self.output_layout = QVBoxLayout()
        self.output_frame.setLayout(self.output_layout)
        self.layout.addWidget(self.output_frame)  # add output_frame to main layout
        
        # Create matplotlib figure and set up GridSpec
        self.fig2 = plt.figure(figsize=(8, 4))  # Adjust figsize as needed
        gs = GridSpec(2, 2, width_ratios=[3, 2])  # ax2 will be twice as wide
        
        self.ax2 = self.fig2.add_subplot(gs[0, 0])
        self.ax3 = self.fig2.add_subplot(gs[1, 0])  # You can leave this blank
        self.ax4 = self.fig2.add_subplot(gs[0, 1])
        self.ax5 = self.fig2.add_subplot(gs[1, 1])
        
        self.ax3.axis('off')  # Turn off ax3 if not used

        self.output_layout = QVBoxLayout()
        self.output_frame.setLayout(self.output_layout)
        
        # Use setStretch to control the vertical size
        self.output_layout.setStretch(0, 3)  # For canvas2
        self.output_layout.setStretch(1, 2)  # For widget3

        # Create a QWidget for ax3 and set it to a grid layout
        self.widget3 = QWidget()
        self.grid3 = QGridLayout()
        self.widget3.setLayout(self.grid3)

        # Add 'Fit 2D Gaussian' button
        self.btn_fit = QPushButton('Fit 2D Gaussian', self)
        self.btn_fit.clicked.connect(self.perform_gaussian_fit)
        self.grid3.addWidget(self.btn_fit, 0, 0)

        # Add boxes and labels for fit statistics
        self.add_stat_box('Amplitude', 1, 0)
        self.add_stat_box('Center X', 2, 0)
        self.add_stat_box('Center Y', 3, 0)
        self.add_stat_box('Chi-square', 1, 1)
        self.add_stat_box('Reduced Chi-square', 2, 1)
        self.add_stat_box('R-squared', 3, 1)
        self.add_stat_box('Sigma X', 1, 2)
        self.add_stat_box('Sigma Y', 2, 2)
        self.add_stat_box('FWHM X', 3, 2)
        self.add_stat_box('FWHM Y', 4, 2)

        # Add the QWidget for ax3 into your layout
        self.output_layout.addWidget(self.widget3)
        
        # Create a dictionary to map custom names to axes
        axes_dict = {'ax2': self.ax2, 'ax3': self.ax3, 'ax4': self.ax4, 'ax5': self.ax5}
        self.canvas2 = MyCanvas(self.fig2, main_app=self, add_subplot=False, axes=axes_dict)

        self.output_layout.addWidget(self.canvas2)  # Now this should work
        self.nav2 = NavigationToolbar(self.canvas2, self)
        self.nav2.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        self.nav2.setFixedWidth(600)  # set to a suitable widthThere 
        self.output_layout.insertWidget(0, self.nav2)

        # Create QTableWidget object
        self.roi_table = QTableWidget()
        self.roi_table.setColumnCount(5)
        self.roi_table.setHorizontalHeaderLabels(['ROI #', 'q_xy (min)', 'q_xy (max)', 'q_z (min)', 'q_z (max)'])
        
        # Add table to the layout
        self.main_layout.addWidget(self.roi_table)
        self.roi_table.cellClicked.connect(self.on_table_click)
        # self.roi_table.cellClicked.connect(self.row_was_selected)
        # self.roi_table.itemSelectionChanged.connect(self.row_was_selected)

        self.roi_selector = None
        self.coords = None

        # # Fix the heights of the canvases
        # self.canvas1.setFixedHeight(400)
        # self.canvas2.setFixedHeight(400)

        # # Align the toolbars at the top
        # self.main_layout.setAlignment(self.nav1, Qt.AlignTop)
        # self.output_layout.setAlignment(self.nav2, Qt.AlignTop)

        # # Allow canvases to expand but keep them at the same height
        # self.main_layout.setStretchFactor(self.canvas1, 1)
        # self.output_layout.setStretchFactor(self.canvas2, 1)
        # self.roi_selector = ROISelector(self.ax1)
        # self.roi_selector = ROISelector(self.ax1, self)
        # self.canvas1.roi_selector = self.roi_selector
        # self.roi_selector.activate()
        # print(f"ROISelector in XrayApp: {self.roi_selector}, active: {self.roi_selector.rectangle_selector.active}")
        # print(f"ROISelector in canvas1: {self.canvas1.roi_selector}, active: {self.canvas1.roi_selector.rectangle_selector.active}")

        # def onclick(event):
        #     print(f"Mouse click event: {event}")
            
        # self.fig1.canvas.mpl_connect('button_press_event', onclick)
    '''
     