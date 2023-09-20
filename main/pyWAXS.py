# -- PyQt5 Imports -- #
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QGridLayout, QWidget, QFileDialog, QGroupBox, QVBoxLayout, QSlider, QLabel, QAction, QDialog, QFormLayout, QLineEdit, QComboBox, QMessageBox, QTextEdit, QTableWidget, QTableWidgetItem, QRadioButton, QToolBar
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QFont

# -- Matplotlib Imports -- #
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.text import Text
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle  # Make sure to import Rectangle
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

# -- Additional Imports -- #
import sys, os, subprocess, re, csv
import xarray as xr
import numpy as np
from pathlib import Path

# -- Custom Imports -- #
from WAXSReduce import WAXSReduce
from WAXSReduce import Integration1D
# from pyWAXSim import SimulatedIntensityWindow

class MyTableWidget(QTableWidget):
    row_selected = pyqtSignal(int)  # Signal to emit the selected row index

    def __init__(self, parent=None):
        super(MyTableWidget, self).__init__(parent)
        self.setSortingEnabled(True)
        
        # Enable custom sorting
        header = self.horizontalHeader()
        header.setSortIndicatorShown(True)
        header.setSectionsClickable(True)
        header.sortIndicatorChanged.connect(self.customSort)

    def lessThan(self, item1, item2):
        value1 = item1.data(Qt.UserRole)
        value2 = item2.data(Qt.UserRole)

        if value1 is not None and value2 is not None:
            return value1 < value2
        else:
            return item1.text() < item2.text()

    def customSort(self, index):
        self.sortItems(index, Qt.AscendingOrder if self.horizontalHeader().sortIndicatorOrder() == Qt.AscendingOrder else Qt.DescendingOrder)

    def selectionChanged(self, selected, deselected):
        super(MyTableWidget, self).selectionChanged(selected, deselected)
        for index in selected.indexes():
            self.row_selected.emit(index.row())

    def selectRow(self, row):
        super(MyTableWidget, self).selectRow(row)
        self.row_selected.emit(row)
        print(f"Row {row} selected, signal emitted.")  # Debugging statement

# File Dialog: Create New Project
class CreateProjectDialog(QDialog):
    waxs_reduce_created = pyqtSignal(object)  # Signal carrying the WAXSReduce instance

    def __init__(self):
        super().__init__()

        # Initialize widgets and layout
        self.initUI()

    def initUI(self):
        '''
        # self.poniPath = QLineEdit(self)
        # self.maskPath = QLineEdit(self)
        # self.tiffPath = QLineEdit(self)
        # self.metadata_keylist = QLineEdit(self)
        # self.inplane_config = QLineEdit(self)
        # self.energy = QLineEdit(self)
        # self.incident_angle = QLineEdit(self)
        # self.hdf5Path = QLineEdit(self)
        # self.projectName = QLineEdit(self)

        # formLayout = QFormLayout()
        # formLayout.addRow("PONI Path:", self.poniPath)
        # formLayout.addRow("Mask Path:", self.maskPath)
        # formLayout.addRow("TIFF Path:", self.tiffPath)
        # formLayout.addRow("Metadata Keylist:", self.metadata_keylist)
        # formLayout.addRow("Inplane Config:", self.inplane_config)
        # formLayout.addRow("Energy:", self.energy)
        # formLayout.addRow("Incident Angle:", self.incident_angle)
        # formLayout.addRow("HDF5 Path:", self.hdf5Path)
        # formLayout.addRow("Project Name:", self.projectName)

        # self.submitButton = QPushButton("Submit", self)
        # self.submitButton.clicked.connect(self.accept)

        # layout = QVBoxLayout()
        # layout.addLayout(formLayout)
        # layout.addWidget(self.submitButton)
        # self.setLayout(layout)
        '''

        self.projectName = QLineEdit(self)
        self.projectName.setPlaceholderText("Project Name")

        self.poniPathBtn = QPushButton("Select PONI File", self)
        self.poniPathLabel = QLabel("No file selected")

        self.maskPathBtn = QPushButton("Select Mask File", self)
        self.maskPathLabel = QLabel("No file selected")

        self.tiffPathBtn = QPushButton("Select TIFF File", self)
        self.tiffPathLabel = QLabel("No file selected")

        self.delimiterComboBox = QComboBox(self)
        self.delimiterComboBox.addItems(["_", "-"])
        
        self.metadata_keylist = QTextEdit(self)
        self.metadata_keylist.setPlaceholderText("Comma-separated values")
        self.metadata_keylist.setFixedHeight(80)  # Set the height to fit approximately 4 lines

        self.inplane_config = QLineEdit("q_xy", self)
        self.energy = QLineEdit("12.7", self)
        self.incident_angle = QLineEdit("0.3", self)

        self.hdf5PathBtn = QPushButton("Select Project Path", self)
        self.hdf5PathLabel = QLabel("Default path: Projects/")

        self.submitButton = QPushButton("Submit", self)
        self.submitButton.clicked.connect(self.validate_and_accept)

        # Layout
        formLayout = QFormLayout()
        formLayout.addRow("Project Name:", self.projectName)
        formLayout.addRow("PONI Path:", self.poniPathBtn)
        formLayout.addRow("", self.poniPathLabel)
        formLayout.addRow("Mask Path:", self.maskPathBtn)
        formLayout.addRow("", self.maskPathLabel)
        formLayout.addRow("TIFF Path:", self.tiffPathBtn)
        formLayout.addRow("", self.tiffPathLabel)
        formLayout.addRow("File Delimiter:", self.delimiterComboBox)
        formLayout.addRow("Metadata Keylist:", self.metadata_keylist)
        formLayout.addRow("Inplane Config:", self.inplane_config)
        formLayout.addRow("Energy:", self.energy)
        formLayout.addRow("Incident Angle:", self.incident_angle)
        formLayout.addRow("Project Path:", self.hdf5PathBtn)
        formLayout.addRow("", self.hdf5PathLabel)
        formLayout.addRow("", self.submitButton)

        layout = QVBoxLayout()
        layout.addLayout(formLayout)
        self.setLayout(layout)

        self.maskPathBtn.clicked.connect(self.open_mask_dialog)
        self.poniPathBtn.clicked.connect(self.open_poni_dialog)
        self.tiffPathBtn.clicked.connect(self.open_tiff_dialog)
        self.hdf5PathBtn.clicked.connect(self.select_projectfolder_dialog)
        self.submitButton.clicked.connect(self.validate_and_accept)
        # Connect other buttons to their respective methods

    def create_project(self):
        dialog = CreateProjectDialog()
        dialog.waxs_reduce_created.connect(self.window.store_waxs_reduce)  # Connect the signal
        result = dialog.exec_()

        if result == QDialog.Accepted:
            # Use the Path attributes directly here
            poniPath = dialog.poniPath
            maskPath = dialog.maskPath
            tiffPath = dialog.tiffPath
            # metadata_keylist = dialog.metadata_keylist.text().split(',')
            metadata_keylist = dialog.metadata_keylist.toPlainText().split(',')
            inplane_config = dialog.inplane_config.text()
            energy = float(dialog.energy.text())
            incident_angle = float(dialog.incident_angle.text())
            hdf5Path = dialog.hdf5Path.text()
            projectName = dialog.projectName.text()

            # Initialize WAXSReduce class
            self.waxs_project = WAXSReduce(poniPath, maskPath, tiffPath, metadata_keylist, inplane_config,
                                        energy, incident_angle, hdf5Path, projectName)
            
            self.waxs_reduce_created.emit(self.waxs_project)  # Emit the created instance

    def validate_and_accept(self):
        error_messages = []
        
        # Validation logic for each field
        if not self.validate_project_name():
            error_messages.append("Invalid Project Name.")
        
        if not self.validate_poni_path():
            error_messages.append("Invalid PONI Path.")
        
        if not self.validate_mask_path():
            error_messages.append("Invalid Mask Path.")
        
        if not self.validate_tiff_path():
            error_messages.append("Invalid TIFF Path.")
        
        if not self.validate_metadata_keylist():
            error_messages.append("Invalid Metadata Keylist.")
        
        if not self.validate_inplane_config():
            error_messages.append("Invalid Inplane Config.")
        
        if not self.validate_energy():
            error_messages.append("Invalid Energy.")
        
        if not self.validate_incident_angle():
            error_messages.append("Invalid Incident Angle.")
        
        if not self.validate_hdf5_path():
            error_messages.append("Invalid Project Path.")

        if error_messages:
            QMessageBox.critical(self, "Error", "\n".join(error_messages), QMessageBox.Ok)
        else:
            self.accept()
            # Emit the waxs_reduce_created signal with the WAXSReduce instance

    def validate_project_name(self):
        project_name = self.projectName.text().strip()
        if re.match("^[a-zA-Z0-9_-]+$", project_name):
            return True
        return False

    def validate_poni_path(self):
        poni_path = Path(self.poniPath.text())
        return poni_path.exists() and poni_path.suffix.lower() == '.poni'

    def validate_mask_path(self):
        mask_path = Path(self.maskPath.text())
        return mask_path.exists() and mask_path.suffix.lower() in ['.edf', '.json']

    def validate_tiff_path(self):
        tiff_path = Path(self.tiffPath.text())
        return tiff_path.exists() and tiff_path.suffix.lower() == '.tiff'

    def validate_metadata_keylist(self):
        keylist_str = self.metadata_keylist.text().strip()
        keylist = [k.strip() for k in keylist_str.split(",")]
        return all(keylist)

    def validate_inplane_config(self):
        config = self.inplane_config.text().strip()
        return bool(re.match("^[a-zA-Z][a-zA-Z0-9_-]*$", config))

    def validate_energy(self):
        try:
            energy = float(self.energy.text())
            return 0 <= energy <= 300
        except ValueError:
            return False

    def validate_incident_angle(self):
        try:
            angle = float(self.incident_angle.text())
            if angle == 0:
                QMessageBox.information(self, "Notice", "0 deg. Grazing Incidence: Transmission Mode Selected", QMessageBox.Ok)
            return angle >= 0
        except ValueError:
            return False

    def validate_hdf5_path(self):
        path_str = self.hdf5Path.text().strip()
        if path_str.lower() == "default path: projects/":
            project_dir = Path("Projects")
            project_dir.mkdir(exist_ok=True)
            return True
        else:
            path = Path(path_str)
            return path.exists() and path.is_dir()

    def open_poni_dialog(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "Select PONI File", "", "PONI Files (*.poni);;All Files (*)", options=options)
        if file:
            self.poniPath = Path(file)
            self.poniPathLabel.setText(str(self.poniPath))

    def open_mask_dialog(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "Select Mask File", "", "Mask Files (*.edf);Mask Files (*.json);All Files (*)", options=options)
        if file:
            self.maskPath = Path(file)
            self.maskPathLabel.setText(str(self.maskPath))

    def open_tiff_dialog(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "Select TIFF (Data) File", "", "TIFF Files (*.tiff);;All Files (*)", options=options)
        if file:
            self.tiffPath = Path(file)
            self.tiffPathLabel.setText(str(self.tiffPath))

    def select_projectfolder_dialog(self):
        options = QFileDialog.Options()
        folder = QFileDialog.getExistingDirectory(self, "Select Project Folder", "", options=options)
        if folder:
            self.hdf5Path = Path(folder)
            self.hdf5PathLabel.setText(str(self.hdf5Path))

# Toolbar @ the top of the window
class MyNavigationToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent, coordinates=True):
        ''' 
        __init__:
            Purpose:
            Initializes a new instance of the MyNavigationToolbar class, inheriting functionalities from NavigationToolbar2QT and adding custom behavior.

            Implementation:
            Calls the constructor of the parent NavigationToolbar2QT class and initializes a reference to the parent window.

            Considerations:
            Assumes the parent window is passed as `parent`.

            Attributes:
            window (QMainWindow): Reference to the parent window. Used for deactivating certain buttons when toolbar actions are triggered.
        '''

        super(MyNavigationToolbar, self).__init__(canvas, parent, coordinates)
        self.window = parent  # Assuming the parent window is passed as `parent`
        
        script_dir = os.path.dirname(os.path.realpath(__file__))
        
        # Create custom buttons
        custom_buttons = [
            {'name': 'Create Project', 'icon': os.path.join(script_dir, 'icons/create_project_icon.png'), 'function': self.create_project},
            {'name': 'Load Project', 'icon': os.path.join(script_dir, 'icons/load_project_icon.png'), 'function': self.load_project},
            # {'name': 'Create Calibrant', 'icon': os.path.join(script_dir, 'icons/create_calibrant_icon.png'), 'function': self.create_calibrant},
            # {'name': 'Load PONI', 'icon': os.path.join(script_dir, 'icons/load_poni_icon.png'), 'function': self.load_poni},
            # {'name': 'Load MASK', 'icon': os.path.join(script_dir, 'icons/load_mask_icon.png'), 'function': self.load_mask},
            {'name': 'Export Project', 'icon': os.path.join(script_dir, 'icons/export_project_icon.png'), 'function': self.export_project},
        ]
        
        for button in custom_buttons:
            action = QAction(QIcon(button['icon']), '', self)
            action.setToolTip(button['name'])
            action.triggered.connect(button['function'])
            self.addAction(action)

        # Move custom buttons to the front
        for i in range(len(custom_buttons)):
            action = self.actions()[-1]
            self.removeAction(action)
            self.insertAction(self.actions()[0], action)

    # Methods for each custom NavigationToolbar Button
    def create_project(self):
        dialog = CreateProjectDialog()
        dialog.waxs_reduce_created.connect(self.window.store_waxs_reduce)  # Connect the signal
        result = dialog.exec_()

        if result == QDialog.Accepted:
            poniPath = dialog.poniPath.text()
            maskPath = dialog.maskPath.text()
            tiffPath = dialog.tiffPath.text()
            metadata_keylist = dialog.metadata_keylist.text().split(',')
            inplane_config = dialog.inplane_config.text()
            energy = float(dialog.energy.text())
            incident_angle = float(dialog.incident_angle.text())
            hdf5Path = dialog.hdf5Path.text()
            projectName = dialog.projectName.text()

            # Initialize WAXSReduce class
            self.window.waxs_reduce = WAXSReduce(
                poniPath, maskPath, tiffPath, metadata_keylist, inplane_config,
                energy, incident_angle, hdf5Path, projectName
            )

    '''
    # Modified load_project method to have similar functionality as loadData
    def load_project(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self.window, "Load Project Data", "", "NetCDF Files (*.nc);;All Files (*)", options=options)
        if file:
            self.window.ds = xr.open_dataset(file, engine='h5netcdf')
            self.window.canvas.plot_data(self.window.ds['intensity'], self.window.ds['peak_positions'])

        if self.window.ds:
            max_intensity = self.window.ds['intensity'].max()
            max_intensity = int(max_intensity.values)
            self.window.slider_vmin.setMaximum(max_intensity)
            self.window.slider_vmax.setMaximum(max_intensity)
    
    '''

    def load_project(self):
        # Check if a project is already loaded
        if hasattr(self.window, 'ds') and self.window.ds is not None:
            reply = QMessageBox.question(self.window, 'Save Current Project', 
                                        'Would you like to save the current project before loading a new one?', 
                                        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, 
                                        QMessageBox.Cancel)

            if reply == QMessageBox.Yes:
                options = QFileDialog.Options()
                filepath, _ = QFileDialog.getSaveFileName(self.window, "Save Current Project Data", "", "NetCDF Files (*.nc);;All Files (*)", options=options)
                if filepath:
                    if not filepath.endswith('.nc'):
                        filepath += '.nc'
                    try:
                        self.window.canvas.export_data(filepath)
                    except Exception as e:
                        QMessageBox.critical(self.window, "Error", f"An error occurred while saving: {e}", QMessageBox.Ok)
                        return  # If save fails, don't proceed to load a new project

            elif reply == QMessageBox.Cancel:
                return  # Don't proceed to load a new project

            # Clear the current dataset only if we are sure that a new project will be loaded
            self.window.ds = None

        # Load a new project
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self.window, "Load Project Data", "", "NetCDF Files (*.nc);;All Files (*)", options=options)
        if file:
            try:
                
                # Initialize the plot to a clean state
                self.window.canvas.init_plot() # clear the plot
                self.window.ds = None # clear the dataset

                # Load the new dataset
                new_ds = xr.open_dataset(file, engine='h5netcdf') # load the dataset
                print(new_ds['intensity'])
                print(new_ds['peak_positions'])

                self.window.ds = new_ds # set the dataset 

                # Plotting: Surround with try-except to catch errors
                try:
                    if new_ds['intensity'] is None or new_ds['peak_positions'] is None:
                        raise ValueError("Intensity or peak positions are None.")
                    
                    # Extract coordinate names
                    coord_names_intensity = list(new_ds['intensity'].dims)
                    coord_names_peak = list(new_ds['peak_positions'].dims)
                    
                    coords = {
                        'xlabel': coord_names_intensity[1] if len(coord_names_intensity) > 1 else None,  
                        'ylabel': coord_names_intensity[0] if len(coord_names_intensity) > 0 else None,
                    }
                    
                    self.window.canvas.plot_data(new_ds['intensity'], new_ds['peak_positions'], coords)

                except Exception as e:
                    QMessageBox.critical(self.window, "Error", f"An error occurred while plotting: {e}", QMessageBox.Ok)
                    return  # If plotting fails, don't proceed

                # Update sliders using the newly loaded dataset
                max_intensity = new_ds['intensity'].max()
                max_intensity = int(max_intensity.values)
                self.window.slider_vmin.setMaximum(max_intensity)
                self.window.slider_vmax.setMaximum(max_intensity)
                
                # # If everything went well, update the current dataset
                # self.window.ds = new_ds

            except Exception as e:
                QMessageBox.critical(self.window, "Error", f"An error occurred while loading the project: {e}", QMessageBox.Ok)

    # Opens PyFAI-calib2 through a subprocess routine. Assumes your environment 
    def create_calibrant(self):
        try:
            subprocess.run(["bash", "-c", "eval \"$(conda shell.bash hook)\" && conda activate pyWAXS && pyFAI-calib2"])
        except Exception as e:
            print(f"An error occurred: {e}")
        
    def export_project(self):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getSaveFileName(self.window, "Export Project Data", "", "NetCDF Files (*.nc);;All Files (*)", options=options)
        
        if filepath:
            if not filepath.endswith('.nc'):
                filepath += '.nc'
            try:
                self.window.canvas.export_data(filepath)
                QMessageBox.information(self.window, "Success", "Project data exported successfully.", QMessageBox.Ok)
            except Exception as e:
                QMessageBox.critical(self.window, "Error", f"An error occurred while exporting: {e}", QMessageBox.Ok)

    def trigger_tool(self, *args, **kwargs):
        ''' 
        trigger_tool:
            Purpose:
            Overrides the trigger_tool method of NavigationToolbar2QT to add custom behavior when toolbar buttons are clicked.

            Implementation:
            Calls the original trigger_tool method and then deactivates the Add Point and Remove Point functionalities in the parent window.

            Considerations:
            This method is automatically called when any toolbar button is clicked.

            Attributes:
            None, but affects the behavior of the parent window by calling its deactivateAddPoint and deactivateRemovePoint methods.
        '''

        super().trigger_tool(*args, **kwargs)
        self.window.deactivateAddPoint()
        self.window.deactivateRemovePoint()

# Figure Layout & Control Methods for Figure
class MyCanvas(FigureCanvas):
    def __init__(self):
        ''' 
        __init__:
            Purpose:
            Initializes a new instance of the MyCanvas class, setting up the figure, axes, and other instance variables required for plotting and interaction.

            Implementation:
            Initializes a figure and subplot.
            Sets initial states for various instance variables that track the state of the plot and interactions.

            Considerations:
            Ensure this is the first method called in any subclass.

            Attributes:
            fig (matplotlib.figure.Figure): The main figure object for plotting.
            ax (matplotlib.axes._subplots.AxesSubplot): The axis object for the main figure.
            peak_positions (xarray.DataArray): DataArray to store the peak positions.
            intensity (xarray.DataArray): DataArray to store the intensity values.
            scatter (matplotlib.collections.PathCollection): Scatter plot object for peak positions.
            highlighted_points (list): List to keep track of highlighted points.
            highlighted_indices (list): List to keep track of indices of highlighted points.
            rectangles (list): List to keep track of rectangular selectors.
            selected_rect (Rectangle): The current selected rectangle.
            RS (RectangleSelector): The RectangleSelector object.
            texts (list): List to keep track of 'X' text objects.
            cid_pick (int): Holds the connection ID for the pick event.
        '''

        self.fig = Figure()
        super(MyCanvas, self).__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.peak_positions = None
        self.intensity = None  
        self.scatter = None
        self.highlighted_points = []  
        self.highlighted_indices = []  
        self.rectangles = []  
        self.selected_rect = None  
        self.RS = None  
        self.texts = []  
        self.cid_pick = self.mpl_connect('pick_event', self.on_rect_pick)
        self.colorbar = None
        self.tableWidget = MyTableWidget()
        self.tableWidget.setColumnCount(7)
        self.tableWidget.setHorizontalHeaderLabels(
            ['qᵣ (Å⁻¹)', '𝛘 (°)', 'qxy (Å⁻¹)', 'qz (Å⁻¹)', 'd-spacing (Å)', 'Intensity', '']
        )
        self.tableWidget.setColumnHidden(6, True)  
        self.tableWidget.setSortingEnabled(True)
        # self.tableWidget.row_selected.connect(self.highlight_scatter_point)
        # Connect the currentRowChanged signal to the highlight_scatter_point function
        self.tableWidget.selectionModel().currentRowChanged.connect(self.highlight_scatter_point)
        self.tableWidget.selectionModel().currentRowChanged.connect(self.some_test_function)
        self.tableWidget.selectionModel().selectionChanged.connect(self.on_table_selection_change)
        self.row_to_point_map = {}

        # Inside MyCanvas.__init__
        self.tableWidget.clicked.connect(self.cell_was_clicked)

    def cell_was_clicked(self, index):
        print(f"Cell clicked: Row {index.row()} Column {index.column()}")

    def some_test_function(self, current, previous):
        print("Row changed. Current:", current.row(), "Previous:", previous.row())

    def set_dataset(self, ds):
        '''Set the dataset from the MyWindow class each time the dataset 'ds' stored in MyCanvas is updated.'''
        self.ds = ds # update the MyCanvas dataset

    def init_plot(self):
        '''Add an init_plot method to initialize a blank plot with no data. This can be called to reset the plot before adding new data.'''
        self.ax.clear()
        
        # Clear or reset other attributes
        self.peak_positions = None
        self.intensity = None
        self.scatter = None
        self.highlighted_points.clear()
        self.highlighted_indices.clear()
        self.rectangles.clear()
        self.selected_rect = None
        if self.RS:
            self.RS.set_active(False)
            self.RS = None
        self.texts.clear()
        
        # Remove the colorbar if it exists
        if self.colorbar:
            self.colorbar.remove()
            self.colorbar = None
        
        self.draw()

    def plot_data(self, intensity, peak_positions, coords=None):
        ''' 
        plot_data:
            Purpose:
            Plot the 2D intensity data and marks the peak positions.

            Implementation:
            Clears the existing axis.
            Plots the 2D heatmap using imshow.
            Plots the scatter points for peak positions.
            Adds or updates the colorbar.

            Considerations:
            Expects intensity and peak_positions to be xarray DataArrays with specific dimensions.
            
            Attributes:
            intensity (xarray.DataArray): Updated with the newly plotted intensity data.
            peak_positions (xarray.DataArray): Updated with the newly plotted peak positions.
            scatter (matplotlib.collections.PathCollection): Updated scatter plot object for peak positions.
            facecolors (numpy.ndarray): Array to keep track of the colors of scatter points.
            colorbar (matplotlib.colorbar.Colorbar): Colorbar for the 2D heatmap.
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

        self.ax.clear()
        
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
        
        # Find the coordinates where peak_positions is 1
        y, x = np.where(peak_positions.values == 1)
        
        # Convert to actual coordinate values
        y_vals = peak_positions.coords[peak_positions.dims[0]].values[y]
        x_vals = peak_positions.coords[peak_positions.dims[1]].values[x]
        
        # Initialize facecolors
        num_points = len(x_vals)
        initial_colors = np.full((num_points, 4), [1, 0, 0, 1])  # Initial color is red for all points
        
        self.scatter = self.ax.scatter(x_vals, y_vals, facecolors=initial_colors)
        self.facecolors = initial_colors

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

        # self.set_dataset(self.ds)  # Update ds in MyCanvas
        self.update_table()
        self.draw()

        # # Debugging statements
        # print(f"Intensity DataArray: {intensity}")
        # print(f"Peak Positions DataArray: {peak_positions}")
        # print(f"Colorbar: {self.colorbar}")
        # print(f"Scatter Plot Object: {self.scatter}")

    def update_scatter(self):
        ''' 
        update_scatter:
            Purpose:
            Updates the scatter plot when new points are added or removed.

            Implementation:
            Finds the new peak positions and updates the scatter plot accordingly.

            Considerations:
            Should be called after any operation that modifies peak positions.

            Attributes:
            peak_positions (xarray.DataArray): Updated with new peak positions if any.
        '''

        y, x = np.where(self.peak_positions.values == 1)
        y_vals = self.peak_positions.coords[self.peak_positions.dims[0]].values[y]
        x_vals = self.peak_positions.coords[self.peak_positions.dims[1]].values[x]
        self.scatter.set_offsets(np.c_[x_vals, y_vals])
        
        self.update_table()
        self.draw()

    @staticmethod
    def find_closest(array, value):
        ''' 
        find_closest:
            Purpose:
            Finds the closest value in an array to a given value.

            Implementation:
            Computes the absolute difference between each element and the given value, and returns the closest one.

            Considerations:
            Utilized for mapping mouse click coordinates to data coordinates.

            Attributes:
            None
        '''

        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def add_point(self, event):
        ''' 
        add_point:
            Purpose:
            Adds a point to the peak_positions DataArray based on mouse click.

            Implementation:
            Maps the mouse click coordinates to the closest data point and sets it as a peak.

            Considerations:
            Make sure to deactivate this when other functionalities like pan or zoom are activated.

            Attributes:
            peak_positions (xarray.DataArray): Updated to include the newly added point.
        '''

        ix = self.find_closest(self.peak_positions.coords[self.peak_positions.dims[1]].values, event.xdata)
        iy = self.find_closest(self.peak_positions.coords[self.peak_positions.dims[0]].values, event.ydata)
        self.peak_positions.loc[{self.peak_positions.dims[0]: iy, self.peak_positions.dims[1]: ix}] = 1
        
        self.update_scatter()
        self.update_table()

    def remove_point(self, event):
        ''' 
        remove_point:
            Purpose:
            Removes a point from the peak_positions DataArray based on mouse click.

            Implementation:
            Finds the closest existing peak to the mouse click and removes it.

            Considerations:
            Make sure to deactivate this when other functionalities like pan or zoom are activated.

            Attributes:
            peak_positions (xarray.DataArray): Updated to exclude the removed point.
        '''

        # Get the coordinates of points that are currently peaks (value is 1)
        y, x = np.where(self.peak_positions.values == 1)
        
        # Convert to actual coordinate values
        y_vals = self.peak_positions.coords[self.peak_positions.dims[0]].values[y]
        x_vals = self.peak_positions.coords[self.peak_positions.dims[1]].values[x]
        
        # Transform xdata and ydata to be in the same coordinate system as the scatter points
        trans_data = self.ax.transData.inverted()
        trans_event = trans_data.transform((event.x, event.y))
        
        # Find the closest point to the clicked position
        distances = np.sqrt((x_vals - trans_event[0])**2 + (y_vals - trans_event[1])**2)
        closest_index = np.argmin(distances)
        
        # Get coordinates of the closest point
        closest_x = x_vals[closest_index]
        closest_y = y_vals[closest_index]
        
        # Convert to index in DataArray
        ix = self.find_closest(self.peak_positions.coords[self.peak_positions.dims[1]].values, closest_x)
        iy = self.find_closest(self.peak_positions.coords[self.peak_positions.dims[0]].values, closest_y)
        
        # Remove the point
        self.peak_positions.loc[{self.peak_positions.dims[0]: iy, self.peak_positions.dims[1]: ix}] = 0
        
        # Update the scatter plot
        self.update_scatter()
        self.update_table()
    
    def toggle_selector(self, event):
        ''' 
        toggle_selector:
            Purpose:
            Toggles the RectangleSelector on and off based on key press.

            Implementation:
            Listens for 'Q' and 'A' key presses to deactivate or activate the RectangleSelector.

            Considerations:
            The RectangleSelector must be initialized before this function is useful.

            Attributes:
            RS (RectangleSelector): Updated based on whether it is activated or deactivated.
        '''

        print(' Key pressed.')
        if event.key in ['Q', 'q'] and self.toggle_selector.RS.active:
            print(' RectangleSelector deactivated.')
            self.toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not self.toggle_selector.RS.active:
            print(' RectangleSelector activated.')
            self.toggle_selector.RS.set_active(True)

## -- Rectangular Box Highlight Tool -- ##
    def line_select_callback(self, eclick, erelease):
        ''' 
        line_select_callback:
            Purpose:
            Callback function for when a rectangular region is selected.

            Implementation:
            Highlights the points within the rectangle and stores them.

            Considerations:
            Should be used as a callback for RectangleSelector.

            Attributes:
            rectangles (list): Updated to include the newly added rectangle.
            texts (list): Updated to include the newly added 'X' text object.
            highlighted_indices (list): Updated to include the indices of newly highlighted points.
        '''

        'eclick and erelease are the press and release events'
        for rect in self.rectangles:
            rect.remove()
        self.rectangles = []  # Clear the existing rectangles
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")

        self.highlight_points(x1, y1, x2, y2)
        self.mpl_connect('pick_event', self.on_pick)  # Add this line

        # Add rectangle
        rect = Rectangle((min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2),
                         fill=False, edgecolor='red', linewidth=1)
        self.rectangles.append(rect)
        self.ax.add_patch(rect)
        
        # Create and add the 'X' text object
        x_text = min(x1, x2)
        y_text = min(y1, y2)
        text = self.ax.text(x_text, y_text, 'X', color='black')
        text.set_picker(True)  # Make it pickable

        # Add it to a list of text objects (make sure to initialize this list in __init__)
        self.texts.append(text)

        self.draw()

    def highlight_points(self, x1, y1, x2, y2):
        ''' 
        highlight_points:
            Purpose:
            Highlights the points within a given rectangular region.

            Implementation:
            Changes the color of the points within the rectangle.

            Considerations:
            Called internally by line_select_callback.

            Attributes:
            highlighted_indices (list): Updated to include the indices of newly highlighted points.
        '''

        xdata, ydata = self.scatter.get_offsets().T
        facecolors = self.scatter.get_facecolors()
        new_highlighted_indices = []  # Points in the new rectangle
        
        # Identify the points that are within the new rectangle
        for i, (x, y) in enumerate(zip(xdata, ydata)):
            if min(x1, x2) < x < max(x1, x2) and min(y1, y2) < y < max(y1, y2):
                new_highlighted_indices.append(i)

        # Update self.highlighted_indices to include the new points
        self.highlighted_indices = list(set(self.highlighted_indices).union(new_highlighted_indices))

        # Update the colors based on self.highlighted_indices
        for i in range(len(facecolors)):
            if i in self.highlighted_indices:
                facecolors[i] = [0, 1, 1, 1]  # Cyan color
            else:
                facecolors[i] = [1, 0, 0, 1]  # Red color
        
        self.scatter.set_facecolors(facecolors)
        self.draw()

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
        self.fig.canvas.mpl_connect('key_press_event', self.toggle_selector)
        self.fig.canvas.mpl_connect('button_press_event', self.on_rect_pick)

    def on_rect_pick(self, event):
        ''' 
        on_rect_pick:
            Purpose:
            Handles the event when a rectangle is clicked.

            Implementation:
            Removes the clicked rectangle.

            Considerations:
            Rectangle picking should be enabled for this to work.

            Attributes:
            rectangles (list): Updated to remove the clicked rectangle.
        '''

        if isinstance(event.artist, Rectangle):
            event.artist.remove()  # Remove the rectangle
            self.rectangles.remove(event.artist)  # Remove the rectangle from the list
            self.draw()
    
    def on_pick(self, event):
        ''' 
        on_pick:
            Purpose:
            Handles the pick event for 'X' text objects.

            Implementation:
            Removes the corresponding rectangle and the 'X' text object.

            Considerations:
            Text objects should be set as pickable for this to work.

            Attributes:
            rectangles (list): Updated to remove the rectangle corresponding to the clicked 'X'.
            texts (list): Updated to remove the clicked 'X' text object.
        '''

        if isinstance(event.artist, Text):
            index = self.texts.index(event.artist)
            rect = self.rectangles[index]
            rect.remove()
            self.rectangles.pop(index)
            event.artist.remove()
            self.texts.pop(index)
            self.draw()

    def export_data(self, filepath):
        ''' 
        export_data:
            Purpose:
            Exports the modified intensity and peak_positions to a netCDF4 (h5netcdf) file.

            Implementation:
            Uses xarray to save the DataArrays to a file.

            Considerations:
            Filepath should be valid and writable.
            
            Attributes:
            intensity, peak_positions
        '''
        ds = xr.Dataset({'intensity': self.intensity, 'peak_positions': self.peak_positions})
        ds.to_netcdf(filepath, engine='h5netcdf')

    def update_color_scale(self, vmin, vmax):
        if len(self.ax.images) == 0:  # Check if images are present
            return
        self.ax.images[0].set_clim(vmin, vmax)  # Set the color limits of the image directly
        self.colorbar.update_normal(self.ax.images[0])  # Update the colorbar
        self.draw()  # Redraw the canvas

    '''
    def update_table(self):
        self.tableWidget.setSortingEnabled(False)  # Disable sorting before update

        if self.intensity is None or self.peak_positions is None:
            print("Intensity or peak_positions is not initialized.")
            return
        
        y, x = np.where(self.peak_positions.values == 1)
        chi_values = self.peak_positions.coords[self.peak_positions.dims[0]].values[y]
        qr_values = self.peak_positions.coords[self.peak_positions.dims[1]].values[x]

        # Convert chi to radians and calculate qxy and qz
        chi_rad = np.radians(90 - chi_values)
        qxy_values = qr_values * np.cos(chi_rad)
        qz_values = qr_values * np.sin(chi_rad)
        # d_values = (2*np.pi)/qr_values
        d_values = np.round((2 * np.pi) / qr_values, decimals=2).astype(np.float64)

        # Fetch intensity values
        intensity_values = [self.intensity.sel({self.peak_positions.dims[0]: chi, self.peak_positions.dims[1]: qr}).values.item() for chi, qr in zip(chi_values, qr_values)]

        # Assert that all arrays have the same length
        assert len(qr_values) == len(chi_values) == len(intensity_values), "Data length mismatch"

        # Set the number of rows and columns
        num_rows = len(qr_values)
        self.tableWidget.setRowCount(num_rows)
        self.tableWidget.setColumnCount(7)

        # Set the column headers
        self.tableWidget.setHorizontalHeaderLabels(['qᵣ (Å⁻¹)', '𝛘 (°)', 'qxy (Å⁻¹)', 'qz (Å⁻¹)', 'd-spacing (Å)', 'Intensity'])

        # Populate the table
        for i in range(num_rows):
            item0 = QTableWidgetItem("{:.2f}".format(qr_values[i]))
            item0.setData(Qt.UserRole, float(qr_values[i]))
            self.tableWidget.setItem(i, 0, item0)

            item1 = QTableWidgetItem("{:.2f}".format(chi_values[i]))
            item1.setData(Qt.UserRole, float(chi_values[i]))
            self.tableWidget.setItem(i, 1, item1)

            item2 = QTableWidgetItem("{:.2f}".format(qxy_values[i]))
            item2.setData(Qt.UserRole, float(qxy_values[i]))
            self.tableWidget.setItem(i, 2, item2)

            item3 = QTableWidgetItem("{:.2f}".format(qz_values[i]))
            item3.setData(Qt.UserRole, float(qz_values[i]))
            self.tableWidget.setItem(i, 3, item3)

            item4 = QTableWidgetItem("{:.2f}".format(d_values[i]))
            item4.setData(Qt.UserRole, float(d_values[i]))
            self.tableWidget.setItem(i, 4, item4)

            item5 = QTableWidgetItem("{:.2f}".format(intensity_values[i]))
            item5.setData(Qt.UserRole, float(intensity_values[i]))
            self.tableWidget.setItem(i, 5, item5)
    
            hidden_item = QTableWidgetItem()
            hidden_item.setData(Qt.UserRole, i)  # Store the scatter point index
            self.tableWidget.setItem(i, 6, hidden_item)  # Column 6 will be the hidden column

        self.tableWidget.setSortingEnabled(True)  # Re-enable sorting after update
    '''

    def update_table(self):
        self.tableWidget.setSortingEnabled(False)
        if self.intensity is None or self.peak_positions is None:
            print("Intensity or peak_positions is not initialized.")
            return
        y, x = np.where(self.peak_positions.values == 1)
        chi_values = self.peak_positions.coords[self.peak_positions.dims[0]].values[y]
        qr_values = self.peak_positions.coords[self.peak_positions.dims[1]].values[x]
        chi_rad = np.radians(90 - chi_values)
        qxy_values = qr_values * np.cos(chi_rad)
        qz_values = qr_values * np.sin(chi_rad)
        d_values = np.round((2 * np.pi) / qr_values, decimals=2).astype(np.float64)
        intensity_values = [self.intensity.sel({self.peak_positions.dims[0]: chi, self.peak_positions.dims[1]: qr}).values.item() for chi, qr in zip(chi_values, qr_values)]
        num_rows = len(qr_values)
        self.tableWidget.setRowCount(num_rows)
        self.tableWidget.setColumnCount(7)
        self.tableWidget.setHorizontalHeaderLabels(['qᵣ (Å⁻¹)', '𝛘 (°)', 'qxy (Å⁻¹)', 'qz (Å⁻¹)', 'd-spacing (Å)', 'Intensity'])
        
        for i in range(num_rows):
            item0 = QTableWidgetItem("{:.2f}".format(qr_values[i]))
            item0.setData(Qt.UserRole, float(qr_values[i]))
            self.tableWidget.setItem(i, 0, item0)

            item1 = QTableWidgetItem("{:.2f}".format(chi_values[i]))
            item1.setData(Qt.UserRole, float(chi_values[i]))
            self.tableWidget.setItem(i, 1, item1)

            item2 = QTableWidgetItem("{:.2f}".format(qxy_values[i]))
            item2.setData(Qt.UserRole, float(qxy_values[i]))
            self.tableWidget.setItem(i, 2, item2)

            item3 = QTableWidgetItem("{:.2f}".format(qz_values[i]))
            item3.setData(Qt.UserRole, float(qz_values[i]))
            self.tableWidget.setItem(i, 3, item3)

            item4 = QTableWidgetItem("{:.2f}".format(d_values[i]))
            item4.setData(Qt.UserRole, float(d_values[i]))
            self.tableWidget.setItem(i, 4, item4)

            item5 = QTableWidgetItem("{:.2f}".format(intensity_values[i]))
            item5.setData(Qt.UserRole, float(intensity_values[i]))
            self.tableWidget.setItem(i, 5, item5)
    
            hidden_item = QTableWidgetItem()
            hidden_item.setData(Qt.UserRole, i)
            self.tableWidget.setItem(i, 6, hidden_item)

        self.tableWidget.setSortingEnabled(True)
        self.tableWidget.setColumnHidden(6, True)  # Hide the last column again

    def highlight_scatter_point(self, row):
        hidden_item = self.tableWidget.item(row, 6)
        if hidden_item:
            point_index = hidden_item.data(Qt.UserRole)
            facecolors = self.scatter.get_facecolors()
            facecolors[point_index] = [0, 1, 1, 1]  # Change to cyan
            self.scatter.set_facecolors(facecolors)
            self.draw()  # Redraw the canvas

    def on_table_selection_change(self, selected, deselected):
        facecolors = self.scatter.get_facecolors()

        # Handle deselected rows
        deselected_rows = list(set(index.row() for index in deselected.indexes()))
        for row in deselected_rows:
            hidden_item = self.tableWidget.item(row, 6)
            if hidden_item:
                point_index = int(hidden_item.data(Qt.UserRole))
                facecolors[point_index, :3] = [1, 0, 0]  # Set deselected to red

        # Handle newly selected rows
        selected_rows = list(set(index.row() for index in selected.indexes()))
        for row in selected_rows:
            hidden_item = self.tableWidget.item(row, 6)
            if hidden_item:
                point_index = int(hidden_item.data(Qt.UserRole))
                facecolors[point_index, :3] = [0, 1, 1]  # Set selected to cyan

        self.scatter.set_facecolors(facecolors)
        self.draw()  # Redraw the canvas

# Window Layout & Figure Updating Methods (pairs with the control methods)
class MyWindow(QMainWindow):
    # ============ Initialization Methods ============
    def __init__(self):
        ''' 
        __init__:
            Purpose:
            Initializes an instance of the MyWindow class, setting up the UI and other instance variables.

            Implementation:
            Calls the initUI method to set up the UI and initializes event connection IDs to None.
            
            Considerations:
            Ensure this is the first method called in any subclass.
            
            Attributes:
            ds (xarray.Dataset): Holds the loaded dataset. Initialized as None.
            add_point_cid (int): Holds the connection ID for adding points. Initialized as None.
            remove_point_cid (int): Holds the connection ID for removing points. Initialized as None.
        '''

        super(MyWindow, self).__init__()
        self.initUI()
        self.ds = None # dataset values
        # self.canvas.set_dataset(self.ds)  # Update ds in MyCanvas
        self.add_point_cid = None
        self.remove_point_cid = None
        self.waxs_reduce = None  # Initialize WAXSReduce
        self.resize(1000, 750)  # Set the size to 800x600 pixels

    def initUI(self):
        ''' 
        initUI:
            Purpose:
            Sets up the initial UI of the MyWindow class.

            Implementation:
            Initializes the canvas, toolbar, and buttons. Sets up the layout and connects actions.
            
            Considerations:
            Make sure to call this after the __init__ method.
            
            Attributes:
            canvas (MyCanvas): An instance of the MyCanvas class, used for plotting and interactions.
            toolbar (NavigationToolbar): The toolbar containing navigation options for the canvas.
        '''

        self.canvas = MyCanvas()
        self.toolbar = MyNavigationToolbar(self.canvas, self) # self to canvas and parent canvas
        
        # Connect built-in toolbar buttons to a custom slot
        for action in self.toolbar.actions():
            if action.text() in ['Pan', 'Zoom']:
                action.triggered.connect(self.deactivate_point_buttons)

        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.realpath(__file__))

        # Construct full icon paths
        add_point_icon_path = os.path.join(script_dir, 'icons/addpoint_icon.png')
        remove_point_icon_path = os.path.join(script_dir, 'icons/removepoint_icon.png')
        highlight_icon_path = os.path.join(script_dir, 'icons/highlight_icon.png')

        btn_add_point = QPushButton()
        btn_add_point.setIcon(QIcon(add_point_icon_path))
        btn_add_point.setToolTip('Add Point')
        btn_add_point.clicked.connect(self.activateAddPoint)
        
        btn_remove_point = QPushButton()
        btn_remove_point.setIcon(QIcon(remove_point_icon_path))
        btn_remove_point.setToolTip('Remove Point')
        btn_remove_point.clicked.connect(self.activateRemovePoint)
        
        btn_highlight = QPushButton()
        btn_highlight.setIcon(QIcon(highlight_icon_path))
        btn_highlight.setToolTip('Highlight Selection')
        btn_highlight.clicked.connect(self.activateHighlight)

        # Create a grid layout
        layout = QGridLayout()

        # Create a vertical widget for the three buttons
        button_group = QGroupBox("Peak Selection Tools")
        vlayout = QVBoxLayout()
        vlayout.addWidget(btn_add_point)
        vlayout.addWidget(btn_remove_point)
        vlayout.addWidget(btn_highlight)
        button_group.setLayout(vlayout)

        # Place widgets in the grid layout
        # LAYOUT FORMATTING ---- Layout Positions: addWidget(QWidget, row, column, rowSpan, columnSpan)
        # layout.addWidget(self.toolbar, 0, 0, 1, 3)  # Span 3 columns
        # layout.addWidget(button_group, 1, 0, 1, 1)  # Buttons on the left
        # layout.addWidget(self.canvas, 1, 1, 2, 2)  # Canvas on the right
        
        # Create sliders for colorbar adjustment
        self.slider_vmin = QSlider(Qt.Horizontal)
        self.slider_vmin.setMinimum(0)
        self.slider_vmin.setMaximum(100)
        self.slider_vmin.valueChanged.connect(self.update_vmin)

        self.slider_vmax = QSlider(Qt.Horizontal)
        self.slider_vmax.setMinimum(0)
        self.slider_vmax.setMaximum(100)
        self.slider_vmax.valueChanged.connect(self.update_vmax)

        self.canvas.tableWidget = QTableWidget()
        self.canvas.tableWidget.setColumnCount(6)
        self.canvas.tableWidget.setHorizontalHeaderLabels(['qᵣ (Å⁻¹)', '𝛘 (°)', 'qxy (Å⁻¹)', 'qz (Å⁻¹)', 'd-spacing (Å)', 'Intensity'])

        # Add the table widget to the layout
        layout.addWidget(self.canvas.tableWidget, 3, 1, 1, 3)
        # layout.addWidget(self.canvas.tableWidget, 3, 1, 1, 1)

        # Create a vertical widget for the Clear Selection, Enable Selection Mode, and Export Table buttons
        selection_group = QGroupBox("Selection Tools")
        selection_vlayout = QVBoxLayout()

        self.clearButton = QPushButton('Clear Selection')
        self.clearButton.clicked.connect(self.clear_selection)

        self.selectModeButton = QRadioButton('Enable Selection Mode')
        self.selectModeButton.setChecked(False)
        self.selectModeButton.toggled.connect(self.toggle_selection_mode)

        btn_export_table = QPushButton("Export Table")
        btn_export_table.clicked.connect(self.export_table_to_csv)

        # Add these new buttons to the vertical layout
        selection_vlayout.addWidget(self.selectModeButton)
        selection_vlayout.addWidget(self.clearButton)
        selection_vlayout.addWidget(btn_export_table)

        selection_group.setLayout(selection_vlayout)

        # Add the QGroupBox to your existing grid layout
        layout.addWidget(selection_group, 3, 0, 1, 1)  # Adjust the grid position as needed

        # Add widgets to layout
        layout.addWidget(self.toolbar, 0, 0, 1, 3)
        layout.addWidget(button_group, 1, 0, 1, 1)
        layout.addWidget(self.canvas, 1, 1, 2, 2)

        # Adjust column widths
        layout.setColumnStretch(0, 1)  # 8% of the width
        layout.setColumnStretch(1, 11)  # 92% of the width

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        self.setWindowTitle('pyWAXS: Main Menu')
        self.show()

        # Here we connect the toolbar to deactivate_point_buttons
        for action in self.toolbar.actions():
            action.triggered.connect(self.deactivate_point_buttons)
        
        # Inside MyWindow.__init__ after canvas initialization
        self.canvas.tableWidget.clicked.connect(self.cell_was_clicked)
        self.canvas.tableWidget.itemSelectionChanged.connect(self.row_was_selected)

        # Create the custom toolbar
        self.customToolbar = QToolBar("My Custom Toolbar")
        self.addToolBar(Qt.LeftToolBarArea, self.customToolbar)
        
        # Migrate the Create Calibrant button
        script_dir = os.path.dirname(os.path.realpath(__file__))
        create_calibrant_icon_path = os.path.join(script_dir, 'icons/create_calibrant_icon.png')
        action_create_calibrant = QAction(QIcon(create_calibrant_icon_path), "Create Calibrant", self)
        action_create_calibrant.triggered.connect(self.create_calibrant)
        self.customToolbar.addAction(action_create_calibrant)

        # Add a button to launch the new application with an icon
        launch_sim_icon_path = os.path.join(script_dir, 'icons/launch_sim_icon.png')
        action_launch_sim = QAction(QIcon(launch_sim_icon_path), "GIWAXS Simulation App", self)
        action_launch_sim.triggered.connect(self.launch_sim_app)
        self.customToolbar.addAction(action_launch_sim)

# ============ UI Update Methods ============
    def update_vmin(self):
        value = self.slider_vmin.value()
        current_vmax = self.canvas.ax.images[0].get_clim()[1]
        self.canvas.update_color_scale(value, current_vmax)

    def update_vmax(self):
        value = self.slider_vmax.value()
        current_vmin = self.canvas.ax.images[0].get_clim()[0]
        self.canvas.update_color_scale(current_vmin, value)

# ============ Data Handling Methods ============
    def loadData(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "Load Data", "", "NetCDF Files (*.nc);;All Files (*)", options=options)
        if file:
            self.ds = xr.open_dataset(file, engine='h5netcdf')
            
            # Extract coordinate names
            coord_names_intensity = list(self.ds['intensity'].dims)
            coord_names_peak = list(self.ds['peak_positions'].dims)
            
            coords = {
                'xlabel': coord_names_intensity[1] if len(coord_names_intensity) > 1 else None,  
                'ylabel': coord_names_intensity[0] if len(coord_names_intensity) > 0 else None,
            }

            self.canvas.plot_data(self.ds['intensity'], self.ds['peak_positions'], coords)
        
        if self.ds:
            max_intensity = self.ds['intensity'].max()
            max_intensity = int(max_intensity.values)
            self.slider_vmin.setMaximum(max_intensity)
            self.slider_vmax.setMaximum(max_intensity)

        print("Loaded dataset:", self.ds)
        print("Intensity dimensions:", self.ds['intensity'].dims)
        print("Peak_positions dimensions:", self.ds['peak_positions'].dims)

        # self.canvas.set_dataset(self.ds)  # Update ds in MyCanvas

    def exportData(self):
        ''' 
        exportData:
            Purpose:
            Opens a file dialog for exporting data and calls the export_data method from MyCanvas.

            Implementation:
            Uses QFileDialog to get the filepath and then invokes MyCanvas.export_data.

            Considerations:
            Assumes that MyCanvas instance is available as self.canvas.
            
            Attributes:
            ds
        '''
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getSaveFileName(self, "Export Data", "", "NetCDF Files (*.nc);;All Files (*)", options=options)
        if filepath:
            if not filepath.endswith('.nc'):
                filepath += '.nc'
            self.canvas.export_data(filepath)

    def store_waxs_reduce(self, waxs_reduce_instance):
        self.waxs_reduce = waxs_reduce_instance  # Store the instance in the MyWindow class

    def export_table_to_csv(self):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getSaveFileName(self, "Export Table", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if filepath:
            if not filepath.endswith('.csv'):
                filepath += '.csv'
            
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write headers
                headers = [self.canvas.tableWidget.horizontalHeaderItem(i).text() for i in range(self.canvas.tableWidget.columnCount())]
                writer.writerow(headers)
                
                # Write data
                for i in range(self.canvas.tableWidget.rowCount()):
                    row_data = [self.canvas.tableWidget.item(i, j).text() if self.canvas.tableWidget.item(i, j) is not None else '' for j in range(self.canvas.tableWidget.columnCount())]
                    writer.writerow(row_data)

# ============ Event Handling Methods ============
    def row_was_selected(self):
        if self.selectModeButton.isChecked():  # Only proceed if in selection mode
            selected_rows = list(set(index.row() for index in self.canvas.tableWidget.selectedItems()))
            for row in selected_rows:
                self.canvas.highlight_scatter_point(row)

    def cell_was_clicked(self, index):
        print(f"Cell clicked: Row {index.row()} Column {index.column()}")

# ============ Point Manipulation Methods ============
    def activateAddPoint(self):
        ''' 
        activateAddPoint:
            Purpose:
            Activates the event connection for adding points.

            Implementation:
            Connects the button_press_event to the add_point method of the MyCanvas class.
            
            Considerations:
            Make sure to deactivate other conflicting functionalities like remove point.
            
            Attributes:
            add_point_cid (int): Holds the new connection ID for adding points.
        '''

        self.deactivateRemovePoint()
        self.add_point_cid = self.canvas.mpl_connect('button_press_event', self.canvas.add_point)

    def deactivateAddPoint(self):
        ''' 
        deactivateAddPoint:
            Purpose:
            Deactivates the event connection for adding points.

            Implementation:
            Disconnects the button_press_event connection for adding points.
            
            Considerations:
            Call this method when you want to disable the add point functionality.
            
            Attributes:
            add_point_cid (int): Set to None, effectively deactivating the connection for adding points.
        '''

        if self.add_point_cid is not None:
            self.canvas.mpl_disconnect(self.add_point_cid)
            self.add_point_cid = None

    def activateRemovePoint(self):
        ''' 
        activateRemovePoint:
            Purpose:
            Activates the event connection for removing points.

            Implementation:
            Connects the button_press_event to the remove_point method of the MyCanvas class.
            
            Considerations:
            Make sure to deactivate other conflicting functionalities like add point.
            
            Attributes:
            remove_point_cid (int): Holds the new connection ID for removing points.
        '''

        self.deactivateAddPoint()
        self.remove_point_cid = self.canvas.mpl_connect('button_press_event', self.canvas.remove_point)

    def deactivateRemovePoint(self):
        ''' 
        deactivateRemovePoint:
            Purpose:
            Deactivates the event connection for removing points.

            Implementation:
            Disconnects the button_press_event connection for removing points.
            
            Considerations:
            Call this method when you want to disable the remove point functionality.
            
            Attributes:
            remove_point_cid (int): Set to None, effectively deactivating the connection for removing points.
        '''

        if self.remove_point_cid is not None:
            self.canvas.mpl_disconnect(self.remove_point_cid)
            self.remove_point_cid = None

    def deactivate_point_buttons(self):
        ''' 
        deactivate_point_buttons:
            Purpose:
            Deactivates all point manipulation functionalities.

            Implementation:
            Calls the deactivateAddPoint and deactivateRemovePoint methods.
            
            Considerations:
            Call this method when activating other functionalities like pan or zoom.
            
            Attributes:
            None
        '''

        self.deactivateAddPoint()
        self.deactivateRemovePoint()
    
        # Deactivate the RectangleSelector if it's active
        if self.canvas.RS and self.canvas.RS.active:
            self.canvas.RS.set_active(False)

    def activateHighlight(self):
        ''' 
        activateHighlight:
            Purpose:
            Activates the rectangle selection for highlighting points.

            Implementation:
            Calls the setup_rectangle_selector method of the MyCanvas class to set up the RectangleSelector.
            
            Considerations:
            Make sure to deactivate other conflicting functionalities.
            
            Attributes:
            None
        '''

        self.canvas.setup_rectangle_selector()

# ============ Selection Tools ============
    def clear_selection(self):
        self.canvas.tableWidget.clearSelection()
        facecolors = self.canvas.scatter.get_facecolors()
        facecolors[:, :3] = [1, 0, 0]  # Revert all points to red
        self.canvas.scatter.set_facecolors(facecolors)
        self.canvas.draw()

    def toggle_selection_mode(self, enabled):
        if enabled:
            self.canvas.scatter.set_picker(5)  # Enable picking
        else:
            self.canvas.scatter.set_picker(None)  # Disable picking

# ============ Custom Side-Toolbar Methods ============
    # Slot for Create Calibrant
    def create_calibrant(self):
        try:
            subprocess.run(["bash", "-c", "eval \"$(conda shell.bash hook)\" && conda activate pyWAXS && pyFAI-calib2"])
        except Exception as e:
            print(f"An error occurred: {e}")
    
    # Slot to launch the new application
    def launch_sim_app(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
        try:
            subprocess.run(["bash", "-c", f"cd {script_dir} && eval \"$(conda shell.bash hook)\" && conda activate pyWAXS && python3 pyWAXSim.py"])
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Set the global QFont to Arial
    font = QFont("Arial")
    app.setFont(font)

    window = MyWindow()
    sys.exit(app.exec_())