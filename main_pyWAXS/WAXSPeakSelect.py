from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy, QPushButton, QVBoxLayout, QWidget, QMenuBar, QMenu, QAction, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import xarray as xr
import numpy as np
import sys, json

class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=10, dpi=100):  # Changed dimensions
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class WAXSPeakSelect(QMainWindow):
    def __init__(self):
        # super().__init__()
        super(WAXSPeakSelect, self).__init__()

        # Initialize DataArray
        # self.data = xr.DataArray(np.random.rand(20, 20), dims=('x', 'y'))
        # self.data.attrs['peaks'] = xr.DataArray(np.full((20, 20), np.nan), dims=('x', 'y'))

        # Create the QAction for loading data
        self.load_action = QAction('Load Data', self)

        # Create a canvas and plot initial data
        self.canvas = MyMplCanvas(self, width=10, height=10, dpi=100)
        # self.update_plot()

        # Create a toolbar for zoom, pan, etc.
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Create a button
        self.button = QPushButton('Save Peaks')
        self.button.clicked.connect(self.save_peaks)

        # Connect the slot for loading data
        self.load_action.triggered.connect(self.load_data)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.button)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Create Menu
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        export_action = QAction('Export to Zarr', self)
        export_action.triggered.connect(self.export_to_zarr)
        file_menu.addAction(export_action)

        # Add Load Data action
        file_menu.addAction(self.load_action)

        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.onclick)

    def update_plot(self):
        self.canvas.axes.clear()
        coords = list(self.data.coords.keys())
        if len(coords) == 2:
            coord1, coord2 = coords

            # Plot the intensity data
            extent = [
                self.data.coords[coord1].min(), 
                self.data.coords[coord1].max(), 
                self.data.coords[coord2].min(), 
                self.data.coords[coord2].max()
            ]

            im = self.canvas.axes.imshow(self.data, cmap='turbo', aspect='auto', extent=extent)

            # Overlay peaks
            peaks = self.data.attrs.get('peaks', None)
            if peaks is not None:
                peak_coords = np.column_stack(np.where(~np.isnan(peaks)))
                self.canvas.axes.scatter(
                    self.data.coords[coord1][peak_coords[:, 0]], 
                    self.data.coords[coord2][peak_coords[:, 1]], 
                    c='red'
                )
                
            # Update geometry and redraw
            self.canvas.fig.set_size_inches(10, 10, forward=True)  
            self.canvas.draw()
            self.canvas.updateGeometry()

        else:
            print("Unsupported number of coordinates. Expected 2.")

    def onclick(self, event):
        x, y = int(event.xdata), int(event.ydata)
        if event.key == 'shift':  # Remove peak
            self.data.attrs['peaks'][y, x] = np.nan
        else:  # Add peak
            self.data.attrs['peaks'][y, x] = self.data[y, x]
        self.update_plot()

    def export_to_zarr(self):
        print("Exporting to Zarr file.")
        self.data.to_zarr("peaks_data.zarr")

    def load_data(self):
        options = QFileDialog.Options()
        filePath = QFileDialog.getExistingDirectory(self, "Load Data", "", options=options)
        if filePath:
            # Load the Dataset from the Zarr file
            ds = xr.open_zarr(filePath)
            
            # Load the DataArray for intensity
            self.data = ds['intensity']
            
            # Deserialize attributes
            deserialized_attrs = {}
            for k, v in self.data.attrs.items():
                try:
                    original_value = json.loads(v)
                except json.JSONDecodeError:
                    original_value = v
                    
                if isinstance(original_value, dict) and 'dims' in original_value:
                    deserialized_attrs[k] = xr.DataArray.from_dict(original_value)
                else:
                    deserialized_attrs[k] = original_value
                    
            self.data.attrs = deserialized_attrs

            # Load and deserialize 'peaks' from the 'class_attrs'
            class_attrs_json = ds['class_attrs'].attrs.get('class_attrs', None)
            if class_attrs_json:
                class_attrs = json.loads(class_attrs_json)
                if 'peaks' in class_attrs:
                    peaks_dict = class_attrs['peaks']
                    self.data.attrs['peaks'] = xr.DataArray.from_dict(peaks_dict)
                    
            # Update the plot
            self.update_plot()

    def save_peaks(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getSaveFileName(self,"Save Peaks", "","Zarr Files (*.zarr);;All Files (*)", options=options)
        if filePath:
            # Update the peaks in the DataArray's attributes
            self.data_xr.attrs['peaks'] = self.peaks
            self.data_xr.to_dataset(name='intensity').to_zarr(filePath, mode='w')

# Initialize the application
app = QApplication(sys.argv)
ex = WAXSPeakSelect()
ex.setWindowTitle('WAXS Peak Selection')
ex.show()
sys.exit(app.exec_())


'''
    # def load_data(self):
    #     # options = QFileDialog.Options()
    #     # filePath, _ = QFileDialog.getOpenFileName(self, "Load Data", "", "Zarr Files (*.zarr);;All Files (*)", options=options)
    #     # if filePath:
    #     options = QFileDialog.Options()
    #     filePath = QFileDialog.getExistingDirectory(self, "Load Data", "", options=options)
    #     if filePath:
    #         # Load the Dataset from the Zarr file
    #         ds = xr.open_zarr(filePath)
            
    #         # Load the DataArray for intensity
    #         self.data = ds['intensity']
            
    #         # Deserialize attributes
    #         deserialized_attrs = {}
    #         for k, v in self.data.attrs.items():
    #             try:
    #                 original_value = json.loads(v)
    #             except json.JSONDecodeError:
    #                 original_value = v  # If it's not a JSON string, it's likely a simple type that doesn't need deserialization

    #             if isinstance(original_value, dict) and 'dims' in original_value:
    #                 # Deserialize DataArray from dictionary
    #                 deserialized_attrs[k] = xr.DataArray.from_dict(original_value)
    #             elif isinstance(original_value, list):
    #                 deserialized_attrs[k] = np.array(original_value)  # Convert list back to ndarray
    #             else:
    #                 deserialized_attrs[k] = original_value
                    
    #         # Restore the original attributes
    #         self.data.attrs = deserialized_attrs

    #         # Load and deserialize 'peaks' from the 'class_attrs'
    #         class_attrs_json = ds['class_attrs'].attrs.get('class_attrs', None)
    #         if class_attrs_json:
    #             class_attrs = json.loads(class_attrs_json)
    #             if 'peaks' in class_attrs:
    #                 peaks_dict = class_attrs['peaks']
    #                 self.data.attrs['peaks'] = xr.DataArray.from_dict(peaks_dict)

    #         # Update the plot
    #         self.update_plot()

    # def update_plot(self):
    #     self.canvas.axes.clear()
    #     coords = list(self.data.coords.keys())
    #     if len(coords) == 2:
    #         coord1, coord2 = coords
    #         self.canvas.fig.set_size_inches(10, 10)  # Set the figure size
    #         self.canvas.axes.imshow(self.data, cmap='turbo', aspect='auto', extent=[self.data.coords[coord1].min(), self.data.coords[coord1].max(), self.data.coords[coord2].min(), self.data.coords[coord2].max()])
            
    #         peaks = self.data.attrs['peaks']
    #         if peaks is not None:
    #             peak_coords = np.column_stack(np.where(~np.isnan(peaks)))
    #             self.canvas.axes.scatter(self.data.coords[coord1][peak_coords[:, 0]], self.data.coords[coord2][peak_coords[:, 1]], c='red')
    #     else:
    #         print("Unsupported number of coordinates. Expected 2.")
        
    #     self.canvas.draw()
'''
    