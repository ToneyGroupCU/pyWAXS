from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QGridLayout, QWidget, QFileDialog, QGroupBox, QVBoxLayout
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import xarray as xr
import numpy as np
import sys

class MyNavigationToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent, coordinates=True):
        super(MyNavigationToolbar, self).__init__(canvas, parent, coordinates)
        self.window = parent  # Assuming the parent window is passed as `parent`

    def trigger_tool(self, *args, **kwargs):
        super().trigger_tool(*args, **kwargs)
        self.window.deactivateAddPoint()
        self.window.deactivateRemovePoint()

class MyCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.peak_positions = None
        self.intensity = None  
        super(MyCanvas, self).__init__(self.fig)
        self.scatter = None
        
    def plot_data(self, intensity, peak_positions):
        self.ax.clear()
        
        # Get extents based on xarray coordinates
        extent = [
            intensity.coords[intensity.dims[1]].min(),
            intensity.coords[intensity.dims[1]].max(),
            intensity.coords[intensity.dims[0]].min(),
            intensity.coords[intensity.dims[0]].max(),
        ]
        
        self.ax.imshow(intensity.values, cmap='turbo', origin='lower', extent=extent, aspect='auto')
        
        # Find the coordinates where peak_positions is 1
        y, x = np.where(peak_positions.values == 1)
        
        # Convert to actual coordinate values
        y_vals = peak_positions.coords[peak_positions.dims[0]].values[y]
        x_vals = peak_positions.coords[peak_positions.dims[1]].values[x]
        
        # self.ax.scatter(x_vals, y_vals, c='red')
        self.scatter = self.ax.scatter(x_vals, y_vals, c='red')
        self.draw()
        self.intensity = intensity
        self.peak_positions = peak_positions

    def update_scatter(self):
        y, x = np.where(self.peak_positions.values == 1)
        y_vals = self.peak_positions.coords[self.peak_positions.dims[0]].values[y]
        x_vals = self.peak_positions.coords[self.peak_positions.dims[1]].values[x]
        self.scatter.set_offsets(np.c_[x_vals, y_vals])
        self.draw()

    @staticmethod
    def find_closest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def add_point(self, event):
        ix = self.find_closest(self.peak_positions.coords[self.peak_positions.dims[1]].values, event.xdata)
        iy = self.find_closest(self.peak_positions.coords[self.peak_positions.dims[0]].values, event.ydata)
        self.peak_positions.loc[{self.peak_positions.dims[0]: iy, self.peak_positions.dims[1]: ix}] = 1
        self.update_scatter()

    def remove_point(self, event):
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

    def highlight_points(self, event):
        # Placeholder for logic to move points
        pass

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.initUI()
        self.ds = None
        self.add_point_cid = None
        self.remove_point_cid = None

    def initUI(self):
        self.canvas = MyCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Connect built-in toolbar buttons to a custom slot
        for action in self.toolbar.actions():
            if action.text() in ['Pan', 'Zoom']:
                action.triggered.connect(self.deactivate_point_buttons)

        btn_load = QPushButton('Load Data')
        btn_load.clicked.connect(self.loadData)
        
        btn_add_point = QPushButton('Add Point')
        btn_add_point.clicked.connect(self.activateAddPoint)
        
        btn_remove_point = QPushButton('Remove Point')
        btn_remove_point.clicked.connect(self.activateRemovePoint)
        
        btn_highlight_selection = QPushButton('Highlight Selection')
        btn_highlight_selection.clicked.connect(self.activateHighlightSelection)
        
        # Create a grid layout
        layout = QGridLayout()

        # Create a vertical widget for the three buttons
        button_group = QGroupBox("Actions")
        vlayout = QVBoxLayout()
        vlayout.addWidget(btn_add_point)
        vlayout.addWidget(btn_remove_point)
        vlayout.addWidget(btn_highlight_selection)
        button_group.setLayout(vlayout)

        # Place widgets in the grid layout
        layout.addWidget(self.toolbar, 0, 0, 1, 2)  # Span 2 columns
        layout.addWidget(button_group, 1, 0)  # Buttons on the left
        layout.addWidget(self.canvas, 1, 1)  # Canvas on the right
        layout.addWidget(btn_load, 2, 0, 1, 2)  # Span 2 columns
        
        # Adjust column widths
        layout.setColumnStretch(0, 1)  # 8% of the width
        layout.setColumnStretch(1, 11)  # 92% of the width

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        self.setWindowTitle('2D Data Plotting and Manipulation')
        self.show()

        # Here we connect the toolbar to deactivate_point_buttons
        for action in self.toolbar.actions():
            action.triggered.connect(self.deactivate_point_buttons)

        # self.toolbar.toolmanager.connect('toolmanager_changed', self.deactivate_point_buttons)

    def loadData(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "Load Data", "", "NetCDF Files (*.nc);;All Files (*)", options=options)
        if file:
            self.ds = xr.open_dataset(file, engine='h5netcdf')
            self.canvas.plot_data(self.ds['intensity'], self.ds['peak_positions'])

    def deactivateAddPoint(self):
        if self.add_point_cid is not None:
            self.canvas.mpl_disconnect(self.add_point_cid)
            self.add_point_cid = None

    def deactivateRemovePoint(self):
        if self.remove_point_cid is not None:
            self.canvas.mpl_disconnect(self.remove_point_cid)
            self.remove_point_cid = None

    def activateAddPoint(self):
        self.deactivateRemovePoint()
        self.add_point_cid = self.canvas.mpl_connect('button_press_event', self.canvas.add_point)

    def activateRemovePoint(self):
        self.deactivateAddPoint()
        self.remove_point_cid = self.canvas.mpl_connect('button_press_event', self.canvas.remove_point)

    def deactivate_point_buttons(self):
        self.deactivateAddPoint()
        self.deactivateRemovePoint()

    def activateHighlightSelection(self):
        self.canvas.mpl_connect('button_press_event', self.canvas.move_point)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())