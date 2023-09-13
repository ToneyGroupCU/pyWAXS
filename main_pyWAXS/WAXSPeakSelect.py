from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QGridLayout, QWidget, QFileDialog, QGroupBox, QVBoxLayout
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.text import Text
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle  # Make sure to import Rectangle
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import xarray as xr
import numpy as np
import sys

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
        self.highlighted_points = []  # To keep track of highlighted points
        self.highlighted_indices = []  # Initialize it here
        self.rectangles = []  # To keep track of rectangles
        self.selected_rect = None  # To keep track of the selected rectangle
        # self.facecolors = None  # To keep track of the colors of scatter points
        self.RS = None  # Add this line to create a new attribute for RectangleSelector
        self.texts = []  # To keep track of 'X' text objects
        self.cid_pick = self.mpl_connect('pick_event', self.on_rect_pick)

    def plot_data(self, intensity, peak_positions):
        ''' 
        plot_data:
            Purpose:
            Plot the 2D intensity data and marks the peak positions.

            Implementation:
            Clears the existing axis.
            Plots the 2D heatmap using imshow.
            Plots the scatter points for peak positions.

            Considerations:
            Expects intensity and peak_positions to be xarray DataArrays with specific dimensions.

            Attributes:
            intensity (xarray.DataArray): Updated with the newly plotted intensity data.
            peak_positions (xarray.DataArray): Updated with the newly plotted peak positions.
            scatter (matplotlib.collections.PathCollection): Updated scatter plot object for peak positions.
            facecolors (numpy.ndarray): Array to keep track of the colors of scatter points.
        '''

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
        
        # Initialize facecolors
        num_points = len(x_vals)
        initial_colors = np.full((num_points, 4), [1, 0, 0, 1])  # Initial color is red for all points
        
        self.scatter = self.ax.scatter(x_vals, y_vals, facecolors=initial_colors)
        self.facecolors = initial_colors
        self.draw()
        self.intensity = intensity
        self.peak_positions = peak_positions

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

class MyWindow(QMainWindow):
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
        self.ds = None
        self.add_point_cid = None
        self.remove_point_cid = None

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
        
        btn_highlight = QPushButton('Highlight Selection')
        btn_highlight.clicked.connect(self.activateHighlight)
        
        # Create a grid layout
        layout = QGridLayout()

        # Create a vertical widget for the three buttons
        button_group = QGroupBox("Actions")
        vlayout = QVBoxLayout()
        vlayout.addWidget(btn_add_point)
        vlayout.addWidget(btn_remove_point)
        vlayout.addWidget(btn_highlight)
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

    def loadData(self):
        ''' 
        loadData:
            Purpose:
            Opens a file dialog and loads the selected NetCDF data file into the application.

            Implementation:
            Uses QFileDialog to select a file and uses xarray to load the data. Then plots the data using the MyCanvas class.
            
            Considerations:
            The file should be in NetCDF format for proper loading.
            
            Attributes:
            ds (xarray.Dataset): Updated with the newly loaded dataset.
        '''

        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "Load Data", "", "NetCDF Files (*.nc);;All Files (*)", options=options)
        if file:
            self.ds = xr.open_dataset(file, engine='h5netcdf')
            self.canvas.plot_data(self.ds['intensity'], self.ds['peak_positions'])

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
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())