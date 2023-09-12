# Python Standard Libraries
import os
import sys

# External Libraries
import fabio
import numpy as np
import scipy as sp
import pyqtgraph as pg
from pyqtgraph import ROI, PlotWidget
from pyqtgraph import functions as fn
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QPushButton, QApplication, QFileDialog, QLabel, QWidget, QToolBar, QSizePolicy
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize
from pyqtgraph import ROI, Point
from pyqtgraph.graphicsItems.ROI import Handle #, addRotateHandle, addScaleHandle
from pyqtgraph.Qt import QtGui, QtCore
import glob2 as glob
from dask import delayed, compute
import dask.array as da
import dask
from dask.delayed import delayed

keylist=['solutionnum', 'chemistry', 'filtopt','purgerate', 'sub', 'solvol', 'sampnum', 'clocktime', 
                                                'xpos', 'thpos', 'exptime', 'scanid', 'framenum', 'det.ext']
wcard='*s_[0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]_maxs*'

pngpath_rotate = '/Users/keithwhite/github_repositories/giwaxs_suite/png_widgeticon/rotationicon.png'
pngpath_loadimage = '/Users/keithwhite/github_repositories/giwaxs_suite/png_widgeticon/loadimage.png'
pngpath_corrections = '/Users/keithwhite/github_repositories/giwaxs_suite/png_widgeticon/correctionsicon.png'
pngpath_imgstack = '/Users/keithwhite/github_repositories/giwaxs_suite/png_widgeticon/layeredstack.png'
pngpath_ponipath = '/Users/keithwhite/github_repositories/giwaxs_suite/png_widgeticon/ponifileload.png'
pngpath_maskfile = '/Users/keithwhite/github_repositories/giwaxs_suite/png_widgeticon/maskicon.png'

from gixsreduce import GIXSDataReduction

class SimpleImageViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # -- Instantiation for GIXSDataReduction Class
        self.gixs = GIXSDataReduction(wcard=wcard, 
                                       keylist=keylist, 
                                       maskdata=None, correctSolidAngle=True, polarization_factor=None, dark=None, flat=None)

        # self.gixs = GIXSDataReduction(wcard=wcard, keylist=keylist, maskdata=None, correctSolidAngle=True, polarization_factor=None, dark=None, flat=None)
        
        self.image_data = None # null attribute for image data
        self.gixssim_data = None # null attribute for simulated data

        # -- Define the Central Widget + Window Geometry
        self.setWindowTitle('GIXS Data Reduction Suite - Toney Group')
        self.setGeometry(100, 100, 1200, 800)
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)  # Change from QHBoxLayout to QVBoxLayout

        # # -- First Tab
        # self.imageView = pg.ImageView()
        # layout.addWidget(self.imageView)

        # Define tab widget
        self.tabWidget = QtWidgets.QTabWidget(self)
        layout.addWidget(self.tabWidget)

        # -- First Tab
        self.tab1 = QtWidgets.QWidget()
        self.tabWidget.addTab(self.tab1, "Single Image Viewer")

        self.imageView = pg.ImageView()
        layout1 = QtWidgets.QVBoxLayout(self.tab1)
        layout1.addWidget(self.imageView)
        self.tab1.setLayout(layout1)  # Ensure layout1 is applied to tab1


        # -- Setup Toolbar (First Tab)
        toolbar = self.addToolBar('ToolBar')
        self.setupLoadImageButton(toolbar)
        self.setupRotateImageButton(toolbar)
        self.setupPONIPath(toolbar)
        self.setupMASKPath(toolbar)
        self.setupLoadTiffStack(toolbar)
        self.setupOpenWindowButton(toolbar)
        self.setupColorMapComboBox(toolbar)
        
        # -- Add Plot for Radial Integration
        self.wedgeROI = None
        self.radialPlot = pg.PlotWidget()
        # layout.addWidget(self.radialPlot)  # Now this is added below the ImageView in the layout
        self.setupRadialPlot()
        # Define the data origin
        self.data_origin = [0, 0]  # initialized to [0, 0]

        # -- Add Radio Button for Azimuthal Integration Plot Display
        self.toggleRadialPlotButton = QtWidgets.QRadioButton("1D Integration", self)
        self.toggleRadialPlotButton.setToolTip('Display 1D Azimuthal Integration')  # Set tooltip text
        self.toggleRadialPlotButton.setChecked(True)
        self.toggleRadialPlotButton.toggled.connect(self.toggle_radial_plot)
        toolbar.addWidget(self.toggleRadialPlotButton)

        # -- Second Tab
        # Create the second tab
        self.tab2 = QtWidgets.QWidget()
        self.tabWidget.addTab(self.tab2, "Hyperspectral Image Viewer")
        
        # -- Add the image view to the second tab
        self.imageView2 = pg.ImageView()
        layout2 = QtWidgets.QVBoxLayout()
        layout2.addWidget(self.imageView2)
        self.tab2.setLayout(layout2)

        # -- Third Tab
        # Create the third tab
        self.tab3 = QtWidgets.QWidget()
        self.tabWidget.addTab(self.tab3, "GIXS Simulation")

        # Add two image views to the third tab
        self.imageView3_1 = pg.ImageView()
        self.imageView3_2 = pg.ImageView()
        layout3 = QtWidgets.QHBoxLayout(self.tab3)  # Use QHBoxLayout here
        layout3.addWidget(self.imageView3_1)
        layout3.addWidget(self.imageView3_2)
        self.tab3.setLayout(layout3)  # Ensure layout3 is applied to tab3

        '''
        # -- ADD WedgeROI
        # self.wedgeROI = WedgeROI(self.data_origin, 10, 50)
        # self.imageView.addItem(self.wedgeROI)

        # self.wedgeROI.sigRegionChangeFinished.connect(self.update_radial_plot)

        # -- SECOND WINDOW
        # Create a splitter to manage ImageView and radialPlot sizes
        # self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        # layout.addWidget(self.splitter)

        # self.setupImageView()  # now 'self.splitter' exists when this is called
        # layout.addWidget(self.imageView)

        # self.setupSecondImageView(layout)
        '''

    def setupImageView(self, layout):
        self.imageView = pg.ImageView()
        # self.splitter.addWidget(self.imageView)

    def setupSecondImageView(self, layout):  # This is a new method for setting up the second ImageView
        self.imageView2 = pg.ImageView()
        layout.addWidget(self.imageView2)
    
    def load_image2(self, array):
        self.imageView2.setImage(array)

    def setupPONIPath(self, toolbar):
        self.PONIPathButton = QtWidgets.QPushButton()
        self.PONIPathButton.setIcon(QtGui.QIcon(pngpath_ponipath))  # Add your icon here
        self.PONIPathButton.setIconSize(QtCore.QSize(25, 25))  # Adjust this to your desired icon size
        self.PONIPathButton.setFixedSize(40, 40)  # Adjust the size of the button
        self.PONIPathButton.setToolTip('Set PONI Path')  # Set tooltip text
        self.PONIPathButton.clicked.connect(self.set_poni_path)
        toolbar.addWidget(self.PONIPathButton)

    def set_poni_path(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open PONI", "", "PONI Files (*.poni);;All Files (*)")
        if file_path:
            try:
                self.gixs.open_poni_dialog(file_path)
            except Exception as e:
                print(f"Error in opening PONI file: {e}")

    def setupLoadImageButton(self, toolbar):  # The method now accepts a QToolBar instead of a QVBoxLayout
        self.loadImageButton = QtWidgets.QPushButton()
        self.loadImageButton.setIcon(QtGui.QIcon(pngpath_loadimage))  # Add your icon here
        self.loadImageButton.setIconSize(QtCore.QSize(25, 25))  # Adjust this to your desired icon size
        self.loadImageButton.setFixedSize(40, 40)  # Adjust the size of the button
        self.loadImageButton.setToolTip('Load Single TIFF')  # Set tooltip text
        self.loadImageButton.clicked.connect(self.load_image)
        toolbar.addWidget(self.loadImageButton)

    @QtCore.pyqtSlot()
    def load_image(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load TIFF", "", "TIFF Files (*.tif *.tiff);;All Files (*)")
        if file_path:
            image = fabio.open(file_path)
            self.image_data = image.data
            self.set_color_map(self.colorMapComboBox.currentText())

            # -- UPDATE WedgeROI position
            # Set the origin of the ROI to the center of the image
            self.data_origin = np.array(self.image_data.shape) // 2
            # Update the position of WedgeROI
            # self.update_wedgeROI_position()

            # Add this line to update the image displayed
            self.imageView.setImage(self.image_data)
            
    def setupRotateImageButton(self, toolbar):
        self.rotateImageButton = QtWidgets.QPushButton()
        self.rotateImageButton.setIcon(QtGui.QIcon(pngpath_rotate))  # replace with your actual .png file path
        self.rotateImageButton.setIconSize(QtCore.QSize(25, 25))  # Adjust this to your desired icon size
        self.rotateImageButton.setFixedSize(40, 40)
        self.rotateImageButton.setToolTip('Rotate Image')
        self.rotateImageButton.clicked.connect(self.rotate_image)
        toolbar.addWidget(self.rotateImageButton)

    @QtCore.pyqtSlot()
    def rotate_image(self):
        if self.image_data is not None:
            self.image_data = np.rot90(self.image_data)
            self.imageView.setImage(self.image_data)

            # Update the position of WedgeROI
            self.update_wedgeROI_position()

            # Update the radial integration plot after rotating the image
            self.update_radial_plot()

    def setupLoadTiffStack(self, toolbar):  # The method now accepts a QToolBar instead of a QVBoxLayout
        self.loadTiffStackButton = QtWidgets.QPushButton()
        self.loadTiffStackButton.setIcon(QtGui.QIcon(pngpath_imgstack))  # Add your icon here
        self.loadTiffStackButton.setIconSize(QtCore.QSize(25, 25))  # Adjust this to your desired icon size
        self.loadTiffStackButton.setFixedSize(40, 40)  # Adjust the size of the button
        self.loadTiffStackButton.setToolTip('Load TIFF Stack')  # Set tooltip text
        self.loadTiffStackButton.clicked.connect(self.load_tiff_stack)
        toolbar.addWidget(self.loadTiffStackButton)

    def load_tiff_stack(self):
        folder_path = self.gixs.select_folder_dialog()
        if folder_path:
            try:
                file_list = glob.glob(os.path.join(folder_path, self.gixs.wcard + '.tiff'))
                tiff_delayed = []
                for filepath in file_list:
                    tiff_delayed.append(delayed(self.gixs.load_tiff)(filepath))

                if tiff_delayed:
                    tiff_arrays = dask.compute(*tiff_delayed)
                    self.dask_tiff_stack = da.stack(tiff_arrays, axis=0)
                else:
                    print("No valid TIFF arrays found in the folder.")
            except Exception as e:
                print(f"Error in accessing TIFF Folder: {e}")

    # self.gixs.import_tiffs()  # Call the import_tiffs method from your GIXSDataReduction object
    # def load_tiff_stack(self):
    #     folder_path = self.gixs.select_folder_dialog()
    #     if folder_path:
    #         try:
    #             file_list = glob.glob(os.path.join(folder_path, self.gixs.wcard + '.tiff'))
    #             tiff_dask_arrays = []
    #             for filepath in file_list:
    #                 tiff_array = self.gixs.load_tiff(filepath)
    #                 if tiff_array is not None:  # Check if the array is valid
    #                     tiff_dask_arrays.append(da.from_array(tiff_array, chunks=(1000, 1000)))
                
    #             if tiff_dask_arrays:
    #                 self.dask_tiff_stack = da.stack(tiff_dask_arrays)
    #             else:
    #                 print("No valid TIFF arrays found in the folder.")
    #         except Exception as e:
    #             print(f"Error in accessing TIFF Folder: {e}")

        # folder_path = self.gixs.select_folder_dialog()  # Use the select_folder_dialog() method from GIXSDataReduction
        # if folder_path:
        #     try:
        #         self.gixs.import_tiffs(folder_path)  # Call the import_tiffs() method from GIXSDataReduction
        #     except Exception as e:
        #         print(f"Error in accessing TIFF Folder: {e}")
    
    def setupMASKPath(self, toolbar):
        self.maskPathButton = QtWidgets.QPushButton()
        self.maskPathButton.setIcon(QtGui.QIcon(pngpath_maskfile))  # Add your icon here
        self.maskPathButton.setIconSize(QtCore.QSize(25, 25))  # Adjust this to your desired icon size
        self.maskPathButton.setFixedSize(40, 40)  # Adjust the size of the button
        self.maskPathButton.setToolTip('Set Mask Path')  # Set tooltip text
        self.maskPathButton.clicked.connect(self.set_mask_path)
        toolbar.addWidget(self.maskPathButton)

    # set_mask_path method
    def set_mask_path(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open MASK", "", "MASK Files (*.edf);;All Files (*)")
        if file_path:
            try:
                self.gixs.open_mask_dialog(file_path)
            except Exception as e:
                print(f"Error in opening MASK file: {e}")
        # try:
        #     self.gixs.open_mask_dialog()  # Call the open_mask_dialog method from your GIXSDataReduction object
        # except Exception as e:
        #     # An error message box pops up if any exception occurs
        #     msg = QtWidgets.QMessageBox()
        #     msg.setIcon(QtWidgets.QMessageBox.Warning)
        #     msg.setText("An error occurred while setting the mask path.")
        #     msg.setInformativeText(str(e))
        #     msg.setWindowTitle("Error")
        #     msg.exec_()

    def setupColorMapComboBox(self, toolbar):  # The method now accepts a QToolBar instead of a QVBoxLayout
        self.colorMapComboBox = QtWidgets.QComboBox()
        self.colorMapComboBox.addItems(['inferno', 'cividis', 'viridis', 'magma', 'plasma'])
        self.colorMapComboBox.currentTextChanged.connect(self.set_color_map)
        self.colorMapComboBox.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.colorMapComboBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.colorMapComboBox.setToolTip('Color Map')  # Set tooltip text
        toolbar.addWidget(self.colorMapComboBox)

    @QtCore.pyqtSlot(str)
    def set_color_map(self, color_map):
        colormap = pg.colormap.get(color_map)
        lut = colormap.getLookupTable(256)
        self.imageView.ui.histogram.gradient.setColorMap(colormap)
        self.imageView.setImage(self.image_data)

    def setupOpenWindowButton(self, toolbar):
        open_window_btn = QPushButton(self)
        open_window_btn.setIcon(QtGui.QIcon(pngpath_corrections))  # replace with your actual .png file path
        open_window_btn.setIconSize(QtCore.QSize(25, 25))  # Adjust this to your desired icon size
        open_window_btn.setFixedSize(40, 40)  # Adjust the size of the button
        open_window_btn.setToolTip('Correction Settings')  # Set tooltip text
        open_window_btn.clicked.connect(self.open_window)
        toolbar.addWidget(open_window_btn)

    def open_window(self):
        self.input_dialog = InputDialog()
        self.input_dialog.show()

    def setupRadialPlot(self):
        self.radialPlot = pg.PlotWidget()
        # self.splitter.addWidget(self.radialPlot)

    def update_wedgeROI_position(self):
        # Check if the image_data and wedgeROI exist before setting position
        if self.image_data is not None and self.wedgeROI is not None:
            self.wedgeROI.setPos(self.data_origin)

    def update_radial_plot(self):
        r, theta = self.wedgeROI.map_to_polar()
        masked_data = self.image_data * self.wedgeROI.getArrayRegion(self.image_data, self.imageView.getImageItem())
        masked_data = np.ma.masked_less_equal(masked_data, 0)
        radial_integration = np.ma.mean(masked_data, axis=1)
        self.radialPlot.plot(r, radial_integration, clear=True)

    @QtCore.pyqtSlot()
    def toggle_radial_plot(self):
        if self.toggleRadialPlotButton.isChecked():
            # Create a new PlotWidget and display it in a new window
            self.radialPlot = pg.PlotWidget()
            self.radialPlot.setWindowTitle('Azimuthal Integration')
            self.radialPlot.show()

            # Update the radial integration plot
            self.update_radial_plot()
        else:
            # Hide the plot widget when the radio button is unchecked
            if self.radialPlot:
                self.radialPlot.hide()

class InputDialog(QWidget):
    def __init__(self):
        super(InputDialog, self).__init__()
        self.layout = QVBoxLayout()
        self.label = QLabel("This is a new window. Enter your variables here.")
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

# The origin of the WedgeROI should be initialized after the file is loaded, and on each instance of loading a new file.
class WedgeROI(pg.ROI):
    def __init__(self, pos, inner_radius, outer_radius, start_angle=0, span_angle=180):
        super().__init__(pos)

        # self.addScaleHandle([inner_radius, start_angle + span_angle / 2], [0, 0], itemClass=WedgeHandle)
        # self.addScaleHandle([outer_radius, start_angle + span_angle / 2], [0, 0], itemClass=WedgeHandle)
        # self.addRotateHandle([outer_radius, start_angle], [0, 0], itemClass=WedgeHandle)
        # self.addRotateHandle([outer_radius, start_angle + span_angle], [0, 0], itemClass=WedgeHandle)

        # Set the parameters of our wedge
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.start_angle = start_angle
        self.span_angle = span_angle

        # Define brush and pen
        self.pen = QtGui.QPen(QtCore.Qt.red)  # Change color as needed
        self.brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 50))  # Change color and transparency as needed

    def paint(self, p, opt, widget):
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(self.pen)
        p.setBrush(self.brush)
        p.drawPie(QtCore.QRectF(-self.outer_radius, -self.outer_radius, 
                                2*self.outer_radius, 2*self.outer_radius), 
                  self.start_angle * 16, self.span_angle * 16)
        p.drawPie(QtCore.QRectF(-self.inner_radius, -self.inner_radius, 
                                2*self.inner_radius, 2*self.inner_radius), 
                  self.start_angle * 16, self.span_angle * 16)

    def map_to_polar(self):
        bounds = self.boundingRect()
        y, x = np.ogrid[-bounds.top(): -bounds.bottom(), bounds.left(): bounds.right()]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return r, theta


def main():
    app = QtWidgets.QApplication(sys.argv)
    viewer = SimpleImageViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

# --------------------------------------------------------
'''
# class CustomHandle(pg.Handle):
#     def __init__(self, radius, angle, **kwargs):
#         super().__init__(**kwargs)
#         self.radius = radius
#         self.angle = angle

#     def getPos(self):
#         return [self.radius * np.cos(self.angle), self.radius * np.sin(self.angle)]

#     def setPos(self, pos):
#         self.radius = np.sqrt(pos[0]**2 + pos[1]**2)
#         self.angle = np.arctan2(pos[1], pos[0])

#     def movePoint(self, pos, modifiers):
#         if self.roi.rotateAllowed():
#             self.rotatePoint(pos, modifiers)
#         else:
#             roiPos = self.roi.mapToParent(self.roi.state['pos'])
#             hPos = self.roi.mapToParent(pos)
#             dp = hPos - roiPos
#             if dp.length() == 0:
#                 return

#             if self.translatable:
#                 angle = np.arctan2(dp.y(), dp.x())
#                 if np.abs(angle) < np.pi / 4 or np.abs(angle) > 3 * np.pi / 4:  # radial direction
#                     dp.setY(0)
#                 else:  # tangential direction
#                     dp.setX(0)

#                 newPos = roiPos + dp
#                 self.roi.checkRemoveHandle(self)
#                 self.roi.state['pos'] = self.roi.mapFromParent(newPos)

#             if self.rotatable:
#                 self.rotatePoint(pos, modifiers)
#             if self.scalable:
#                 self.scalePoint(pos, modifiers)

#         self.roi.stateChanged(finish=False)

# class WedgeROI(pg.ROI):
#     def __init__(self, pos, size, **args):
#         super().__init__(pos, size, **args)

#         # create four handles
#         self.addHandle({'item': CustomHandle(center=(0.5, 0.5)), 'pos': (0, 0.5), 'center': (0.5, 0.5)})
#         self.addHandle({'item': CustomHandle(center=(0.5, 0.5)), 'pos': (1, 0.5), 'center': (0.5, 0.5)})
#         self.addHandle({'item': CustomHandle(center=(0.5, 0.5)), 'pos': (0.5, 0), 'center': (0.5, 0.5)})
#         self.addHandle({'item': CustomHandle(center=(0.5, 0.5)), 'pos': (0.5, 1), 'center': (0.5, 0.5)})

#     def paint(self, p, *args):
#         p.setRenderHint(QtGui.QPainter.Antialiasing)
#         p.setPen(self.pen)
#         p.setBrush(self.currentBrush)
#         rect = self.boundingRect()
#         start_angle = self.startAngle()
#         span_angle = self.spanAngle()
#         p.drawPie(rect, start_angle, span_angle)

#     def boundingRect(self):
#         return QtCore.QRectF(0, 0, self.size()[0], self.size()[1])

#     def mapToItem(self, item, pos):
#         pos = self.mapToScene(pos)
#         return item.mapFromScene(pos)

#     def mapToPolar(self, pos):
#         x = pos.x() - self.size()[0] / 2
#         y = pos.y() - self.size()[1] / 2
#         r = np.sqrt(x ** 2 + y ** 2)
#         theta = np.arctan2(y, x)
#         return r, np.degrees(theta)
    
#     def startAngle(self):
#         # Define start angle here
#         return 0

#     def spanAngle(self):
#         # Define span angle here
#         return 180 * 16  # 180 degrees, Qt uses 1/16th of degree
'''

'''
    # @QtCore.pyqtSlot()
    # def load_image(self):
    #     file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load TIFF", "", "TIFF Files (*.tif *.tiff);;All Files (*)")
    #     if file_path:
    #         image = fabio.open(file_path)
    #         self.image_data = image.data

    #         if self.wedgeROI is None:
    #             self.wedgeROI = WedgeROI([0, 0], 10, 50)
    #             self.imageView.addItem(self.wedgeROI)

    #         self.set_color_map(self.colorMapComboBox.currentText())

        # @QtCore.pyqtSlot()
    # def toggle_radial_plot(self):
    #     # Show or hide the radialPlot depending on the state of the toggleRadialPlotButton
    #     self.radialPlot.setVisible(self.toggleRadialPlotButton.isChecked())

    # @QtCore.pyqtSlot()
    # def load_image(self):
    #     file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load TIFF", "", "TIFF Files (*.tif *.tiff);;All Files (*)")
    #     if file_path:
    #         image = fabio.open(file_path)
    #         self.image_data = image.data
    #         self.set_color_map(self.colorMapComboBox.currentText())

    #         # Set the origin of the ROI to the center of the image
    #         self.data_origin = np.array(self.image_data.shape) // 2

    #         # Update the position of WedgeROI instead of recreating it
    #         self.wedgeROI.setPos(self.data_origin)

    #         # Add this line to update the image displayed
    #         self.imageView.setImage(self.image_data)

    # @QtCore.pyqtSlot()
    # def rotate_image(self):
    #     self.image_data = np.rot90(self.image_data)
    #     self.imageView.setImage(self.image_data)
    #     # Update the radial integration plot after rotating the image
    #     self.update_radial_plot()


# Python Standard Libraries
import os
import sys

# External Libraries
import fabio
import numpy as np
import scipy as sp
import pyqtgraph as pg
from pyqtgraph import ROI, PlotWidget
from pyqtgraph import functions as fn
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QPushButton, QApplication, QFileDialog, QLabel, QWidget, QToolBar, QSizePolicy
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize

pngpath_rotate = '/Users/keithwhite/github_repositories/giwaxs_suite/png_widgeticon/rotationicon.png'
pngpath_corrections = '/Users/keithwhite/github_repositories/giwaxs_suite/png_widgeticon/loadimage.png'
pngpath_loadimage = '/Users/keithwhite/github_repositories/giwaxs_suite/png_widgeticon/correctionsicon.png'

# The origin of the WedgeROI should be initialized after the file is loaded, and on each instance of loading a new file.
class WedgeROI(pg.ROI):
    def __init__(self, pos, inner_radius, outer_radius, start_angle=0, span_angle=180):
        super().__init__(pos)

        # Set the parameters of our wedge
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.start_angle = start_angle
        self.span_angle = span_angle

    def paint(self, p, opt, widget):
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(self.pen)
        p.setBrush(self.brush)
        p.drawPie(QtCore.QRectF(-self.outer_radius, -self.outer_radius, 
                                2*self.outer_radius, 2*self.outer_radius), 
                  self.start_angle * 16, self.span_angle * 16)
        p.drawPie(QtCore.QRectF(-self.inner_radius, -self.inner_radius, 
                                2*self.inner_radius, 2*self.inner_radius), 
                  self.start_angle * 16, self.span_angle * 16)

    def map_to_polar(self):
        bounds = self.boundingRect()
        y, x = np.ogrid[-bounds.top(): -bounds.bottom(), bounds.left(): bounds.right()]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return r, theta

class InputDialog(QWidget):
    def __init__(self):
        super(InputDialog, self).__init__()
        self.layout = QVBoxLayout()
        self.label = QLabel("This is a new window. Enter your variables here.")
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

class SimpleImageViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Simple ImageViewer')
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QHBoxLayout(central_widget)

        # Create a splitter to manage ImageView and radialPlot sizes
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        layout.addWidget(self.splitter)

        self.setupImageView()  
        
        toolbar = self.addToolBar('ToolBar')
        self.setupRotateImageButton(toolbar)
        self.setupOpenWindowButton(toolbar)
        self.setupColorMapComboBox(toolbar)
        self.setupLoadImageButton(toolbar)

        self.setupRadialPlot()

        # Add a radio button to select/deselect the radialPlot
        self.toggleRadialPlotButton = QtWidgets.QRadioButton("Show Radial Integration", self)
        self.toggleRadialPlotButton.setChecked(True)
        self.toggleRadialPlotButton.toggled.connect(self.toggle_radial_plot)
        toolbar.addWidget(self.toggleRadialPlotButton)

        # Define the data origin
        self.data_origin = [0, 0]  # initialized to [0, 0]

        # Add WedgeROI
        self.wedgeROI = WedgeROI(self.data_origin, 10, 50)
        self.imageView.addItem(self.wedgeROI)

    def setupImageView(self):
        self.imageView = pg.ImageView()
        self.splitter.addWidget(self.imageView)

    def setupRotateImageButton(self, toolbar):
        self.rotateImageButton = QPushButton(QIcon(pngpath_rotate), '')
        self.rotateImageButton.setToolTip('Rotate Image')
        toolbar.addWidget(self.rotateImageButton)

    def setupOpenWindowButton(self, toolbar):
        self.openWindowButton = QPushButton(QIcon(pngpath_corrections), '')
        self.openWindowButton.setToolTip('Open a new Window')
        toolbar.addWidget(self.openWindowButton)

    def setupColorMapComboBox(self, toolbar):
        self.colorMapComboBox = QtWidgets.QComboBox(self)
        for colorMapName in ['grey', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']:
            self.colorMapComboBox.addItem(colorMapName)
        self.colorMapComboBox.setCurrentText('inferno')
        self.colorMapComboBox.setToolTip('Select Color Map')
        toolbar.addWidget(self.colorMapComboBox)

    def setupLoadImageButton(self, toolbar):
        self.loadImageButton = QPushButton(QIcon(pngpath_loadimage), '')
        self.loadImageButton.setToolTip('Load Image')
        toolbar.addWidget(self.loadImageButton)
        self.loadImageButton.clicked.connect(self.load_image)

    def setupRadialPlot(self):
        self.radialPlot = PlotWidget()
        self.splitter.addWidget(self.radialPlot)

    @QtCore.pyqtSlot()
    def load_image(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load TIFF", "", "TIFF Files (*.tif *.tiff);;All Files (*)")
        if file_path:
            image = fabio.open(file_path)
            self.image_data = image.data
            self.set_color_map(self.colorMapComboBox.currentText())

            # Set the origin of the ROI to the center of the image
            self.data_origin = np.array(self.image_data.shape) // 2

            # Update the position of WedgeROI instead of recreating it
            self.wedgeROI.setPos(self.data_origin)

            # Add this line to update the image displayed
            self.imageView.setImage(self.image_data)

    def set_color_map(self, color_map_name):
        if color_map_name == 'grey':
            color_map = pg.ColorMap(pos=np.linspace(0.0, 1.0, 2), color=[(0, 0, 0), (255, 255, 255)])
        elif color_map_name == 'viridis':
            color_map = pg.ColorMap(pos=np.linspace(0.0, 1.0, 256), color=pg.colormap.get('viridis').bytes())
        elif color_map_name == 'plasma':
            color_map = pg.ColorMap(pos=np.linspace(0.0, 1.0, 256), color=pg.colormap.get('plasma').bytes())
        elif color_map_name == 'inferno':
            color_map = pg.ColorMap(pos=np.linspace(0.0, 1.0, 256), color=pg.colormap.get('inferno').bytes())
        elif color_map_name == 'magma':
            color_map = pg.ColorMap(pos=np.linspace(0.0, 1.0, 256), color=pg.colormap.get('magma').bytes())
        elif color_map_name == 'cividis':
            color_map = pg.ColorMap(pos=np.linspace(0.0, 1.0, 256), color=pg.colormap.get('cividis').bytes())

        self.imageView.setColorMap(color_map)

    def toggle_radial_plot(self):
        if self.toggleRadialPlotButton.isChecked():
            self.radialPlot.show()
        else:
            self.radialPlot.hide()

def main():
    app = QtWidgets.QApplication(sys.argv)
    viewer = SimpleImageViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()


# # Python Standard Libraries
# import os
# import sys

# # External Libraries
# import fabio
# import numpy as np
# import scipy as sp
# import pyqtgraph as pg
# from pyqtgraph import ROI, PlotWidget
# from pyqtgraph import functions as fn
# from PyQt5 import QtWidgets, QtCore, QtGui
# from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QPushButton, QApplication, QFileDialog, QLabel, QWidget, QToolBar, QSizePolicy
# from PyQt5.QtGui import QIcon
# from PyQt5.QtCore import QSize

# # The origin of the WedgeROI should be initialized after the file is loaded, and on each instance of loading a new file.
# class WedgeROI(pg.ROI):
#     def __init__(self, pos, inner_radius, outer_radius, start_angle=0, span_angle=180):
#         super().__init__(pos)

#         # Set the parameters of our wedge
#         self.inner_radius = inner_radius
#         self.outer_radius = outer_radius
#         self.start_angle = start_angle
#         self.span_angle = span_angle

#     def paint(self, p, opt, widget):
#         p.setRenderHint(QtGui.QPainter.Antialiasing)
#         p.setPen(self.pen)
#         p.setBrush(self.brush)
#         p.drawPie(QtCore.QRectF(-self.outer_radius, -self.outer_radius, 
#                                 2*self.outer_radius, 2*self.outer_radius), 
#                   self.start_angle * 16, self.span_angle * 16)
#         p.drawPie(QtCore.QRectF(-self.inner_radius, -self.inner_radius, 
#                                 2*self.inner_radius, 2*self.inner_radius), 
#                   self.start_angle * 16, self.span_angle * 16)

#     def map_to_polar(self):
#         bounds = self.boundingRect()
#         y, x = np.ogrid[-bounds.top(): -bounds.bottom(), bounds.left(): bounds.right()]
#         r = np.sqrt(x**2 + y**2)
#         theta = np.arctan2(y, x)
#         return r, theta

# class InputDialog(QWidget):
#     def __init__(self):
#         super(InputDialog, self).__init__()
#         self.layout = QVBoxLayout()
#         self.label = QLabel("This is a new window. Enter your variables here.")
#         self.layout.addWidget(self.label)
#         self.setLayout(self.layout)

# class SimpleImageViewer(QtWidgets.QMainWindow):
#     def __init__(self):
#         super().__init__()

#         self.setWindowTitle('Simple ImageViewer')
#         self.setGeometry(100, 100, 1200, 800)

#         central_widget = QtWidgets.QWidget()
#         self.setCentralWidget(central_widget)
#         layout = QtWidgets.QHBoxLayout(central_widget)

#         # Create a splitter to manage ImageView and radialPlot sizes
#         self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
#         layout.addWidget(self.splitter)

#         self.setupImageView()  # now 'self.splitter' exists when this is called
#         self.setupSecondImageView(layout)

#         toolbar = self.addToolBar('ToolBar')
#         self.setupRotateImageButton(toolbar)
#         self.setupOpenWindowButton(toolbar)
#         self.setupColorMapComboBox(toolbar)
#         self.setupLoadImageButton(toolbar)

#         self.setupRadialPlot()

#         # Add a radio button to select/deselect the radialPlot
#         self.toggleRadialPlotButton = QtWidgets.QRadioButton("Show Radial Integration", self)
#         self.toggleRadialPlotButton.setChecked(True)
#         self.toggleRadialPlotButton.toggled.connect(self.toggle_radial_plot)
#         toolbar.addWidget(self.toggleRadialPlotButton)

#         # Define the data origin
#         self.data_origin = [0, 0]  # initialized to [0, 0]

#         # Add WedgeROI
#         self.wedgeROI = WedgeROI(self.data_origin, 10, 50)
#         self.imageView.addItem(self.wedgeROI)

#         # Add plot for radial integration
#         self.radialPlot = pg.PlotWidget()
#         layout.addWidget(self.radialPlot)
#         self.wedgeROI.sigRegionChangeFinished.connect(self.update_radial_plot)


#     def setupImageView(self, layout):
#         self.imageView = pg.ImageView()
#         self.splitter.addWidget(self.imageView)

#     def setupSecondImageView(self, layout):  # This is a new method for setting up the second ImageView
#         self.imageView2 = pg.ImageView()
#         layout.addWidget(self.imageView2)

#     def setupColorMapComboBox(self, toolbar):  # The method now accepts a QToolBar instead of a QVBoxLayout
#         self.colorMapComboBox = QtWidgets.QComboBox()
#         self.colorMapComboBox.addItems(['inferno', 'cividis', 'viridis', 'magma', 'plasma'])
#         self.colorMapComboBox.currentTextChanged.connect(self.set_color_map)
#         self.colorMapComboBox.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
#         self.colorMapComboBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
#         self.colorMapComboBox.setToolTip('Color Map')  # Set tooltip text
#         toolbar.addWidget(self.colorMapComboBox)

#     def setupLoadImageButton(self, toolbar):  # The method now accepts a QToolBar instead of a QVBoxLayout
#         self.loadImageButton = QtWidgets.QPushButton()
#         self.loadImageButton.setIcon(QtGui.QIcon('/Users/keithwhite/github_repositories/giwaxs_suite/class_mvc-architecture/loadimage.png'))  # Add your icon here
#         self.loadImageButton.setIconSize(QtCore.QSize(25, 25))  # Adjust this to your desired icon size
#         self.loadImageButton.setFixedSize(40, 40)  # Adjust the size of the button
#         self.loadImageButton.setToolTip('Load Image')  # Set tooltip text
#         self.loadImageButton.clicked.connect(self.load_image)
#         toolbar.addWidget(self.loadImageButton)

#     def setupRotateImageButton(self, toolbar):
#         self.rotateImageButton = QtWidgets.QPushButton()
#         self.rotateImageButton.setIcon(QtGui.QIcon('/Users/keithwhite/github_repositories/giwaxs_suite/class_mvc-architecture/rotationicon.png'))  # replace with your actual .png file path
#         self.rotateImageButton.setIconSize(QtCore.QSize(25, 25))  # Adjust this to your desired icon size
#         self.rotateImageButton.setFixedSize(40, 40)
#         self.rotateImageButton.setToolTip('Rotate Image')
#         self.rotateImageButton.clicked.connect(self.rotate_image)
#         toolbar.addWidget(self.rotateImageButton)

#     def setupOpenWindowButton(self, toolbar):
#         open_window_btn = QPushButton(self)
#         open_window_btn.setIcon(QtGui.QIcon('/Users/keithwhite/github_repositories/giwaxs_suite/class_mvc-architecture/correctionsicon.png'))  # replace with your actual .png file path
#         open_window_btn.setIconSize(QtCore.QSize(25, 25))  # Adjust this to your desired icon size
#         open_window_btn.setFixedSize(40, 40)  # Adjust the size of the button
#         open_window_btn.setToolTip('Correction Settings')  # Set tooltip text
#         open_window_btn.clicked.connect(self.open_window)
#         toolbar.addWidget(open_window_btn)

#     def setupImageView(self):
#         self.imageView = pg.ImageView()
#         self.splitter.addWidget(self.imageView)

#     def setupRadialPlot(self):
#         self.radialPlot = pg.PlotWidget()
#         self.splitter.addWidget(self.radialPlot)

#     def open_window(self):
#         self.input_dialog = InputDialog()
#         self.input_dialog.show()

#     @QtCore.pyqtSlot(str)
#     def set_color_map(self, color_map):
#         colormap = pg.colormap.get(color_map)
#         lut = colormap.getLookupTable(256)
#         self.imageView.ui.histogram.gradient.setColorMap(colormap)
#         self.imageView.setImage(self.image_data)

#     @QtCore.pyqtSlot()
#     def load_image(self):
#         file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load TIFF", "", "TIFF Files (*.tif *.tiff);;All Files (*)")
#         if file_path:
#             image = fabio.open(file_path)
#             self.image_data = image.data
#             self.set_color_map(self.colorMapComboBox.currentText())

#             # Set the origin of the ROI to the center of the image
#             self.data_origin = np.array(self.image_data.shape) // 2

#             # Update the position of WedgeROI instead of recreating it
#             self.wedgeROI.setPos(self.data_origin)

#     def update_radial_plot(self):
#         r, theta = self.wedgeROI.map_to_polar()
#         masked_data = self.image_data * self.wedgeROI.getArrayRegion(self.image_data, self.imageView.getImageItem())
#         masked_data = np.ma.masked_less_equal(masked_data, 0)
#         radial_integration = np.ma.mean(masked_data, axis=1)
#         self.radialPlot.plot(r, radial_integration, clear=True)
    
#     @QtCore.pyqtSlot()
#     def rotate_image(self):
#         self.image_data = np.rot90(self.image_data)
#         self.imageView.setImage(self.image_data)

#     @QtCore.pyqtSlot()
#     def toggle_radial_plot(self):
#         # Show or hide the radialPlot depending on the state of the toggleRadialPlotButton
#         self.radialPlot.setVisible(self.toggleRadialPlotButton.isChecked())

#     @QtCore.pyqtSlot()
#     def rotate_image(self):
#         self.image_data = np.rot90(self.image_data)
#         self.imageView.setImage(self.image_data)
#         # Update the radial integration plot after rotating the image
#         self.update_radial_plot()

# def main():
#     app = QtWidgets.QApplication(sys.argv)
#     viewer = SimpleImageViewer()
#     viewer.show()
#     sys.exit(app.exec_())


# if __name__ == '__main__':
#     main()


# # Python Standard Libraries
# import os
# import sys

# # External Libraries
# import fabio
# import numpy as np
# import pyqtgraph as pg
# from pyqtgraph import ROI, PlotWidget
# from pyqtgraph import functions as fn
# from PyQt5 import QtWidgets, QtCore, QtGui
# from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QPushButton, QApplication, QFileDialog, QLabel, QWidget, QToolBar, QSizePolicy
# from PyQt5.QtGui import QIcon
# from PyQt5.QtCore import QSize

# class WedgeROI(pg.ROI):
#     def __init__(self, pos, size):
#         super().__init__(pos, size, movable=True, rotatable=True)

#         # Set some custom parameters for the shape of our wedge.
#         self.start_angle = 0
#         self.span_angle = 120

#         # Create rotate handles to control the start and end angles of the wedge.
#         self.addRotateHandle([1, 0.5], [0.5, 0.5])  # Start angle handle
#         self.addRotateHandle([0, 0.5], [0.5, 0.5])  # End angle handle

#         # Create scale handles to control the radial span of the wedge.
#         self.addScaleHandle([0.5, 0], [0.5, 0.5])  # Outer radius handle
#         self.addScaleHandle([0.5, 1], [0.5, 0.5])  # Inner radius handle

#     def paint(self, p, opt, widget):
#         rect = self.boundingRect()
#         p.setRenderHint(QtGui.QPainter.Antialiasing)
#         p.setPen(self.pen)
#         p.setBrush(self.brush)
#         p.drawPie(rect, self.start_angle * 16, self.span_angle * 16)

#     def map_to_polar(self):
#         # Get the bounds of the wedge
#         bounds = self.boundingRect()

#         # Compute the meshgrid for the wedge
#         y, x = np.ogrid[-bounds.top(): -bounds.bottom(), bounds.left(): bounds.right()]

#         # Convert cartesian coordinates to polar coordinates
#         r = np.sqrt(x**2 + y**2)
#         theta = np.arctan2(y, x)
        
#         return r, theta
    
# # This widget is used to input variables in a new window
# class InputDialog(QWidget):
#     def __init__(self):
#         super(InputDialog, self).__init__()

#         self.layout = QVBoxLayout()
#         self.label = QLabel("This is a new window. Enter your variables here.")
#         self.layout.addWidget(self.label)
#         self.setLayout(self.layout)


# # This widget is used to display image, and control the settings
# class SimpleImageViewer(QtWidgets.QMainWindow):
#     def __init__(self):
#         super().__init__()

#         self.setWindowTitle('Simple ImageViewer')
#         self.setGeometry(100, 100, 1200, 800)

#         central_widget = QtWidgets.QWidget()
#         self.setCentralWidget(central_widget)
#         layout = QtWidgets.QHBoxLayout(central_widget)  # Changed from QVBoxLayout to QHBoxLayout

#         self.setupImageView(layout)
#         self.setupSecondImageView(layout)  # Added this line to setup the second ImageView
#         self.setupImageView()

#         toolbar = self.addToolBar('ToolBar')
#         self.setupRotateImageButton(toolbar)
#         self.setupOpenWindowButton(toolbar)
#         self.setupColorMapComboBox(toolbar)  # Moved the color map combo box to the toolbar
#         self.setupLoadImageButton(toolbar)  # Moved the load image button to the toolbar

#         # Create a splitter to manage ImageView and radialPlot sizes
#         self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
#         layout.addWidget(self.splitter)

#         self.setupImageView()
#         self.setupRadialPlot()
#         # self.setupToolbar()
        
#         # Add a radio button to select/deselect the radialPlot
#         self.toggleRadialPlotButton = QtWidgets.QRadioButton("Show Radial Integration", self)
#         self.toggleRadialPlotButton.setChecked(True)
#         self.toggleRadialPlotButton.toggled.connect(self.toggle_radial_plot)
#         toolbar.addWidget(self.toggleRadialPlotButton)

#         # Add WedgeROI
#         self.wedgeROI = WedgeROI([0, 0], [1, 1])
#         self.imageView.addItem(self.wedgeROI)

#         # Add plot for radial integration
#         self.radialPlot = pg.PlotWidget()
#         layout.addWidget(self.radialPlot)
#         self.wedgeROI.sigRegionChangeFinished.connect(self.update_radial_plot)

#     def setupImageView(self, layout):
#         self.imageView = pg.ImageView()
#         layout.addWidget(self.imageView)

#     def setupSecondImageView(self, layout):  # This is a new method for setting up the second ImageView
#         self.imageView2 = pg.ImageView()
#         layout.addWidget(self.imageView2)

#     def setupColorMapComboBox(self, toolbar):  # The method now accepts a QToolBar instead of a QVBoxLayout
#         self.colorMapComboBox = QtWidgets.QComboBox()
#         self.colorMapComboBox.addItems(['inferno', 'cividis', 'viridis', 'magma', 'plasma'])
#         self.colorMapComboBox.currentTextChanged.connect(self.set_color_map)
#         self.colorMapComboBox.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
#         self.colorMapComboBox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
#         self.colorMapComboBox.setToolTip('Color Map')  # Set tooltip text
#         toolbar.addWidget(self.colorMapComboBox)

#     def setupLoadImageButton(self, toolbar):  # The method now accepts a QToolBar instead of a QVBoxLayout
#         self.loadImageButton = QtWidgets.QPushButton()
#         self.loadImageButton.setIcon(QtGui.QIcon('/Users/keithwhite/github_repositories/giwaxs_suite/class_mvc-architecture/loadimage.png'))  # Add your icon here
#         self.loadImageButton.setIconSize(QtCore.QSize(25, 25))  # Adjust this to your desired icon size
#         self.loadImageButton.setFixedSize(40, 40)  # Adjust the size of the button
#         self.loadImageButton.setToolTip('Load Image')  # Set tooltip text
#         self.loadImageButton.clicked.connect(self.load_image)
#         toolbar.addWidget(self.loadImageButton)

#     def setupRotateImageButton(self, toolbar):
#         self.rotateImageButton = QtWidgets.QPushButton()
#         self.rotateImageButton.setIcon(QtGui.QIcon('/Users/keithwhite/github_repositories/giwaxs_suite/class_mvc-architecture/rotationicon.png'))  # replace with your actual .png file path
#         self.rotateImageButton.setIconSize(QtCore.QSize(25, 25))  # Adjust this to your desired icon size
#         self.rotateImageButton.setFixedSize(40, 40)
#         self.rotateImageButton.setToolTip('Rotate Image')
#         self.rotateImageButton.clicked.connect(self.rotate_image)
#         toolbar.addWidget(self.rotateImageButton)

#     def setupOpenWindowButton(self, toolbar):
#         open_window_btn = QPushButton(self)
#         open_window_btn.setIcon(QtGui.QIcon('/Users/keithwhite/github_repositories/giwaxs_suite/class_mvc-architecture/correctionsicon.png'))  # replace with your actual .png file path
#         open_window_btn.setIconSize(QtCore.QSize(25, 25))  # Adjust this to your desired icon size
#         open_window_btn.setFixedSize(40, 40)  # Adjust the size of the button
#         open_window_btn.setToolTip('Correction Settings')  # Set tooltip text
#         open_window_btn.clicked.connect(self.open_window)
#         toolbar.addWidget(open_window_btn)

#     def setupImageView(self):
#         self.imageView = pg.ImageView()
#         self.splitter.addWidget(self.imageView)

#     def setupRadialPlot(self):
#         self.radialPlot = pg.PlotWidget()
#         self.splitter.addWidget(self.radialPlot)

#     def open_window(self):
#         self.input_dialog = InputDialog()
#         self.input_dialog.show()

#     @QtCore.pyqtSlot(str)
#     def set_color_map(self, color_map):
#         colormap = pg.colormap.get(color_map)
#         lut = colormap.getLookupTable(256)
#         self.imageView.ui.histogram.gradient.setColorMap(colormap)
#         self.imageView.setImage(self.image_data)

#     @QtCore.pyqtSlot()
#     def load_image(self):
#         file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load TIFF", "", "TIFF Files (*.tif *.tiff);;All Files (*)")
#         if file_path:
#             image = fabio.open(file_path)
#             self.image_data = image.data
#             self.set_color_map(self.colorMapComboBox.currentText())

#     @QtCore.pyqtSlot()
#     def rotate_image(self):
#         self.image_data = np.rot90(self.image_data)
#         self.imageView.setImage(self.image_data)

#     def update_radial_plot(self):
#         r, theta = self.wedgeROI.map_to_polar()
#         masked_data = self.image_data * self.wedgeROI.getArrayRegion(self.image_data, self.imageView.getImageItem())
#         masked_data = np.ma.masked_less_equal(masked_data, 0)
#         radial_integration = np.ma.mean(masked_data, axis=1)
#         self.radialPlot.plot(r, radial_integration, clear=True)

#     @QtCore.pyqtSlot()
#     def toggle_radial_plot(self):
#         # Show or hide the radialPlot depending on the state of the toggleRadialPlotButton
#         self.radialPlot.setVisible(self.toggleRadialPlotButton.isChecked())

#     @QtCore.pyqtSlot()
#     def rotate_image(self):
#         self.image_data = np.rot90(self.image_data)
#         self.imageView.setImage(self.image_data)
#         # Update the radial integration plot after rotating the image
#         self.update_radial_plot()
    
# def main():
#     app = QtWidgets.QApplication(sys.argv)
#     viewer = SimpleImageViewer()
#     viewer.show()
#     sys.exit(app.exec_())


# if __name__ == '__main__':
#     main()


class WedgeHandle(pg.Handle):
    def __init__(self, radius, angle, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius
        self.angle = angle

    def getPos(self):
        return [self.radius * np.cos(self.angle), self.radius * np.sin(self.angle)]

    def setPos(self, pos):
        self.radius = np.sqrt(pos[0]**2 + pos[1]**2)
        self.angle = np.arctan2(pos[1], pos[0])

class CustomHandle(RotateHandle, ScaleHandle):
    def movePoint(self, pos, modifiers):
        if self.roi.rotateAllowed():
            self.rotatePoint(pos, modifiers)
        else:
            roiPos = self.roi.mapToParent(self.roi.state['pos'])
            hPos = self.roi.mapToParent(pos)
            dp = hPos - roiPos
            if dp.length() == 0:
                return

            if self.translatable:
                angle = np.arctan2(dp.y(), dp.x())
                if np.abs(angle) < np.pi / 4 or np.abs(angle) > 3 * np.pi / 4:  # radial direction
                    dp.setY(0)
                else:  # tangential direction
                    dp.setX(0)

                newPos = roiPos + dp
                self.roi.checkRemoveHandle(self)
                self.roi.state['pos'] = self.roi.mapFromParent(newPos)

            if self.rotatable:
                self.rotatePoint(pos, modifiers)
            if self.scalable:
                self.scalePoint(pos, modifiers)

        self.roi.stateChanged(finish=False)

class WedgeROI(ROI):
    def __init__(self, pos, size, **args):
        ROI.__init__(self, pos, size, **args)

        # create four handles
        self.handles = [{'item': Handle(center=(0.5, 0.5)), 'pos': (0, 0.5), 'center': (0.5, 0.5)},
                        {'item': Handle(center=(0.5, 0.5)), 'pos': (1, 0.5), 'center': (0.5, 0.5)},
                        {'item': Handle(center=(0.5, 0.5)), 'pos': (0.5, 0), 'center': (0.5, 0.5)},
                        {'item': Handle(center=(0.5, 0.5)), 'pos': (0.5, 1), 'center': (0.5, 0.5)}]

        self.addScaleHandle((0, 0.5), (1, 0.5))
        self.addScaleHandle((1, 0.5), (0, 0.5))
        self.addScaleHandle((0.5, 0), (0.5, 1))
        self.addScaleHandle((0.5, 1), (0.5, 0))

    def paint(self, p, *args):
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(self.pen)
        p.setBrush(self.currentBrush)
        p.drawEllipse(self.boundingRect())

        rect = self.boundingRect()
        start_angle = self.startAngle()
        span_angle = self.spanAngle()

        p.drawPie(rect, start_angle, span_angle)

    def boundingRect(self):
        return QtCore.QRectF(0, 0, self.size()[0], self.size()[1])

    def mapToItem(self, item, pos):
        pos = self.mapToScene(pos)
        return item.mapFromScene(pos)

    def mapToPolar(self, pos):
        x = pos.x() - self.size()[0] / 2
        y = pos.y() - self.size()[1] / 2
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        return r, np.degrees(theta)
    
    def startAngle(self):
        # Define start angle here
        return 0

    def spanAngle(self):
        # Define span angle here
        return 180 * 16  # 180 degrees, Qt uses 1/16th of degree
'''