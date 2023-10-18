# -- PyQt5 Imports -- #
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QGridLayout, QWidget, QFileDialog, QGroupBox, QVBoxLayout, QSlider, QLabel, QAction, QDialog, QFormLayout, QLineEdit, QComboBox, QMessageBox, QTextEdit, QTableWidget, QTableWidgetItem, QRadioButton, QToolBar, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QFont

# -- Matplotlib Imports -- #
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.text import Text
from matplotlib.figure import Figure

# -- Custom Imports -- #
# from pyWAXS import MyCanvas
# from WAXSDiffSim import WAXSDiffSim
# import WAXSAFF

# Standard imports
import numpy as np
import os
from pathlib import Path
from typing import Union, List

class SimWindow(QMainWindow):
    def __init__(self):
        super(SimWindow, self).__init__()
        self.initUI()

        # - Relevant Data Paths
        self.poscar_path = None
        self.project_path = None
        self.project_name = None 

        self.ds_sim = None # simulation dataset

    def initUI(self):
        # Canvas and table for displaying simulated intensity and peak values
        self.canvas = MyCanvas()

        # Table Widget Layout
        self.tableWidget = QTableWidget()
        self.tableWidget.setColumnCount(9)
        self.tableWidget.setHorizontalHeaderLabels(['h','k','l','qᵣ (Å⁻¹)', '𝛘 (°)', 'qxy (Å⁻¹)', 'qz (Å⁻¹)', 'd-spacing (Å)', 'Intensity'])

        # Toolbar Widget Layout
        self.toolbar = SimNavigationToolbar(self.canvas, self)

        # Load POSCAR Button
        btn_load_poscar = QPushButton("Load POSCAR")
        btn_load_poscar.clicked.connect(self.load_poscar_file)

        # Input Field Group Layout
        sigma1 = QLineEdit()
        sigma2 = QLineEdit()
        sigma3 = QLineEdit()
        hkl_dim = QLineEdit()
        thetax = QLineEdit()
        thetay = QLineEdit()

        vlayout = QVBoxLayout()
        field_group = QGroupBox("Simulation Parameters")
        vlayout.addWidget(QLabel("Sigma 1:"))
        vlayout.addWidget(sigma1)
        vlayout.addWidget(QLabel("Sigma 2:"))
        vlayout.addWidget(sigma2)
        vlayout.addWidget(QLabel("Sigma 3:"))
        vlayout.addWidget(sigma3)
        vlayout.addWidget(QLabel("(hkl) Dimension:"))
        vlayout.addWidget(hkl_dim)
        vlayout.addWidget(QLabel("Theta X:"))
        vlayout.addWidget(thetax)
        vlayout.addWidget(QLabel("Theta Y:"))
        vlayout.addWidget(thetay)
        field_group.setLayout(vlayout)

        # LAYOUT FORMATTING ---- Layout Positions: addWidget(QWidget, row, column, rowSpan, columnSpan)
        # Set the size policy for the field_group to be fixed
        field_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Set the size policy for canvas and table to be expanding
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tableWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create a grid layout and add widgets
        layout = QGridLayout()
        layout.addWidget(self.toolbar, 0, 0, 1, 4)
        layout.addWidget(field_group, 1, 0, 1, 1)
        layout.addWidget(self.canvas, 1, 1, 1, 3)  # Span 3 columns
        layout.addWidget(self.tableWidget, 2, 1, 1, 3)  # Span 3 columns
        layout.addWidget(btn_load_poscar, 2, 0, 1, 1)

        # Adjust the column and row stretch factors
        layout.setColumnStretch(0, 1)  # Smaller stretch factor for field_group
        layout.setColumnStretch(1, 4)  # Larger stretch factor for canvas and table

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.setWindowTitle('pyWAXS: GIWAXS Simulation')
        self.show()

    def load_poscar_file(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "Load POSCAR", "", "VASP Files (*.vasp);;All Files (*)", options=options)
        if file:
            self.poscar_path = file
            # f = open(address)
            self.file_data = open(self.poscar_path)
            ind = 0
            for x in self.file_data:
                ind += 1
                if ind == 3:
                    x=x.lstrip()
                    x=x.rstrip()
                    x=x.split()
                    x=np.array(x)
                    p=x.astype(float)
                    a1=p
                if ind == 4:
                    x=x.lstrip()
                    x=x.rstrip()
                    x=x.split()
                    x=np.array(x)
                    p=x.astype(float)
                    a2=p
                if ind == 5:
                    x=x.lstrip()
                    x=x.rstrip()
                    x=x.split()
                    x=np.array(x)
                    p=x.astype(float)
                    a3=p
                if ind == 6:
                    x=x.lstrip()
                    x=x.rstrip()
                    x=x.split()
                    iii=0
                    pp=np.zeros(len(x))
                    for p in x:
                        pp[iii]=WAXSAFF.atom_dict[p]
                        iii+=1
                    x=np.array(pp)
                    z=x.astype(int)
                if ind == 7:
                    x=x.lstrip()
                    x=x.rstrip()
                    x=x.split()
                    x=np.array(x)
                    z1=x.astype(int)
                    temp=np.sum(z1)
                    position=np.zeros((temp,4))
                if ind > 8:
                    x=x.lstrip()
                    x=x.rstrip()
                    x=x.split("         ")
                    x=np.array(x)
                    p=x.astype(float)
                    position[ind-9,1]=p[0]
                    position[ind-9,2]=p[1]
                    position[ind-9,3]=p[2]
            self.file_data.close()
            ind = 0
            iii = 0

            for ii in z1:
                position[iii:iii+ii+1,0]=z[ind]
                iii=iii+ii
                ind=ind+1
            
            # Storing values in the DataFrame
            self.data.loc[len(self.data)] = [self.poscar_path, a1, a2, a3, position, None, None]
            # self.diffsim_df.loc[len(self.diffsim_df)] = [address, a1, a2, a3, position, None, None]

            # return a1,a2,a3,position

class SimCanvas(FigureCanvas):
    def __init__(self):
        # super().__init__(figure)
        # ============ Figure Initialization ============
        self.fig = Figure()
        super(MyCanvas, self).__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

class SimNavigationToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent=None, coordinates=True):
        # super().__init__(canvas, parent, coordinates)
        super(SimNavigationToolbar, self).__init__(canvas, parent, coordinates)
        self.window = parent  # Assuming the parent window is passed as `parent`

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = SimWindow()
    window.show()
    sys.exit(app.exec_())
