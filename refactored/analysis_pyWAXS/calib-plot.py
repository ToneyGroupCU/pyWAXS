import sys
import os
import numpy as np
from scipy.signal import find_peaks
from collections import defaultdict
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QCheckBox, QPushButton, QLabel, QFileDialog, QMessageBox, QLineEdit, QFormLayout, QDialog
import pyqtgraph as pg


class PeakSelector(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Peak Selector')
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Input fields for wavelength and threshold
        self.input_form = QFormLayout()
        self.layout.addLayout(self.input_form)

        self.wavelength_input = QLineEdit("0.976")
        self.input_form.addRow("Wavelength:", self.wavelength_input)

        self.threshold_input = QLineEdit("0.002")
        self.input_form.addRow("Threshold:", self.threshold_input)
        
        self.tolerance_input = QLineEdit("0.05")
        self.input_form.addRow("Tolerance:", self.tolerance_input)

        # Load data button
        self.load_button = QPushButton("Load .xy or .int file")
        self.layout.addWidget(self.load_button)
        self.load_button.clicked.connect(self.load_file)

        # Add a PlotWidget instance
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)

    def load_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog(self)
        file_dialog.setOptions(options)
        file_dialog.setNameFilters(["Two Theta files (*.xy *.int)", "All Files (*)"])
        if file_dialog.exec_():
            filename = file_dialog.selectedFiles()[0]
        else:
            return

        try:
            # Load data from the file
            if filename.endswith('.xy'):
                data = np.genfromtxt(filename, skip_header=2)
            elif filename.endswith('.int'):
                data = np.genfromtxt(filename, skip_header=3)

            two_theta_values = data[:, 0]
            intensity_values = data[:, 1]

            # Convert two_theta to q
            wavelength = float(self.wavelength_input.text())
            self.q_values = 4 * np.pi * np.sin(np.deg2rad(two_theta_values / 2)) / wavelength

            # Normalize intensity values
            self.intensity_normalized = intensity_values / np.max(intensity_values)

            # Filter out data where q is less than 0.1
            mask = self.q_values >= 0.1
            self.q_values = self.q_values[mask]
            self.intensity_normalized = self.intensity_normalized[mask]

            # Find peaks
            threshold = float(self.threshold_input.text())
            peak_indices, _ = find_peaks(self.intensity_normalized, height=threshold)
            self.peak_q_values = self.q_values[peak_indices]
            self.peak_intensities = self.intensity_normalized[peak_indices]

            # Find sets with approximately constant difference
            self.find_diff_sets()

            # Set plot properties
            self.plot_widget.setLabel('left', 'Intensity (arb. units)')
            self.plot_widget.setLabel('bottom', 'q (Å^{-1})')
            self.plot_widget.setTitle('Normalized Intensity')
            self.plot_widget.showGrid(x=True, y=True)

        except Exception as e:
            error_dialog = QMessageBox()
            error_dialog.setWindowTitle("Error")
            error_dialog.setText(str(e))
            error_dialog.exec_()
            return

    def find_diff_sets(self):
        diffs = np.diff(self.peak_q_values)
        diff_sets = defaultdict(list)
        for i, diff in enumerate(diffs):
            for set_diff, set_peaks in list(diff_sets.items()):
                if abs(diff - set_diff) / set_diff < float(self.tolerance_input.text()):  # within user-defined tolerance
                    diff_sets[set_diff].append((self.peak_q_values[i + 1], self.peak_intensities[i + 1]))
                    break
            else:
                diff_sets[diff].append((self.peak_q_values[i + 1], self.peak_intensities[i + 1]))

        self.show_sets(diff_sets)

    def show_sets(self, diff_sets):
        set_dialog = QDialog(self)
        set_dialog.setWindowTitle("Select Peak Sets")
        layout = QVBoxLayout()
        set_dialog.setLayout(layout)

        checkboxes = []
        for set_diff, set_peaks in sorted(diff_sets.items(), reverse=True):  # Sorting in descending order
            if len(set_peaks) > 1:  # Include only those sets which contain more than one element
                checkbox = QCheckBox(f"Set with approximate difference {set_diff:.2f}")
                layout.addWidget(checkbox)
                checkboxes.append((checkbox, set_peaks))

        plot_button = QPushButton("Plot")
        layout.addWidget(plot_button)

        plot_button.clicked.connect(lambda: self.plot(checkboxes, set_dialog))

        set_dialog.exec_()

    def plot(self, checkboxes, dialog):
        self.plot_widget.clear()

        # Plot the data
        self.plot_widget.plot(self.q_values, self.intensity_normalized, pen='b', name='Data')

        # Plot the selected peaks
        for checkbox, set_peaks in checkboxes:
            if checkbox.isChecked():
                for idx, (q_value, intensity) in enumerate(sorted(set_peaks, key=lambda x: x[0])):
                    self.plot_widget.plot([q_value], [intensity], pen='r', symbol='o', symbolBrush='r')
                    text_item = pg.TextItem(text=f"{idx+1} ({q_value:.2f})", color='r')
                    self.plot_widget.addItem(text_item)
                    text_item.setPos(q_value, intensity)

        # Close the dialog
        dialog.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PeakSelector()
    window.show()
    sys.exit(app.exec_())
