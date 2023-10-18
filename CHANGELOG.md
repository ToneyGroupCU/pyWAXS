## [0.1.0] - 2023-10-18
### Added
- Initial release with basic data reduction functionalities.

Requirements: Installation of current pyWAXS.yml version as virtual conda environment for running with your notebook kernel.

setup.py: Used to call pyWAXS classes from your Jupyter notebook with a pip install statement.

(Jupyter Notebook) Class Descriptions:
    - WAXSTransform: Based on pyFAI and pygix. Class used in WAXSReduce to create a detector object using 'pygix'. Used to apply detector image corrections to TIFF raw datasets loaded into WAXSReduce instance.
    - WAXSReduce: Single datafile loading and reduction within the class. Includes 2D image correction and 1D integration methods.
    - WAXSAFF: Atomic form factors used for simulating (GI)WAXS images.
    - WAXSFileManager: Used to create file trees for storing project information.
    - WAXSSearch: Houses the algorithms used to search for peaks in a 2D GIWAXS imagge.
    
(Jupyter Notebook) Skeleton (Partial) Class Descriptions:
    - MoleculeConstructor: Used to build molecular CIFs from bond angle and bond distance tables. Capable of constructing crystal CIFs based on space group, Wyckoff sites and atom types.
    - WAXSAnalyze: Skeleton class, will incorporate methods to compare simulation data with real datasets.
    - WAXSDiffSim: Framework for simulating (GI)WAXS patterns from POSCAR (VASP) files, converted from CIFs.
    - WAXSComputeCrystal: Compute the Bragg peak positions of an input crystal system/set of lattice constants. Output (h k l) values for these assignments. Useful for mapping to real datasets.
    - WAXSReverse: Reverse GIWAXS solver, used to brute force determine lattice constants.
    - WAXSExperiment: Placeholder, will house multiple class instances to compare multiple simulations, multi-image process datasets, and output large datasets.

(UI) Class Descriptions:
    - pyWAXS: Main UI, can be executed from bash using iPython commands (e.g., 'python3 pyWAXS.py). Used for project loading and peak identification.
    - pyWAXSSim: Subwindow call of the main pyWAXS UI. Will be used to call WAXSDiffSim methods in UI for simulating GIWAXS datasets.
    - WAXSVisualer: Simple visualizer to view 2D image data and perform ROI boxcut integrations.