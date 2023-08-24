# pyWAXS
Python-based GIWAXS image analysis software package from the Toney Group, University of Colorado Boulder.

UPDATED: 08/24/2023
# Developer Notes on File Structure/Organization
-- MAIN FOLDER --
Folder Name: pyWAXS_main
Description: Main class files - .py scripts stored here should be a composite of scripts/notebooks found in all other folders. These will be the primary working classes. Please do not push to this unless you are confident in the updates you are adding.

-- INFORMATION & DATA FOLDERS --
Folder Name: notes
Description: Notes pertaining to algorithm development and implementation in the main pyWAXS classes.

Folder Name: examples
Description: Example data files from Keith's GIWAXS experiments at 11-BM (CMS), will also add an example notebook to this with templates at some point.

Folder Name: cif_repo (needs to be added)
Description: CIF image repository, also contains exported POSCAR files. Let's try to upkeep the folder organization scheme here.

-- OTHER FOLDERS: All other folders contain scripts and code segments that are being actively pulled from to construct the main working classes in 'pyWAXS_main'.
Folder Name: pyWAXS_analysis
Description: Scripts pertaining to GIWAXS image analysis, such as peak indexing, peak searching, single atom-basis Bragg peak calculations, etc.

Folder Name: pyWAXS_pyQt5app
Description: Scripts used to construct a working GUI application using the pyQt5 module.

Folder Name: pyWAXS_reduction
Description: Scripts for reducing GIWAXS data, there is some overlap here with the NSLS-II JupyterHub data reduction scripts.

Folder Name: pyWAXS_simulation
Description: Scripts for diffraction image simulation, essentially Zihan's 2D_diffraction repository on his personal Github page.

<!-- Folder Name: pyWAXS_nslsiijupyterhub
Description: Scripts adapted from the PyHyperScattering notebooks/code that are also implemented in the pyHyperScattering CMS branch repo on our group page. -->