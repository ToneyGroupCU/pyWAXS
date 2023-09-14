# pyWAXS
Our group's Python-based GIWAXS data reduction and analysis package.

Toney Group, University of Colorado Boulder.

Developers: Keith White, Zihan Zhang, Andrew Levin

Updated | 09/13/2023

## Getting Setup
Navigate in your terminal to where you want to save the repository:
```bash
    cd /path/to/githubrepositories
```

Clone the repository to your local directory:
```bash
    git clone https://github.com/ToneyGroupCU/pyWAXS.git
```

Navigate to the newly cloned repo:
```bash
    cd pyWAXS
```

Create a conda virtual environment from the pyWAXS.yml file:
```bash
    conda env create -f pyWAXS.yml
```

This will install all conda-forge and pip dependencies. 

Once the environment is solved, open the repository in VS Code (recommended) or your preferred IDE.

## Getting Started
At present, the files you will be working with are stored in the 'main' folder. The 'refactored' folder contains code segments that are actively being recycled, repurposed, or discarded.

Open the file:

```python
    WAXSNotebook.ipynb
```

Now you should read through the notebook as a guide. You'll want to update the file paths to reflect the accurate basePath directory. This will be explained in the notebook. 

## UI Applications
IPython is a console installed in the pyWAXS environment that allows you to run .py scripts as applications from the command line.

The .py scripts listed below can be run as executables:
#### WAXSVisualizer.py: 
(Previously EasyGIWAXS.py) Allows you to load your GIWAXS data to perform ROI (region-of-interest) based integrations.

To run this script, create a new tab in the terminal (CTRL + T) navigate and the 'main' pyWAXS directory:
```bash
    cd /path/to/githubrepositories/pyWAXS/main
```
Activate the pyWAXS environment:
```bash
    conda activate pyWAXS
```
Now you can run the .py script with IPython:
```bash
    python3 WAXSVisualizer.py
```
#### WAXSPeakSelect.py: 
Used to Add/Remove peaks that the automated peak finding algorithm did not find, or found accidentally.

Allows you to load NETCDF4 (HDF5) files contained 2D reduced image data and peak position data. These files are exported from the WAXSNotebook.ipynb (see internal notebook documentation and notes). 

Similarly for WAXSPeakSelect.py, we can run the script by navigating to the file directory in our terminal and running:
```bash
    python3 WAXSPeakSelect.py
```

<!-- # Developer Notes on File Structure/Organization -->
<!-- -- MAIN FOLDER --
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
Description: Scripts for diffraction image simulation, essentially Zihan's 2D_diffraction repository on his personal Github page. -->

<!-- Folder Name: pyWAXS_nslsiijupyterhub
Description: Scripts adapted from the PyHyperScattering notebooks/code that are also implemented in the pyHyperScattering CMS branch repo on our group page. -->