#!keithwhite@Keiths-MacBook-Pro/opt/anaconda3/envs/pyWAXS
# --------
# keith white
# date: 05/17/2023
# script: waxssim_main.py
# --
# purpose: simulate giwaxs data from inputted poscar file. import real data to compare
# to simulated data. select regions of interest in the real data to compare to the 
# simulated data. set a number of initial guesses for to the crystallite 
# orientation parameters. run a chi-square minimization routine to fit the 
# simulated data to the real data.
# --------

# -- functions -- #
# -- simulated data
# load poscar file
# read-in poscar file
# input parameters
    # crystallite orientation parameters
    # extent of (h k l) 
    # image resolution
    # scherrer grain-size analysis
# generate intensity map

# -- real data
# read-in .tiff file image
# create a detector corrections object for .tiff file image corrections
# apply detector corrections to the .tiff file image
# select regions of interest on the .tiff file image (2 - 4)
# find peaks on the tiff image file

# -- comparator functions
# create a project folder that includes CIFs, data, PONI, and corrected image .csv files
# check found peaks against simulated peak positions - group sets with same delta between
# reflectons
# check (qxy, qz) positions in poscar file against selected peaks in real data set
# modify lattice constants (a, b, c, alpha, beta, gamma) to match simulation to the real data
# set initial conditions for least squares minimization
# modify all gaussian parameters over specified step size in sigma, creating
# a 3D parameter space of textured images. 
# Use a peak finding algorithm to find the local minima of each mapped parameter space.
# Compare all local minima to find the global minima
# Output the lattice constants and correction parameters
# Output a list of the found peaks in the data, and whether they match the CIF or not.