##################################################################################################################################
# CUSTOM CONDA ENVIRONMENT: 'pyWAXS' - need to generate .yml file for env reproduction.
# Jupyter Notebook Kernel: keithwhite@Keiths-MacBook-Pro/opt/anaconda3/envs/pyWAXS
# ----------------------------------------------------------------------------------------- #
# Contributors: Keith White, Andrew Levin, Thomas Chaney, Zihan Zhang
# Toney Group, University of Colorado Boulder
# Updated: 04/10/2023
# Version Number: NSLS-II, Version 1.4
# Description: Python script function library for processing 2D GIWAXS images, both batch processing and single image processing.
##################################################################################################################################

# -- IMPORT LIBRARIES ------------------------------------------------ #
# import nslsii_11bmcms_pkglibimport as lib
# -------- PyFAI -------- #
import pyFAI
import pyFAI.gui
import pyFAI.detectors
import pyFAI.calibrant
# -------- PyGIX -------- #
import pygix
# import pygix.plotting as gixsplt
# -------- Standard Libraries -------- #
import math, fabio, silx, os, re, time, csv, io, pylatex, lmfit, psutil, cv2, sys, gc
import numpy as np
import pandas as pd
import glob2 as glob
from IPython.display import clear_output
from PIL import Image
from pathlib import Path
from lmfit import Model
# from zipfile import ZipFile
# --------- SciPy ----------- #
import scipy as sp
from scipy import signal
from scipy.optimize import curve_fit
from scipy.special import wofz
from scipy.signal import fftconvolve
from scipy.signal import argrelextrema
from scipy.signal import find_peaks, peak_prominences, peak_widths
# -------- Matplotlib -------- #
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import matplotlib as mpl
from matplotlib import cm
import matplotlib.tri as tri
from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.pyplot import figure
# matplotlib.use('agg') # setting the backend
# from matplotlib.gridspec import GridSpec
# import matplotlib.gridspec as gridspec
# from mpl_toolkits.mplot3d import  Axes3D
# from matplotlib.collections import PolyCollection
# from ipywidgets import AppLayout, FloatSlider
# from matplotlib.table import Table
# from matplotlib.backends.backend_agg import FigureCanvasAgg
# -------- Custom Imports (.py) Files -------- #
# import AFF
# import timer as stopwatch

print("Using pyFAI version", pyFAI.version)

# --- Create pygix detector transform object --- #
pg = pygix.Transform()
# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #
# ----- Directory Management Functions ------ #
# create a name for .png file with manual identifier
def create_imagename(datapath, savepath, timestamp, totaltime, identifier = '_2D_', extension = '_.png'):
    """Create a unique image name from the data filepath input. Identifier is optional, automatically appends '_2D_' before the image extension. Extension defaults to '.png'."""
    # CREATE A UNIQUE FILENAME/PATH FOR THE 2D IMAGE
    # img_timestamp = time.strftime("%Y%m%d_%H%M%S") # Create a time string, with Year, Month, Day + '_' + Hour, Minute, Second. This is appended to the save_path to give the analyzed/reduced data a unique identifier.
    img_path = os.path.splitext(datapath)[0] + identifier + timestamp + '_' + str(totaltime) + extension # Technique for creating a save path, strip the current path and append. But we want a new folder...
    img_name = os.path.basename(img_path) # Strip the filename from the data filepath
    img_path = os.path.join(savepath, img_name)
    return img_path, img_name

# create a name for a csv file
def create_csvname(datapath, savepath, timestamp, totaltime, identifier = '_2dcake_', extension = '.csv'):
    """Create a unique image name from the data filepath input. Identifier is optional, automatically appends '_2D_' before the image extension. Extension defaults to '.csv'."""
    # CREATE A UNIQUE FILENAME/PATH FOR THE 2D IMAGE
    # csv_timestamp = time.strftime("%Y%m%d_%H%M%S") # Create a time string, with Year, Month, Day + '_' + Hour, Minute, Second. This is appended to the save_path to give the analyzed/reduced data a unique identifier.
    csv_path = os.path.splitext(datapath)[0] + '_' + str(totaltime) + identifier + timestamp + extension
    csv_name = os.path.basename(csv_path)
    csv_path = os.path.join(savepath, csv_name)
    # dataframe2D.to_csv(csv_path2D, index=False)
    return csv_path

# create a csv file
def create_csvfile(data_frame, csv_path):
    data_frame.to_csv(csv_path, index=False)
    # Should create a header of information: PONI File, Incident Angle, Sample Orientation, Background File
    return

# Creates a unique timestamped folder in the specified path
def createfolder(samplename, datafolder):
    """ Creates a unique folder from the folder prefix 'samplename' (not necessarily the samplename) in the specified 'data_folder' path (parent containing folder). Returns the new save_path, as well as the timestamp."""
    timestamp = time.strftime("%Y%m%d_%H%M%S") # Create time stamp to make a unique folder identifier.
    savefolder = samplename + "_" + timestamp # Create a save folder from the sample name and the folder time stamp.
    savepath = os.path.join(datafolder, savefolder) # Merge the new time stamped folder name with the existing data_folder path
    os.mkdir(savepath) # Make the new directory for the batch process save folder.
    return savepath, timestamp

# Uses glob to strip select files from path based on input parameter 'globStr', or glob string.
def grabpaths(data_folder, globStr):
    """ Grab all of the files from the input data folder with specification from the searchString. (Default) searchString = '*_[0-9][0-9][0-9][0-9]_maxs.tiff' """
    file_paths = [] # list that will store the file paths used for analysis
    file_paths = sorted(glob.iglob(data_folder + globStr)) # Uses 'glob.iglob' to match the phrase and grab data file paths that are indexed with the specified identifier.
    return file_paths

# ----- Data Mining Functions ----- #
# mines metadata from the filename in the datapath and zips to dict with keylist
def metadataminer(samplename, datapath, keylist, basetime, index):
    file_dict = {}

    filename = os.path.splitext(datapath)[0] + '.tiff' # strip file name from the path
    basename = os.path.basename(filename) # create basepath name
    minedvallist = basename.split("_")
    
    file_dict = dict(zip(keylist, minedvallist))

    file_dict['samplename'] = samplename
    file_dict['path'] = datapath # Create a dictionary item for the data filepath
    file_dict['basename'] = basename # store the basename

    clocktime = file_dict.get('clocktime', '0')
    clocktime = re.findall("\d+\.\d+", clocktime)
    clocktime = clocktime[0] if clocktime else '0'
    file_dict['clocktime'] = clocktime

    xpos = file_dict.get('xpos', '0')
    xpos = re.findall("\d+\.\d+", xpos)
    xpos = xpos[0] if xpos else '0'
    file_dict['xpos'] = xpos

    thpos = file_dict.get('thpos', '0')
    thpos = re.findall("\d+\.\d+", thpos)
    thpos = thpos[0] if thpos else '0'
    file_dict['thpos'] = thpos

    exptime = file_dict.get('exptime', '0')
    exptime = re.findall("\d+\.\d+", exptime)
    exptime = str(np.round(float(exptime[0]), 1)) if exptime else '0'  # round exposure time to nearest decimal place
    file_dict['exptime'] = exptime

    if index == 0:
        basetime = float(clocktime)
        
    totaltime = (float(clocktime) - basetime) + (float(exptime) * index)
    file_dict['totaltime'] = str(totaltime)

    # create an entry name for this dictionary in the larger collection of dictionaries
    scanid = file_dict.get('scanid', '')
    dict_entryname = f"{samplename}_{scanid}_{str(totaltime)}"
    file_dict['dict_entryname'] = str(dict_entryname)

    return file_dict, dict_entryname

# dynamic version of create_imagename - generates image name using mined metadata
def makeimgname(datapath, savepath, timestamp, datafile_dict, index, identifier = '_2drecipmap_', ext = '.png'):
    """Create a unique image name from the data filepath input. Identifier is optional, automatically appends '_2D_' before the image extension. Extension defaults to '.png'."""
    # CREATE A UNIQUE FILENAME/PATH FOR THE 2D IMAGE
    totaltime = datafile_dict.get('totaltime')
    imgpath = os.path.splitext(datapath)[0] + '_' + str(timestamp) + identifier + str(totaltime) + '_' + str(index) + ext # Technique for creating a save path, strip the current path and append. But we want a new folder...
    imgname = os.path.basename(imgpath) # Strip the filename from the data filepath
    imgpath = os.path.join(savepath, imgname)
    return imgpath, imgname

# ----- Data Reduction Functions -----#
# cakes the corrected 2drecipspacemap and generates a pole fig with linear bkg interpolation
def cakedchipolefig(data, mask, chilims, qlims, bkgoffset = 0.02):
    '''Function Description: chi_integrate() takes input parameters from the user, and caked image output arrays from pygix. The function will generate a pole figure within the specified bounds, including a background subtraction at 0.02 
    inverse Angstroms from the bounds of integration. Additionally, this function will exclude rows/columns in the caked image that have a detector gap, or void region from the Ewald sphere correction.'''
    '''Function Variables: data_caked, data output from the Pygix caking function, mapped to reciprocal space.; qr, qr array output from the pygix caking function; chi, chi array output from the Pygix caking function;
    upper_chi = upper integration bound for plotting the pole figure; lower_chi = lower integration bound for plotting the pole figure; upper_q = upper q-value integration bound for the intensity integration;
    lower_q = lower q-value integration bound for the intensity integration'''
    
    data_caked, qr, chi = pg.transform_image(data, process='polar', # Convert detector image to caked qspace (chi vs q plot)
                                            method = 'bbox',
                                            unit='q_A^-1',
                                            mask=mask)
    
    chimin = chilims[0] # lower integration bound in chi
    chimax = chilims[1] # upper integration bound in chi
    qmin = qlims[0] # lower display bound in q
    qmax = qlims[1] # upper display bound in q

    lowerqbkg = qmin - bkgoffset # linear bkg interpolation min
    upperqbkg = qmax + bkgoffset # linear bkg interpolation max
    
    qdata = [] # empty array for q
    upperbkg = [] # empty array for upper bkg
    lowerbkg = [] # empty array for lower bkg
    
    int_array = np.zeros(np.shape(data_caked))
    data_upperbkg = np.zeros(np.shape(chi))
    data_lowerbkg = np.zeros(np.shape(qr))
    
    flag = 0 # flag var for detector gaps covering peak

    for row in range(np.shape(data_caked)[0]): # process each row on the input caked image - if within chi range, scan columns.
        if chimin <= chi[row] <= chimax:
            for column in range(np.shape(data_caked)[1]):
                if lowerqbkg <= qr[column] <= qmin: # this grabs values in "lower and upper bkg regions" for peak background
                    lowerbkg.append(data_caked[row,column])
                if qmax <= qr[column] <= upperqbkg:
                    upperbkg.append(data_caked[row,column])
                        
            lowerbkg = np.asarray(lowerbkg)
            data_lowerbkg[row] = np.median(lowerbkg) # median ensures an outlier pixel doesnt impact bkg subtraction.
            upperbkg = np.asarray(upperbkg)
            data_upperbkg[row] = np.median(upperbkg) # median ensures an outlier pixel doesnt impact bkg subtraction.
            lowerbkg = []
            upperbkg = []
   
    for row in range(np.shape(data_caked)[0]): # process each row and column to check if pixel is within bounds.
        for column in range(np.shape(data_caked)[1]):
            if qmin <= qr[column] <= qmax and chimin <= chi[row] <= chimax:
                # Here is where linear interpolation of peak background occurs...
                bkgslope = ((data_upperbkg[row] - data_lowerbkg[row])/(qmax - qmin))
                bkgval = data_lowerbkg[row]+((qr[column]-qmin)*bkgslope)
                # Background corrected intensity is saved...
                if data_caked[row,column] == 0:
                    flag = 1
                int_array[row,column] = data_caked[row,column] - bkgval
            bkgval = 0
        if flag == 1:
            int_array[row,:] = int_array[row,:]*0
        flag = 0
    
    data_cakepole = np.sum(int_array, axis=1) # intensities summed across q to give intensity vs chi plots
    # int_data[int_data <= 0] = np.nan # int_data[int_data <= 0] = 0

    chi1d = data_cakepole[0] # TAKE 1D INTEGRATION RESULTS AND SEPERATE FOR PLOTTING q-values to be plotted
    poleintensity1d = data_cakepole[1] # INTENSITY VALUES TO BE PLOTTED
    return chi1d, poleintensity1d

# cakes the corrected 2drecipspacemap and generates and integrates over the rebinned chi plot
def cakedchiintegration(caked_data, chi, chilims, qr, qlims):
    # data_caked, qr, chi = pg.transform_image(data, process='polar', # Convert detector image to caked qspace (chi vs q plot)
    #                                         method = 'bbox',
    #                                         unit='q_A^-1',
    #                                         mask=mask)
    
    chimin, chimax = chilims
    qmin, qmax = qlims

    int_array = np.zeros(np.shape(caked_data))

    for row in range(np.shape(caked_data)[0]): # recursively scan each row and column to see if the pixel is within bounds
        for column in range(np.shape(caked_data)[1]):
            if qmin <= qr[column] <= qmax and chimin <= chi[row] <= chimax:
                int_array[row,column] = caked_data[row,column] # add pixels within bounds to new int_array
            if caked_data[row, column] < 1:
                int_array[row, column] = np.nan

    int_data = np.nanmean(int_array, axis=0) #sum int_array across all chi giving q vs intensity
    # print (np.shape(int_data))

    qr = qr[int_data!=0] # chuck out qr vals that are zero
    int_data = int_data[int_data!=0] # clear rows along chi with partially missing data
    data_cakedchiint = np.stack([qr, int_data], axis=1) # stack the qr integrations

    data_cakedchiint = np.transpose(data_cakedchiint) # transpose the array
    qrchi = data_cakedchiint[0] # qvals for plotting
    intensity = data_cakedchiint[1] # INTENSITY VALUES TO BE PLOTTED

    return qrchi, intensity

# build or rebuild detector object
def buildDetObj(pg, datafile_dict, corrections, poni_file, mask_file):
    # Create the detector objects for 2D image transformations and 1D azimuthal integrations.
    # pg = pygix.Transform() # create detector transform object, 'pg' for pygix detector 
    
    pg.load(poni_file) # load the poni file into the transform object
    sample_orientation = corrections[2]

    pg.sample_orientation = sample_orientation # if error with this, set to 3 for GI geometry
    mask = fabio.open(mask_file).data # load mask file

    incident_angle = datafile_dict.get('thpos') # grab incidence angle from metadata dictionary, apply to detector object
    pg.incident_angle = float(incident_angle) # apply incidence angle for ewald sphere correction

    return pg, mask

# transform input raw data
def gixsdataTransform (data, corrections, pg, mask, dark, flat, case='caked'):
    # unpack the list of correction variables
    chicorr, qsqrcorr, sample_orientation, rot1, rot2, rot3, correctSolidAngle, maskarray, polarization_factor, dark, flat, ffilt = corrections
    maskarray = mask
    """ corrections (list) position-var index
        # [0] chicorr : boolean
            # sin(chi) correction term to rescale preferred scatterers - only applied to rebinned caked images
        # [1] qsqrcorr : boolean
            # q-squared correction, often accompanies sin(chi)
        # [2] sample_orientation : integer
            # passes sample orientation for pygix detector object
        # [3] rot1 : float
            # passes detector rotation (rot1) applied to detector object
        # [4] rot2 : float
            # passes detector rotation (rot2) applied to detector object
        # [5] rot3 : float
            # passes detector rotation (rot3) applied to detector object
        # [6] correctSolidAngle : boolean
            # applies solid angle correction to pixels based on angle subtended
            # by scattering vectors onto detector intercept plane w/ respect to PONI positions
        # [7] mask : ndarray
            # applies mask to take care of hot/dead/edge/unusable pixels
        # [8] polarization_factor : float
            # from -1 to 1 based on ellipsoidal polarization and handedness
        # [9] dark : ndarray
            # dark image correction file
        # [10] flat : ndarray
            # flat field image correction file for interpixel sensitivity
        # [11] ffilt : boolean
            # apply fourier filter to 1d integrated intensities
    """

    if case == None: # return the input data as the output
        output_data = data
    
    if case == 'recip': # return the corrected reciprocal space map
        # recip space transform
        recip_data, qxy, qz = pg.transform_reciprocal(data, # Convert detector image to q-space (sample reciprocal)
                                            method = 'bbox', #for some reason splitpix doesnt work?
                                            unit='A', # unit='A'
                                            mask=maskarray, correctSolidAngle = correctSolidAngle, polarization_factor=polarization_factor, dark=dark, flat=flat)
        
        output_data = recip_data
        arrA = qxy
        arrB = qz

    """ pygix reciprocal_transform parameters
        ----------
        data : ndarray
            2D array from detector (raw image).
        filename : str
            Output filename in 2/3 column ascii format.
        correctSolidAngle : bool
            Correct for solid angle of each pixel if True.
        variance : ndarray
            Array containing the variance of the data. If not available, 
            no error propagation is done.
        error_model : str
            When variance is unknown, an error model can be given: 
            "poisson" (variance = I), "azimuthal" (variance = 
            (I-<I>)^2).
        x_range : (float, float), optional
            The lower and upper unit of the in-plane unit. If not 
            provided, range is simply (data.min(), data.max()). Values 
            outside the range are ignored.
        y_range : (float, float), optional
            The lower and upper range of the out-of-plane unit. If not 
            provided, range is simply (data.min(), data.max()). Values 
            outside the range are ignored.
        mask : ndarray
            Masked pixel array (same size as image) with 1 for masked 
            pixels and 0 for valid pixels.
        dummy : float
            Value for dead/masked pixels.
        delta_dummy : float
            Precision for dummy value
        polarization_factor : float
            Polarization factor between -1 and +1. 0 for no correction.
        dark : ndarray
            Dark current image.
        flat : ndarray
            Flat field image.
        method : str
            Integration method. Can be "np", "cython", "bbox",
            "splitpix", "lut" or "lut_ocl" (if you want to go on GPU).
        unit : str
            Grazing-incidence units. Can be "2th_af_deg", "2th_af_rad", 
            "qxy_qz_nm^-1", "qxy_qz_A^-1", "qxy_qz_nm^-1" or 
            "qxy_qz_A^-1". For GISAXS qy vs qz is typically preferred;
            for GIWAXS, qxy vs qz.
            (TTH_AF_DEG, TTH_AF_RAD, QY_QZ_NM, QY_QZ_A, QXY_QZ_NM, 
            QXY_QZ_A).
        safe : bool
            Do some extra check to ensure LUT is still valid. False is
            faster.
        normalization_factor : float
            Value of a normalization monitor."""
    
    # gixsplt.implot(recip_data, qxy, qz, xlim=(-5, 28), ylim=(-.5, 32), mode='rsm')

    if case == 'caked': # return the caked data
        # caking transformation
        caked_data, qr, chi = pg.transform_image(data, process='polar', # Convert detector image to caked qspace (chi vs q plot)
                                                method = 'bbox',
                                                unit='q_A^-1',
                                                mask=maskarray, correctSolidAngle = correctSolidAngle, polarization_factor=polarization_factor, dark=dark, flat=flat)
    
        output_data = caked_data
        arrA = qr
        arrB = chi

    return output_data, arrA, arrB

# ----- Data Analysis Functions ----- #
# index the peaks based on input peak params
def peakindextool(qrchi, intensity, peakparams):
    qrchi_peaks = [] # null array for q-values of indexed peaks
    intensity_peaks = [] # null array for intensity values of indexed peaks
    numindex_peaks = [] # null array for numeric array index value of indexed peaks

    # ADD FUNCTIONALITY HERE
        # Can the input params be made to have q-based dependence?
        # Addition of a FT Filter?
    
    prominence, width, distance = peakparams
    numindex_peaks = sp.signal.find_peaks(intensity, prominence=prominence, width=width, distance=distance)[0] # Locate all of the peaks in the 1D pattern, based on prominence and relative peak width. 

    for j in range(0, len(numindex_peaks)): # Create a paired array of the corresponding q-values.
        qrchi_peaks.append(qrchi[numindex_peaks[j]])
        intensity_peaks.append(intensity[numindex_peaks[j]])
    
    np_numindex_peaks = np.array(numindex_peaks) # Create numpy array of corresponding peak q-values
    np_intensity_peaks = np.array(intensity_peaks) # Create numpy array of identied peak intensities
    np_qrchi_peaks = np.array(qrchi_peaks) # Peak intensity values

    df_peakid = pd.DataFrame({"qinvA" : np_qrchi_peaks, "intensity" : np_intensity_peaks, "azimuth1d_index" : np_numindex_peaks}) # CREATE DATAFRAME FOR CSV OUTPUT
    
    return np_qrchi_peaks, np_intensity_peaks, np_numindex_peaks, df_peakid

# apply peak labels to the input data that has been indexed, save the labelled image.
def applypeaklabels(qrchi1d, intensity1d, datafile_dict, datapath, azimuth1d_path, timestamp, chilims, np_qrchi_peaks, np_intensity_peaks, np_numindex_peaks, df_peakid):
    fig = figure(figsize = (20,12))
    
    index = 0
    identifier = '_chimin' + str(chilims[0]) + '_chimax' + str(chilims[1]) + '_peakid_'
    imgpath, imgname = makeimgname(datapath, azimuth1d_path, timestamp, datafile_dict, index, identifier = identifier, ext = '.png')

    totaltime = datafile_dict.get('totaltime')
    composition = datafile_dict.get('composition')
    sampnum = datafile_dict.get('sampnum')
    solutionnum = datafile_dict.get('solutionnum')

    imgheader = composition + ' - ' + solutionnum + ' - ' + sampnum + '_chimin' + str(chilims[0]) + '_chimax' + str(chilims[1]) + ' | Time: ' + totaltime
    plt.title(imgheader, fontsize = 30) # title
    plt.plot(qrchi1d, intensity1d)
    plt.plot(np_qrchi_peaks,np_intensity_peaks[np_numindex_peaks], "ob");
    
    for xy_label in zip(np_qrchi_peaks,np_intensity_peaks[np_numindex_peaks]): # Coordinate (X,Y) labelling on the output plot.
        plt.annotate('(%.3f, %.3f)' %xy_label, xy=xy_label, textcoords = 'data')

    plt.tick_params(axis='both', which='major', labelsize=16) 
    plt.xlabel('q ($\AA^{-1}$)', fontsize=60)
    plt.xticks(fontsize=40)
    plt.xlim(np.min(qrchi1d), np.max(qrchi1d))

    plt.ylabel('Intensity (arb. units)',fontsize=60)
    plt.yticks(fontsize=40)
    plt.ylim(0, round(np.max(intensity1d)*1.05)) # multiplier to add height to the figure

    plt.tight_layout()
    plt.savefig(imgpath, dpi=200)
    plt.close()
   
    return

# ----- Image Formatting Functions ----- #
# format the 2d reciprocal space map and save the image.
def format2drecip(data_recip, qxy, qz, datafile_dict, datapath, recipmap2d_path, timestamp):
    # matplotlib.use('agg') # swap the backend to help with memory leaks

    fig = figure(figsize = (28,14))

    contrastmin = np.nanpercentile(np.min(data_recip),0) #COLOR SATURATION LIMIT
    contrastmax = np.nanpercentile(np.max(data_recip),100) #COLOR SATURATION LIMIT
    extent=(np.min(qxy),np.max(qxy),np.min(qz),np.max(qz))

    recipmap = plt.imshow(data_recip, cmap='turbo', extent=extent, origin = "lower", aspect='auto')

    index = 0
    imgpath, imgname = makeimgname(datapath, recipmap2d_path, timestamp, datafile_dict, index, identifier = '_2drecipmap_', ext = '.png')

    totaltime = datafile_dict.get('totaltime')
    composition = datafile_dict.get('composition')
    sampnum = datafile_dict.get('sampnum')
    solutionnum = datafile_dict.get('solutionnum')

    # imgheader = composition + ' - ' + solutionnum + ' - ' + sampnum + ' | Time: ' + totaltime
    if solutionnum is not None and totaltime is not None:
        imgheader = f"{solutionnum} | Time: {totaltime}"
    else:
        imgheader = ""  # Provide a default value if either solutionnum or totaltime is None

    plt.title(imgheader, fontsize = 30) # title
    plt.xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)', fontsize=58) # x label
    plt.xticks(fontsize=40)
    plt.xlim(np.min(qxy), np.max(qxy))

    plt.ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)',fontsize=58) # y label
    plt.yticks(fontsize=40)
    plt.ylim(0, np.max(qz))

    recipmap.set_clim(contrastmin, contrastmax)
    # cb = plt.colorbar(recipmap).set_label(label='',size=36)
    plt.tight_layout()
    plt.savefig(imgpath, dpi=500); # save image
    
    plt.close("all")
    plt.close()
    plt.close(fig)

    # delete loop iterated vars
    del fig
    plt.pause(.01)
    gc.collect()

    return

# format the 2d caked image and save the image.
def format2dcake(caked_data, qr, chi, datafile_dict, datapath, recipmap2d_path, timestamp):
    # matplotlib.use('agg') # swap the backend to help with memory leaks

    fig = figure(figsize = (28,14))

    contrastmin = np.nanpercentile(np.min(caked_data),0) #COLOR SATURATION LIMIT
    contrastmax = np.nanpercentile(np.max(caked_data),100) #COLOR SATURATION LIMIT
    extent=(np.min(qr),np.max(qr),np.min(chi),np.max(chi))

    cake = plt.imshow(caked_data,
        cmap='turbo',
        extent=extent,
        origin = 'lower',
        aspect='auto',
        interpolation='spline16')
    
    index = 0
    imgpath, imgname = makeimgname(datapath, recipmap2d_path, timestamp, datafile_dict, index, identifier = '_2dcake_', ext = '.png')

    totaltime = datafile_dict.get('totaltime')
    composition = datafile_dict.get('composition')
    sampnum = datafile_dict.get('sampnum')
    solutionnum = datafile_dict.get('solutionnum')

    if solutionnum is not None and totaltime is not None and composition is not None and sampnum is not None:
        imgheader = composition + ' - ' + solutionnum + ' - ' + sampnum + ' | Time: ' + totaltime
    else:
        imgheader = ""  # Provide a default value if either solutionnum or totaltime is None

    plt.title(imgheader, fontsize = 30) # title
    plt.xlabel('$\mathregular{q}$ ($\AA^{-1}$)', fontsize=58)
    plt.xticks(fontsize=40)
    plt.xlim(np.min(qr), np.max(qr))

    plt.ylabel('$\chi$ (deg.)',fontsize=58)
    plt.yticks(fontsize=40)
    plt.ylim(np.min(chi), np.max(chi))

    cake.set_clim(contrastmin, contrastmax)
    # cb = plt.colorbar(recipmap).set_label(label='',size=36)
    plt.tight_layout()
    plt.savefig(imgpath, dpi=500); # save image
    
    plt.close("all")
    plt.close()
    plt.close(fig)

    # delete loop iterated vars
    del fig
    plt.pause(.01)
    gc.collect()

    return

# format the integrated 1d image and save the image.
def format1dint(qrchi1d, intensity1d, datafile_dict, datapath, azimuth1d_path, timestamp, chilims):
    fig = figure(figsize = (20,12))
    
    index = 0
    identifier = '_chimin' + str(chilims[0]) + '_chimax' + str(chilims[1]) + '_azimuth1d_'
    imgpath, imgname = makeimgname(datapath, azimuth1d_path, timestamp, datafile_dict, index, identifier = identifier, ext = '.png')

    totaltime = datafile_dict.get('totaltime')
    composition = datafile_dict.get('composition')
    sampnum = datafile_dict.get('sampnum')
    solutionnum = datafile_dict.get('solutionnum')

    if solutionnum is not None and totaltime is not None and composition is not None and sampnum is not None and chilims is not None:
        imgheader = composition + ' - ' + solutionnum + ' - ' + sampnum + '_chimin' + str(chilims[0]) + '_chimax' + str(chilims[1]) + ' | Time: ' + totaltime
    else:
        imgheader = ""  # Provide a default value if either solutionnum or totaltime is None
    
    # imgheader = composition + ' - ' + solutionnum + ' - ' + sampnum + '_chimin' + str(chilims[0]) + '_chimax' + str(chilims[1]) + ' | Time: ' + totaltime
    plt.title(imgheader, fontsize = 30) # title
    plt.plot(qrchi1d,intensity1d)
    
    plt.title(imgheader, fontsize = 20)
    plt.tick_params(axis='both', which='major', labelsize=16) 
    plt.xlabel('q ($\AA^{-1}$)', fontsize=60)
    plt.xticks(fontsize=40)
    plt.xlim(np.min(qrchi1d), np.max(qrchi1d))

    plt.ylabel('Intensity (arb. units)',fontsize=60)
    plt.yticks(fontsize=40)
    plt.ylim(0, round(np.max(intensity1d)*1.05)) # multiplier to add height to the figure

    plt.tight_layout()
    plt.savefig(imgpath, dpi=200); # save image
    plt.close("all")
    plt.close()
    plt.close(fig)

    # delete loop iterated vars
    del fig
    plt.pause(.01)
    gc.collect()

    return

# format the 1d I(chi) pole figure and save the image.
def format1dchipole():
    # POLE FIGURE GENERATION
    # chi_intVals = chi_integrate(data_caked, qr, chi, polefig_params)
    
    # # GENERATE A POLE FIGURE OF THE SPECIFIED PEAK
    # figure(figsize = (20,12))
    # plt.plot(chi,chi_intVals)

    # # IMAGE TITLE
    # # img_title_Pole = samplename + '_poleFig' + '_chiMin' + str(lower_chi) + '_chiMax' +str(upper_chi) + '_qMin' + str(lower_q) + '_qMax' +str(upper_q) # Axis labelling on the output plot, and image rescaling.
    # img_title_Pole = 'Pole Fig. ' + 'Param.: chiMin' + str(lower_chi) + '_chiMax' +str(upper_chi) + '_qMin' + str(lower_q) + '_qMax' +str(upper_q) # Axis labelling on the output plot, and image rescaling.
    # plt.title(img_title_Pole, fontsize = 14)
    
    # # X-AXIS
    # plt.xlabel('$\chi$ (deg.)', fontsize = 22)
    # plt.xlim(0, upper_chi)
    
    # # Y-AXIS
    # plt.ylabel("Intensity (arb. units)", fontsize = 22)
    # plt.ylim(0, round(np.max(chi_intVals)*1.05))
    
    # plt.tick_params(axis='both', which='major', labelsize=16) # CAKED IMAGE TICK PARAMETERS
    
    # # SAVE THE IMAGE
    # plt.tight_layout()
    # plt.savefig(chi_linecut_csvpath, dpi=200)
    # plt.close()

    # # ------------------------------------------------- #
    # # ----------  (5.2) SAVE POLE FIGURE DATA --------- #
    # # ------------------------------------------------- #
    # # CREATE DATAFRAME FOR FILE OUTPUT
    # df_chi_integration = pd.DataFrame({'chi_integration': chi_intVals})
    # df_chi = pd.DataFrame({'chi': chi})
    # df_qr_chiint = pd.DataFrame({'q_r': qr})
    # df_chiint = df_chi_integration.assign(chi = df_chi)
    # df_chiint = df_chiint.assign(qr = df_qr_chiint)

    # # SAVE THE CSV
    # csvpath_poleFig = create_csvname(data_filepath, chi_linecut_path, identifier = '_poleFig_', extension = '.csv') # CREATE UNIQUE CSV PATH
    # create_CSVFile(df_chiint, csvpath_poleFig) # CREATE THE CSV FILES AND SAVE
    return

# ----- Complete Data Reduction Functions ----- #
# single giwaxs image reduction
# def giwaxs_reduce(datafolder, samplename, qparams, chiparams, peakparams, corrections, keylist, poni_file, mask_file, globStr = '*s_[0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]_maxs.tiff'):
def giwaxs_reduce(datafolder, samplename, qparams, chiparams, corrections, keylist, poni_file, mask_file, globStr = '*s_[0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]_maxs.tiff'):
    """ Takes a glob string and reduces all giwaxs images with string, for nsls-ii 11-bm cms. globStr input should define a single image extension for single image reduction."""

    plt.close('all') # close old plots from previous runs
    plt.pause(.01) # pause to clear the cache
    gc.collect() # garbage collection to free up memory
    
    # -- General Setup - List files for reduction, generate analysis folder path if not created
    datapaths = grabpaths(datafolder, globStr) # Grab filepath(s) from the specified file format in the glob string 'globStr' input field
    os.chdir(datafolder) # Make working directory the specified data folder
    os.chdir('..') # move out of the raw data folder
    cwd = os.getcwd() # get current working directory

    # build paths for reduced data
    analysispath = os.path.join(cwd, 'analysis/') # Make a data analysis filepath if it doesn't already exist
    if not os.path.exists(analysispath):
        os.makedirs(analysispath)
    
    os.chdir(analysispath) # Move into data analysis filepath
    cwd = os.getcwd() # get curent working directory
    savepath, timestamp = createfolder(samplename, cwd) # Create a uniquely timestamped savepath for the sample name in the data folder.

    # place 2d recip map/caked image/corresponding csv files in here
    recipmap2d_path = os.path.join(savepath, 'recipmap2d/') # save folder for reciprocal space maps
    if not os.path.exists(recipmap2d_path): # if this path doesn't exist create it
        os.makedirs(recipmap2d_path)

    # Put 1d integration(s)/pole figs/peak id in here.
    azimuth1d_path = os.path.join(savepath, 'azimuth1d/') # save folder for reciprocal space maps
    if not os.path.exists(azimuth1d_path): # if this path doesn't exist create it
        os.makedirs(azimuth1d_path) 
   
    # builds detector object and mask from metafile data (need to do this)
    datapath = datapaths[0] # grab a datafile to setup allocation based on file dimensions
    dataname = os.path.splitext(datapath)[0] + '.tiff' # strip filepath from directory and reappend '.tiff' extension
    data = fabio.open(dataname).data # open the '.tiff' file
    
    # mine metadata from file
    datafile_dict = {} # create empty datafile dictionary
    basetime = 0 # set null basetime for totaltime tr-giwaxs frame counter
    index = 0
    datafile_dict, dict_entryname = metadataminer(samplename, datapath, keylist, basetime, index)
    basetime = float(datafile_dict.get('clocktime')) # gives total time for the frame in the metadata output
    
    # create the pygix detector object for 2D image transform
    pg = pygix.Transform()
    pg, mask = buildDetObj(pg, datafile_dict, corrections, poni_file, mask_file)
    os.chdir(datafolder)

    # -- Execution - Loop through files in datapaths list to reduce data files sequentially.
    for index in range(0, len(datapaths)):    
        
        # -- Setup: Build set of analysis permutations to execute below.
        # should build an entire set of permutations from these three parameter inputs
        # qparams = # q integration parameters
        # chiparams =  # chi integration parameters to compare
        # poleparams = # pole figure bounds list
        # peakparams = # peak finder params

        chimin, chimax = chiparams
        chilims = [chimin, chimax]

        qmin, qmax = qparams
        qlims = [qmin, qmax]

        # -- Setup: Grab the datafile, strip down the metadata to a dict/dict_entryname
        datapath = datapaths[index] # grab the new datapath

        # -- Setup: Extract image metadata.
        datafile_dict = {} # clear the temp variable
        datafile_dict, dict_entryname = metadataminer(samplename, datapath, keylist, basetime, index) # grab the metadata and generate an entry name
        totaltime = round(float(datafile_dict.get('totaltime')), 2)
        basename = datafile_dict.get('basename')
        # exptime = float(datafile_dict.get('exptime'))
        incident_angle = float(datafile_dict.get('thpos'))
        pg.incident_angle = incident_angle # change incidence angle if different between scans

        # -- Setup: Unpack the 2D image data.
        dataname = os.path.splitext(datapath)[0] + '.tiff' # open the TIFF
        data = fabio.open(dataname).data # extract data

        # -- Reduction: Apply 2d reciprocal space map corrections.
        recip_data, qxy, qz = gixsdataTransform (data, corrections, pg, mask=mask, dark=None, flat=None, case='recip')
        # -- Image Format/File Save: save 2d reciprocal space map .png
        format2drecip(recip_data, qxy, qz, datafile_dict, datapath, recipmap2d_path, timestamp)
        # -- File Save: save 2d reciprocal space map .csv
        df_recip_data = pd.DataFrame(recip_data)
        df_qxy = pd.DataFrame({'q_xy': qxy})
        df_qz = pd.DataFrame({'q_z': qz})
        df_recipmap = df_recip_data.assign(qxy = df_qxy) # merge outputs into single array
        df_recipmap = df_recipmap.assign(qz = df_qz)
        csvpath_recipmap = create_csvname(datapath, recipmap2d_path, timestamp, totaltime, identifier = '_2drecip_', extension = '.csv')
        create_csvfile(df_recipmap, csvpath_recipmap) # create csv file and save
        
        # -- Reduction: Apply 2d caking corrections.
        caked_data, qr, chi = gixsdataTransform (data, corrections, pg, mask=mask, dark=None, flat=None, case='caked')
        # -- Image Format/File Save: save 2d caked image .png
        format2dcake(caked_data, qr, chi, datafile_dict, datapath, recipmap2d_path, timestamp)
        # -- File Save: Save 2d caked image .csv
        df_caked_data = pd.DataFrame(caked_data)
        df_qr = pd.DataFrame({'q_r': qr})
        df_chi = pd.DataFrame({'chi': chi})
        df_cakemap = df_caked_data.assign(qr = df_qr) # merge outputs into single array
        df_cakemap = df_cakemap.assign(chi = df_chi)
        csvpath_chimap = create_csvname(datapath, recipmap2d_path, timestamp, totaltime, identifier = '_2dcake_', extension = '.csv')
        create_csvfile(df_cakemap, csvpath_chimap) # create csv file and save

        # -- Reduction: integrate 2d caked image (over each specified chi bound and q lim)
        qrchi1d, intensity1d = cakedchiintegration(caked_data, chi, chilims, qr, qlims)
        # -- Image Format/File Save: save .png -> azimuth1d
        format1dint(qrchi1d, intensity1d, datafile_dict, datapath, azimuth1d_path, timestamp, chilims)
        # -- File Save: save .csv -> azimuth1d
        np_qrchi1d = np.array(qrchi1d) # Create numpy array of 1D q-range   
        np_intensity1d = np.array(intensity1d) # Create numpy array of 1D intensity values
        df_azimuth1d = pd.DataFrame({"qinvA" : np_qrchi1d, "intensity1d" : np_intensity1d}) #Create the dataframe.
        csvpath_azimuth1d = create_csvname(datapath, azimuth1d_path, timestamp, totaltime, identifier = '_azimuth1d_', extension = '.csv')
        create_csvfile(df_azimuth1d, csvpath_azimuth1d) # create csv file and save
        
        # -- Analysis: find peaks in the 1d integrated image
        # np_qrchi_peaks, np_intensity_peaks, np_numindex_peaks, df_peakid = peakindextool(qrchi1d, intensity1d, peakparams)
        # -- Image Format/File Save: save .png -> azimuth1d
        # applypeaklabels(qrchi1d, intensity1d, datafile_dict, datapath, azimuth1d_path, timestamp, chilims, np_qrchi_peaks, np_intensity_peaks, np_numindex_peaks, df_peakid)
        # -- File Save: save .csv -> azimuth1d
        # csvpath_peakid = create_csvname(datapath, azimuth1d_path, timestamp, totaltime, identifier = '_peakid_', extension = '.csv')
        # create_csvfile(df_peakid, csvpath_peakid) # create csv file and save

        # -- Reduction: create pole figures from 2d caked image
        # chi1d, poleintensity1d = cakedchipolefig(data, mask, chilims, qlims, bkgoffset = 0.02)
        # -- Image Format/File Save: save .png -> azimuth1d
        # -- File Save: save .csv -> azimuth1d
        # -- File Save: Display composite image w/ widget + save image + metadata file

        # -- Execution: - Print the active status.
        clear_output(wait=False) # clear terminal output
        print(basename + '\n')
        print ("Processing: " + dict_entryname) # print file for user to see integration progress
        # watchtime = stopwatch.now()
        # print (watchtime)
   
    return

# tr-GIWAXS series image reduction
def trGIWAXS(datafolder, samplename, chilims, qlims, corrections, keylist, poni_file, mask_file, saveOpt, globStr = '*s_[0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]_maxs.tiff'):
    """time-resolved giwaxs series integration for data from nslsii 11bmcms."""
    
    plt.close('all') # close old plots from previous runs
    # del datafile_dict
    # del fig
    # del recipmap
    # del data_recip

    plt.pause(.01)
    gc.collect() # garbage collection to free up memory
    # unpack list vars from frontend
    """ file save options
        # save2D = saveOpt[0]
        # save1D = saveOpt[1]
        # save2DCSV = saveOpt[2]
        # save1DCSV = saveOpt[3]
        # savetrGIWAXS = saveOpt[4]
        # savepeakID = saveOpt[5]
        # savepeakIDCSV = saveOpt[6]
        # savemeta = saveOpt[7]"""
    
    save2D, save1D, save2DCSV, save1DCSV, savetrGIWAXS, savecakeCSV, savepeakID, savepeakIDCSV, savemeta = saveOpt

    """ chi/q limits
        # chimin = chilims[0] # lower integration bound in chi
        # chimax = chilims[1] # upper integration bound in chi
        # qmin = qlims[0] # lower display bound in q
        # qmax = qlims[1] # upper display bound in q """
    
    chimin, chimax = chilims
    qmin, qmax = qlims

    """ glob string def
        # globStr = '*s_[0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]_maxs.tiff'"""
    
    # -- General Setup - List files for reduction, generate analysis folder path if not created
    # pull data filepaths, mine from glob string input
    datapaths = grabpaths(datafolder, globStr) # Grab filepaths from the specified file format in the glob string 'globStr' input field
    os.chdir(datafolder) # Make working directory the specified data folder
    os.chdir('..') # move out of the raw data folder
    cwd = os.getcwd() # get current working directory

    # build paths for reduced data
    analysispath = os.path.join(cwd, 'analysis/') # Make a data analysis filepath if it doesn't already exist
    if not os.path.exists(analysispath):
        os.makedirs(analysispath)
    
    os.chdir(analysispath) # Move into data analysis filepath
    cwd = os.getcwd() # get curent working directory
    savepath, timestamp = createfolder(samplename, cwd) # Create a uniquely timestamped savepath for the sample name in the data folder.

    # -- Option Dependent Path Generation
    if save2D == True: # if user wants to save 2D giwaxs images create the recip map filepath
        recipmap2d_path = os.path.join(savepath, 'recipmap2d/') # save folder for reciprocal space maps
        if not os.path.exists(recipmap2d_path): # if this path doesn't exist create it
            os.makedirs(recipmap2d_path) 
    
    if savetrGIWAXS ==True:
        trgiwaxs_path = os.path.join(savepath, 'trgiwaxs/') # save folder for tr-giwaxs
        if not os.path.exists(trgiwaxs_path): # if this path doesn't exist create it
            os.makedirs(trgiwaxs_path)
   
    if savecakeCSV ==True:
        trcake_path = os.path.join(savepath, 'trcake/') # for caked amd corrected 2d data stack
        if not os.path.exists(trcake_path): # if this path doesn't exist create it
            os.makedirs(trcake_path)

    # -- General Setup - Builds detector object and mask from metafile data
    # grab pathname info 
    datapath = datapaths[0] # grab a datafile to setup allocation based on file dimensions
    dataname = os.path.splitext(datapath)[0] + '.tiff' # strip filepath from directory and reappend '.tiff' extension
    data = fabio.open(dataname).data # open the '.tiff' file
    
    # mine metadata from file
    datafile_dict = {} # create empty datafile dictionary
    basetime = 0 # set null basetime for totaltime tr-giwaxs frame counter
    index = 0
    datafile_dict, dict_entryname = metadataminer(samplename, datapath, keylist, basetime, index)
    basetime = float(datafile_dict.get('clocktime'))
    
    # create the pygix detector object for 2D image transform
    pg = pygix.Transform()
    pg, mask = buildDetObj(pg, datafile_dict, corrections, poni_file, mask_file)
    os.chdir(datafolder)

    """ if buildDetObj() fails for some reason uncomment below
        # ai = pyFAI.load(poni_file) # if using pyFAI azimuthal integrator
        # pg = pygix.Transform() # create detector transform object, 'pg' for pygix detector 
        # pg.load(poni_file) # load the poni file into the transform object
        # pg.sample_orientation = corr_samporient # if error with this, set to 3 for GI geometry
        # mask = fabio.open(mask_file).data # load mask file
        # os.chdir(datafolder) # move to the /raw datafolder """

    """ # sequence for building null arrays of the appropriate size
        # Setup array dimensions for null allocation and indexing
        # datapath = datapaths[0] # grab a datafile to setup allocation based on file dimensions
        # dataname = os.path.splitext(datapath)[0] + '.tiff' # strip filepath from directory and reappend '.tiff' extension
        # data = fabio.open(dataname).data # open the '.tiff' file """

    # -- Option Dependent Memory Allocation - Creates appropriately sized loop arrays
    if savetrGIWAXS == True: # create empty arrays for trgiwaxs
        caked_data, qr, chi = gixsdataTransform (data, corrections, pg, mask=mask, dark=None, flat=None, case='caked')
        qr1d, intensity1d = cakedchiintegration(caked_data, chi, chilims, qr, qlims)

        intensity_time = np.zeros([len(intensity1d),len(datapaths)]) # null array for intensity stack
        qr_time = np.zeros([len(qr1d),len(datapaths)]) # null array for qr stack
        map_time = np.zeros([1, len(datapaths)]) # null array for time series

    # -- General Execution - Loop through files in datapaths list to reduce data files sequentially.
    for index in range(0, len(datapaths)):    
        
        # -- General Setup - Grab the datafile, strip down the metadata to a dict/dict_entryname
        datafile_dict = {} # clear the temp variable
        datapath = datapaths[index] # grab the new datapath
        datafile_dict, dict_entryname = metadataminer(samplename, datapath, keylist, basetime, index) # grab the metadata and generate an entry name
        # datafiles_dict [dict_entryname] = datafile_dict # store the file dictionary to the collection using the entry name
        
        # -- General Setup - Grab and typecast metadata mined from data filepath
        incident_angle = float(datafile_dict.get('thpos'))
        pg.incident_angle = incident_angle # change incidence angle if different between scans
        totaltime = round(float(datafile_dict.get('totaltime')), 2)
        basename = datafile_dict.get('basename')
        exptime = float(datafile_dict.get('exptime'))

        # -- General Setup - extract the loop data from '.tiff' using fabIO
        dataname = os.path.splitext(datapath)[0] + '.tiff' # open the TIFF
        data = fabio.open(dataname).data # extract data

        """ # if datafile_dict process is to fail, uncomment and use this.
            # if index == 0:
            #     map_time[0,i] = 0
            # else:
            #     if exposure_time <= 0.495:
            #         map_time[0,i] = map_time[0,i-1] + (exposure_time + 0.005)
            #     else:
            #         map_time[0,i] = map_time[0,i-1] + exposure_time """

        """ # save cases to expand path generation options later on
            # if save2DCSV != 0:
            #     recipmap2dCSV_path = os.path.join(savepath, 'recipmap2dcsv/') # save folder for reciprocal space maps
            #     if not os.path.exists(recipmap2dCSV_path): # if this path doesn't exist create it
            #         os.makedirs(recipmap2dCSV_path)

            # if save1D != 0: # if user wants to save 1D integrated image create the filepath
            #     azimuthint1d_path = os.path.join(savepath, 'azimuthint1d/') # save folder for reciprocal space maps
            #     if not os.path.exists(azimuthint1d_path): # if this path doesn't exist create it
            #         os.makedirs(azimuthint1d_path)
            
            # if save1DCSV != 0:
            #     azimuthint1dCSV_path = os.path.join(savepath, 'azimuthint1dcsv/') # save folder for reciprocal space maps
            #     if not os.path.exists(azimuthint1dCSV_path): # if this path doesn't exist create it
            #         os.makedirs(azimuthint1dCSV_path)
            
            # if savepeakID != 0:
            #     peakID1d_path = os.path.join(savepath, 'peakID1d/') # save folder for tr-giwaxs
            #     if not os.path.exists(peakID1d_path): # if this path doesn't exist create it
            #         os.makedirs(peakID1d_path)

            # if savepeakIDCSV != 0:
            #     peakID1dCSV_path = os.path.join(savepath, 'peakID1dcsv/') # save folder for tr-giwaxs
            #     if not os.path.exists(peakID1dCSV_path): # if this path doesn't exist create it
            #         os.makedirs(peakID1dCSV_path)

            # if savemeta != 0:
            #     metadata_path = os.path.join(savepath, 'filemetadata/') # save folder for tr-giwaxs
            #     if not os.path.exists(metadata_path): # if this path doesn't exist create it
            #         os.makedirs(metadata_path) """

        # -- Option Dependent Data Reduction - Check unpacked saveOpt cases for which reductions to perform and store.
        # - Performs 1d integration, does not save 1d files, rather, a concatenated set 1d ints, q vals, and time stamps.
        if savetrGIWAXS == True: 
            # cake the data, then, integrate the caked data
            caked_data, qr, chi = gixsdataTransform (data, corrections, pg, mask=mask, dark=None, flat=None, case='caked')

            if savecakeCSV == True:
                # print ('the cake is not a lie')
                # caked_data, qr, chi = gixsdataTransform (data, corrections, pg, mask=mask, dark=None, flat=None, case='caked')
                
                # create dataframes from image arrays
                df_caked_data = pd.DataFrame(caked_data)
                df_qr = pd.DataFrame({'q_r': qr})
                df_chi = pd.DataFrame({'chi': chi})

                # merge outputs into single array
                df_cakemap = df_caked_data.assign(qr = df_qr)
                df_cakemap = df_cakemap.assign(chi = df_chi)
                
                # CREATE UNIQUE CSV PATH
                csvpath_chimap = create_csvname(datapath, trcake_path, timestamp, totaltime, identifier = '_2dcake_', extension = '.csv')
                create_csvfile(df_cakemap, csvpath_chimap) # create csv file and save

            qr1d, intensity1d = cakedchiintegration(caked_data, chi, chilims, qr, qlims)
            
            # format qr and intensity as numpy arrays
            np_qr1d = np.array(qr1d)
            np_intensity1d = np.array(intensity1d)
            
            # give them titles for the dataframe headers
            title_qr = "q_invA_" + str(index)
            title_intensity = "intensity_" + str(index)

            # create the dataframe - this is needed to save a CSV for 1dint array if desired
            df_int1d = pd.DataFrame(None)
            df_int1d = pd.DataFrame({title_qr : np_qr1d, title_intensity : np_intensity1d}) #Create the dataframe.

            df_time = pd.DataFrame(None) # Declare the dataframe for the time series integration
            df_time = pd.concat([df_time, df_int1d], axis=1)

            qr_1d = np.array(df_int1d.drop([title_intensity],axis=1))
            intensity_1d = np.array(df_int1d.drop([title_qr],axis=1))
            
            intensity_time[:,index] = intensity_1d[:,0]
            qr_time[:, index] = qr_1d[:,0]
            map_time[0,index] = totaltime

        # save the 2d reciprocal space maps as pngs
        if save2D == True: # if user wants to save 2D giwaxs images create the recip map filepath
            # recip space transform
            data_recip, qxy, qz = pg.transform_reciprocal(data, # Convert detector image to q-space (sample reciprocal)
                                                method = 'bbox', #for some reason splitpix doesnt work?
                                                unit='A', # unit='A'
                                                mask=mask, correctSolidAngle = True)
            
            fig = figure(figsize = (28,14))
            
            img_min = np.percentile(data_recip, 2) # Plot q-space detector image (color scaling minimum)
            img_max = np.percentile(data_recip, 99.95) # Plot q-space detector image (color scaling maximum)
            
            matplotlib.use('agg') # swap the backend to help with memory leaks
            recipmap = plt.imshow(data_recip, # Create reciprocal space map with qr Ewald sphere correction applied.
                norm=matplotlib.colors.Normalize(vmin=img_min,vmax=img_max),
                cmap='turbo', # color map formatting
                extent=(np.min(qxy),np.max(qxy),np.min(qz),np.max(qz)), # extent defines the visual bounds of the image.
                origin = 'lower');
        
            imgpath, imgname = makeimgname(datapath, recipmap2d_path, timestamp, datafile_dict, index, identifier = '_2drecipmap_', ext = '.png')

            thpos = datafile_dict.get('thpos')
            totaltime = float(datafile_dict.get('totaltime'))
            composition = datafile_dict.get('composition')
            solvsys = datafile_dict.get('solvsys')
            purgerate = datafile_dict.get('purgerate')
            sampnum = datafile_dict.get('sampnum')
            solutionnum = datafile_dict.get('solutionnum')

            # imgheader = composition + ' - ' + solvsys + ', solnum: ' + solutionnum + ' | thpos: ' + thpos + ' deg. , purge rate: ' + purgerate + ' scfh , runnum #: ' + sampnum + ' | Time' + str(totaltime)
            imgheader = composition + ' - ' + solvsys + ' | Time: ' + str(totaltime)
            plt.title(imgheader, fontsize = 30) # title
            plt.xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)', fontsize=58) # x label
            plt.xticks(fontsize=40)
            plt.xlim(np.min(qxy), np.max(qxy))

            plt.ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)',fontsize=58) # y label
            plt.yticks(fontsize=40)
            plt.ylim(0, np.max(qz))

            plt.tight_layout();
            cb = plt.colorbar(recipmap).set_label(label='',size=36)
            plt.savefig(imgpath, dpi=500); # save image
            plt.close("all")
            plt.close()
            plt.close(fig)
            # delete loop iterated vars
            del datafile_dict
            del fig
            del recipmap
            del data_recip
            plt.pause(.01)
            gc.collect()
        
        # -- General Execution - Print the active status.
        # print status 
        clear_output(wait=False) # clear terminal output
        print(basename + '\n')
        print ("Processing: " + dict_entryname) # print file for user to see integration progress
        
        # watchtime = stopwatch.now()
        # print (watchtime)
        """ memory cache tracking and dictionary clearing 
            # print (str(len(datafile_dict)))
            # datadict_byte = sys.getsizeof(datafile_dict)
            # print (str(datadict_byte))
            # datafile_dict.clear() # clear the dictionary on each iteration to prevent overgrowth/metadata mingling
            # total_memory, used_memory, free_memory = map(
            #     int, os.popen('free -t -m').readlines()[-1].split()[1:]) # Getting all memory usage using os.popen()
            # print("RAM memory used:", round((used_memory/total_memory) * 100, 2)) # print memory usage """
   
    if savetrGIWAXS == True:
        os.chdir(trgiwaxs_path) # move into directory
        # -- Plotting tr-GIWAXS Data
        
        figure(figsize = (25,25)) 

        timeMax = 1639
        extent = np.min(map_time), 820, np.min(qr_time), np.max(qr_time) # set image bounds x/y
        # extent = np.min(map_time), np.max(map_time), np.min(qr_time), np.max(qr_time) # set image bounds x/y
        
        dispCut = intensity_time[:, 0:timeMax]
        trWAXS2D = plt.imshow(dispCut, cmap='turbo', extent=extent, origin = "lower", aspect='auto')
        
        imgpath_trgiwaxs, imgname_trgiwaxs = create_imagename(datapath, trgiwaxs_path, timestamp, totaltime, identifier = '_trGIWAXS_', extension = '.png')

        plt.title("(tr-GIWAXS) " + samplename, fontsize = 40)
        plt.tick_params(axis='both', which='major', labelsize=32) # Image tick parameters
        
        plt.xlabel('Time (s)', fontsize=60)
        # plt.xlim(np.min(map_time), np.max(map_time))
        plt.xlim(np.min(map_time), 820)

        plt.ylabel('$\mathregular{q}$ ($\AA^{-1}$)',fontsize=60)
        plt.ylim(np.min(qr_time),np.max(qr_time))

        contrastmin = np.nanpercentile(np.min(dispCut),0) #COLOR SATURATION LIMIT
        contrastmax = np.nanpercentile(np.max(dispCut),100) #COLOR SATURATION LIMIT
        trWAXS2D.set_clim(contrastmin, contrastmax)

        plt.tight_layout()
        plt.savefig(imgpath_trgiwaxs, dpi=500) # save the image

        # -- Saving tr-GIWAXS Files
        csvpath_intensity_time = create_csvname(datapath, trgiwaxs_path, timestamp, totaltime, identifier = '_trGIWAXS_intensityvals_', extension = '.csv')
        df_intensity_time = pd.DataFrame(intensity_time)
        create_csvfile(df_intensity_time, csvpath_intensity_time)

        csvpath_qr_time = create_csvname(datapath, trgiwaxs_path, timestamp, totaltime, identifier = '_trGIWAXS_qrvals_', extension = '.csv')
        df_qr_time = pd.DataFrame(qr_time)
        create_csvfile(df_qr_time, csvpath_qr_time)

        csvpath_map_time = create_csvname(datapath, trgiwaxs_path, timestamp, totaltime, identifier = '_trGIWAXS_timevals_', extension = '.csv')        
        df_map_time = pd.DataFrame(map_time)
        create_csvfile(df_map_time, csvpath_map_time)

        # xmax = np.max(map_time)
        # xmin = np.min(map_time)
        # xticks = np.arange(xmin, xmax, xmax/10)
        # plt.xticks(xticks, fontsize = 28)
    
        # ymin = np.round(np.min(qr_time),3)
        # ymin = ymin + 0.005
        # # print (ymin)
        # yticks = [ymin, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75]
        # plt.yticks(yticks, fontsize = 28)
        # plt.ylim(ymin, 2.75)

        # memory cleanup for loop iteration solving
        # del datafile_dict
        # del fig
        # plt.pause(.01)
        # gc.collect()
   
    """ # -- pseudocode -- #
        # create a corrections setup function to define a corrections object that interfaces with the detector transform class
        # in new session, regenerate the defined corrections object from an import function that calls the file class/directory designed for that object
            # corrections object settings can be saved as a custom file type w/ a custom unique identifier to be later reloaded
            # settings contain correction params + metadata, e.g.: lorentz, sinchi, polarization, poni-path, mask-path, bkg-settings, etc.
        # use the corrections object to interface with the pygix detector class to transform raw .tiff images
        # store caked transform .rtf /.csv /.xy files in a folder specific to the corrections object w/ the assigned identifier
        # in future runs, if that corrections object is recalled, check if the caked images exist in the related directory generated for that object
            # if yes, no need to regenerate files - we can operate on the process caked datafiles for our chi-integrations/pole figure generation
        
        # pseudocode for above
            # User Inputs: data folder, sample name parameters, chi integration parameters, 
            # poni/bkg/mask files, image correction parameters, file mining parameters, boolean option to save 2D images
            # load data folder
            # create a list of all the .tiff files inside of the folder that match conditions for trGIWAXS
            # mine the filename, create an array of mined data linked to the filepath
            # calculate the timestamp from the mined data (timestamp + exp*framenubmber)- add this to the 2D array
            # create a unique directory for the data analysis
            # create a subfolder for the 2D images and time series integrated data
                # we can use this folder to create montages of the time-series images
                # create a blank .csv for the time series integrations
                # create blank .csv for the q-ranges (we will write directly to these each iteration rather than storing to RAM)
                # create blank .csv for metadata array
            # each iteration (loop over length)
                # open .tiff
                # apply image corrections with  (poni, optional corrections: sinchi, etc)
                # set saturation limits and save 2D image (if desired)
                # cake the image
                # integrate the caked image over specified chi with pixel-pixel averaging
                # normalize to exposure time
                # open .csv files
                # write to q.csv, int1D.csv, metadata_array
                # save/close .csv files """

    return

def plot_trWAXS(imgpath, trgiwaxs_path, map_time, qr_time, intensity_time, plotparams):
    
    time_min, time_max, exptime, header, headerfontsize, xfontsize, yfontsize, autosave, qmin, qmax, cmin, cmax, cmap, imgdpi, ext, samplename, tickfontsize, cblabel, cbsize = plotparams
    """ # Plotting Parameters
    # [0] time_max : float
    # [1] time_min : float
    # [2] exptime : float
    # [3] header : string
    # [4] headerfontsize : integer
    # [5] xfontsize : integer
    # [6] yfontsize : integer
    # [7] autosave : boolean
    # [8] qmin :float
    # [9] qmax : float
    # [10] cmin : float
    # [11] cmax : float
    # [12] cmap : string
    # [13] ext : string
    """

    plt.close('all')
    plt.pause(0.01)
    gc.collect()

    os.chdir(trgiwaxs_path) # set directory with trgiwaxs data
    
    figure(figsize = (10,8)) # generate figure
    
    slice_min = int(float(time_min)/np.round(exptime,2))
    slice_max = int(float(time_max)/np.round(exptime,2))
    mapslice = intensity_time[:, slice_min:slice_max] # cutout the data to plot

    extent = np.min(map_time), np.max(map_time), np.min(qr_time), np.max(qr_time) # spans the entire dataset - axes rescaled with xlim/ylim

    # normmin = np.min(mapslice)
    # normmax = np.max(mapslice)

    contrastmin = np.percentile(mapslice, cmin)
    contrastmax = np.percentile(mapslice, cmax)

    img = plt.imshow(mapslice, 
                    norm=matplotlib.colors.Normalize(vmin=contrastmin,vmax=contrastmax), 
                    cmap='turbo', 
                    extent=extent, 
                    origin = "lower", 
                    aspect='auto')

    plt.tick_params(axis='both', which='major', labelsize=tickfontsize) # Image tick parameters
    
    plt.title(header, fontsize = headerfontsize)
    
    plt.xlabel('Time (s)', fontsize=xfontsize)
    # plt.xlim(np.min(map_time), np.max(map_time))
    plt.xlim(time_min, time_max)

    plt.ylabel('$\mathregular{q}$ ($\AA^{-1}$)',fontsize=yfontsize)
    # plt.ylim(np.min(qr_time),np.max(qr_time))
    plt.ylim(qmin, qmax)

    if cmin > cmax:
        cmin = 0
        cmax = 100
        print ("Invalid contrast scaling limits, setting cmin = 0, cmax = 100.")
    
    if cmin < 0:
        cmin = 0
        print("Invalid cmin contrast limit, setting cmin = 0.")
    
    if cmax > 100:
        cmax = 100
        print("Invalid cmax contrast limit, setting cmax = 100.")
    
    # contrastmin = np.nanpercentile(np.min(mapslice), cmin) # color saturation minimum
    # contrastmax = np.nanpercentile(np.max(mapslice), cmax) # color saturation maximum
    # plt.set_cmap(cmap) # set the color map appropriately
    # img.set_clim(contrastmin, contrastmax)
    # cbar = plt.colorbar(img).set_label(label=cblabel, size=cbsize)

    plt.tight_layout()
    plt.colorbar(img)

    if autosave == True:
        plt.savefig(imgpath, dpi=imgdpi) # save the image if desired
        print("Image Saved: " + str(imgpath))
        
    return

def loadcsv_trgiwaxs(filelist, plotparams):
    filepath_map_time, filepath_qr_time, filepath_intensity_time  = filelist

    plt.close() # close the previous plot

    map_time = np.loadtxt(filepath_map_time, dtype="double", delimiter=",")
    qr_time = np.loadtxt(filepath_qr_time, dtype="double", delimiter=",")
    intensity_time = np.loadtxt(filepath_intensity_time, dtype="double", delimiter=",")
    
    map_time = np.delete(map_time, 0, 0)
    qr_time = np.delete(qr_time, 0, 0)
    intensity_time = np.delete(intensity_time, 0, 0)

    # path = "/"
    trgiwaxs_path = os.path.dirname(os.path.abspath(filepath_map_time))
    # trgiwaxs_path = os.path.join(trgiwaxs_path, path)

    img_timestamp = time.strftime("%Y%m%d_%H%M%S") # Create a time string, with Year, Month, Day + '_' + Hour, Minute, Second. This is appended to the save_path to give the analyzed/reduced data a unique identifier.

    cmin = plotparams[10]
    cmax = plotparams[11]
    ext = plotparams[14]
    samplename = plotparams[15]

    # imgpath = os.path.splitext(filepath_map_time)[0] + '_' + str(img_timestamp) + "_cmin" + str(cmin) + "_cmax" + str(cmax) + ext
    imgname = samplename + '_' + str(img_timestamp) + "_cmin" + str(cmin) + "_cmax" + str(cmax) + ext
    # imgname = os.path.basename(imgpath) # Strip the filename from the data filepath
    imgpath = os.path.join(trgiwaxs_path, imgname)

    plot_trWAXS(imgpath, trgiwaxs_path, map_time, qr_time, intensity_time, plotparams)
    return

def giwaxs_single(datafolder, samplename, corrections, keylist, poni_file, mask_file, globStr = '*s_[0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]_maxs.tiff'):
    plt.close('all') # close old plots from previous runs
    plt.pause(.01) # pause to clear the cache
    gc.collect() # garbage collection to free up memory
    
    # -- General Setup - List files for reduction, generate analysis folder path if not created
    datapath = grabpaths(datafolder, globStr)[0] # Grab filepath(s) from the specified file format in the glob string 'globStr' input field
    os.chdir(datafolder) # Make working directory the specified data folder
    os.chdir('..') # move out of the raw data folder
    cwd = os.getcwd() # get current working directory

    # build paths for reduced data
    analysispath = os.path.join(cwd, 'analysis/') # Make a data analysis filepath if it doesn't already exist
    if not os.path.exists(analysispath):
        os.makedirs(analysispath)
    
    os.chdir(analysispath) # Move into data analysis filepath
    cwd = os.getcwd() # get current working directory
    savepath, timestamp = createfolder(samplename, cwd) # Create a uniquely timestamped savepath for the sample name in the data folder.

    # place 2d recip map/caked image/corresponding csv files in here
    recipmap2d_path = os.path.join(savepath, 'recipmap2d/') # save folder for reciprocal space maps
    if not os.path.exists(recipmap2d_path): # if this path doesn't exist create it
        os.makedirs(recipmap2d_path)
   
    # builds detector object and mask from metafile data (need to do this)
    dataname = os.path.splitext(datapath)[0] + '.tiff' # strip filepath from directory and reappend '.tiff' extension
    data = fabio.open(dataname).data # open the '.tiff' file
    
    # mine metadata from file
    datafile_dict = {} # create empty datafile dictionary
    basetime = 0 # set null basetime for totaltime tr-giwaxs frame counter
    index = 0
    datafile_dict, dict_entryname = metadataminer(samplename, datapath, keylist, basetime, index)
    basetime = float(datafile_dict.get('clocktime')) # gives total time for the frame in the metadata output
    framenum = float(datafile_dict.get('framenum'))
    exptime = round(float(datafile_dict.get('framenum')),2)
    totaltime = basetime + (exptime*framenum)
    
    # create the pygix detector object for 2D image transform
    pg = pygix.Transform()
    pg, mask = buildDetObj(pg, datafile_dict, corrections, poni_file, mask_file)
    incident_angle = float(datafile_dict.get('thpos'))
    pg.incident_angle = incident_angle # change incidence angle if different between scans
    os.chdir(datafolder)

    # -- Setup: Unpack the 2D image data.
    dataname = os.path.splitext(datapath)[0] + '.tiff' # open the TIFF
    data = fabio.open(dataname).data # extract data

    # -- Reduction: Apply 2d reciprocal space map corrections.
    recip_data, qxy, qz = gixsdataTransform (data, corrections, pg, mask=mask, dark=None, flat=None, case='recip')
    # -- Image Format/File Save: save 2d reciprocal space map .png
    format2drecip(recip_data, qxy, qz, datafile_dict, datapath, recipmap2d_path, timestamp)
    # -- File Save: save 2d reciprocal space map .csv
    df_recip_data = pd.DataFrame(recip_data)
    df_qxy = pd.DataFrame({'q_xy': qxy})
    df_qz = pd.DataFrame({'q_z': qz})
    df_recipmap = df_recip_data.assign(qxy = df_qxy) # merge outputs into single array
    df_recipmap = df_recipmap.assign(qz = df_qz)
    csvpath_recipmap = create_csvname(datapath, recipmap2d_path, timestamp, totaltime, identifier = '_2drecip_', extension = '.csv')
    create_csvfile(df_recipmap, csvpath_recipmap) # create csv file and save
    
    return

def loadcsv_2drecipmap(csvpath):

    df_recipmap = pd.read_csv(csvpath)
    df_qxy = df_recipmap['qxy']
    np_qxy = np.array(df_qxy)
    np_qxy = np_qxy[np.logical_not(np.isnan(np_qxy))]

    df_qz = df_recipmap['qz']
    np_qz = np.array(df_qz)
    np_qz = np_qz[np.logical_not(np.isnan(np_qz))]

    df_recipmap = df_recipmap.drop(['qxy'], axis=1)
    df_recipmap = df_recipmap.drop(['qz'], axis=1)
    np_recipmap = np.array(df_recipmap)

    return np_recipmap, np_qxy, np_qz

# -- Zihan Functions
def cut(a1,a2,a3,a4,b1,b2,b3,b4,xp,yp,data):
    xL=math.floor((b2-b1)/(a2-a1)*xp)
    yL=math.floor((b4-b3)/(a4-a3)*yp)
    xs=math.floor((b1-a1)/(a2-a1)*xp)
    ys=math.floor((b3-a3)/(a4-a3)*yp)
    return data[ys:ys+yL,xs:xs+xL]

# integrate GIWAXS to get I vs qz. a1-a4 are original image size (qyxmin,qxymax,qzmin,qzmax); b1-b4 are the area to be integrated.
# xp,yp are how many points in xy and z direction. Data is GIWAXS data.
def qzint(a1,a2,a3,a4,b1,b2,b3,b4,xp,yp,data):
    xL=math.floor((b2-b1)/(a2-a1)*xp)
    yL=math.floor((b4-b3)/(a4-a3)*yp)
    xs=math.floor((b1-a1)/(a2-a1)*xp)
    ys=math.floor((b3-a3)/(a4-a3)*xp)
    data1=data[ys:ys+yL,xs:xs+xL]
    data2=sum(np.transpose(data1))
    data3=np.zeros([yp,xp])
    data3[ys:ys+yL,xs:xs+xL]=data[ys:ys+yL,xs:xs+xL]
    return data2,data3

# integrate GIWAXS to get I vs qxy.
def qxyint(a1,a2,a3,a4,b1,b2,b3,b4,xp,yp,data):
    xL=math.floor((b2-b1)/(a2-a1)*xp)
    yL=math.floor((b4-b3)/(a4-a3)*yp)
    xs=math.floor((b1-a1)/(a2-a1)*xp)
    ys=math.floor((b3-a3)/(a4-a3)*xp)
    data1=data[ys:ys+yL,xs:xs+xL]
    data2=sum(data1)
    data3=np.zeros([yp,xp])
    data3[ys:ys+yL,xs:xs+xL]=data[ys:ys+yL,xs:xs+xL]
    return data2,data3

# integrate GIWAXS to get I vs q.
def angularint(a1,a2,a3,a4,qp,angle1,angle2,qqmin,xp,yp,data):
    data1=np.zeros([yp,xp])
    xline=np.linspace(1,xp-1,xp-1)
    xline=xline.astype(int)
    yline=np.linspace(1,yp-1,yp-1)
    yline=yline.astype(int)
    I=np.zeros(qp)
    for i in xline:
        for j in yline:
            a=(a2-a1)/xp*i+a1
            b=(a4-a3)/yp*j+a3
            q=np.sqrt(a*a+b*b)
            qi=math.floor(q/(a4/qp))
            angle=np.arccos(a/q)
            if angle>angle1:
                if angle<angle2:
                    if q<a4:
                        if q>qqmin:
                            I[qi]=I[qi]+data[j,i]
                            data1[j,i]=data1[j,i]+data[j,i]
    return I,data1

# Python To Do (04/19/23)
    # fix monte-carlo peak finder for simulation data
    # implement (hkl) indexing tool
    # setup research account/understand alpine allocation process
    # reading on DASK parallelization
    # non-linear least squares minimization for 2D images.

    # download pyhyperscattering

    # create dynamic AFF.py with pyDABAX
    # create CSV loader functions
        # trGIWAXS loader
        # caked 2d image loader
    # 1d waterfall plotting from trGIWAXS CSV
    # dezingering function
    # how to deal with exposures with lower intensity?
    # dynamic plt.imshow() updates in trGIWAXS function
    # dynamic writing for CSV outputs - each loop iteration