import re, sys, os
import pandas as pd
import glob2 as glob
import dask as dk

import numpy as np
import scipy as sp
from scipy import signal
from scipy.optimize import curve_fit
from scipy.special import wofz
from scipy.signal import fftconvolve
from scipy.signal import argrelextrema
from scipy.signal import find_peaks, peak_prominences, peak_widths

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

import pygix
import pyFAI
import pyFAI.gui
import pyFAI.detectors
import pyFAI.calibrant
print("Using pyFAI version", pyFAI.version)

class DetectorCorr():
    """Defines a detector object that is called within a GIXSReduce() session 
    to apply image corrections to a raw GIXS image file."""
    def __init__(self):
        self.detname = 'Pilatus1M' # define detector name
        self.ponipath = None # define poni filepath (.edf)
        self.maskpath = None # define mask filepath (.edf)
        self.incidentangle = 0.3 # default incidence angle
        self.sample_orientation = 3 # can be integer from 1 - 3

class GIXSReduce:
    def __init__(self):
        detsettings = DetectorCorr() # establish the detector settings
        self.datafolder = None
        self.filelist = None
        self.globdelim = None
        self.samplename = None
        self.azimuth1Dpath, self.pole1Dpath, self.recip2Dpath, self.cake2Dpath, self.trGIXSpath, self.peakindexpath