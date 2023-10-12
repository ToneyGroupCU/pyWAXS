##################################################################################################################################
# CUSTOM CONDA ENVIRONMENT: 'pyWAXS' - need to generate .yml file for env reproduction.
# Jupyter Notebook Kernel: keithwhite@Keiths-MacBook-Pro/opt/anaconda3/envs/pyWAXS
# ----------------------------------------------------------------------------------------- #
# Contributors: Zihan Zhang, Keith White
# Toney Group, University of Colorado Boulder
# Updated: 04/07/2023
# Version Number: NSLS-II, Version 1.3
# Description: 2D diffraction simulation for anisotropically oriented crystallites.
##################################################################################################################################

# -- IMPORT LIBRARIES ------------------------------------------------ #
import os, gc, time
import numpy as np
import scipy as sp
import glob2 as glob
import scipy.ndimage as ndimage

# -------- Matplotlib -------- #
import matplotlib
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.tri as tri
from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.pyplot import figure
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec
import fractions
# -------------------------------------------------------------------- #

# -------- .py Scripts -------- #
import AFF
import gixsreducesim_script as WAXS
# import montecarlo_peaks as mcpeaks
# import cif2vasp as cif2vasp

# -------------------------------------------------------------------- #
# def convertCIF (CIF_filepath):
#     cif2vasp.readCifFile(CIF_filepath)
#     return

# -- Description: Strips down the input path (filename) and returns the parent folder.
def stripparentpath (poscar_path):
    parent_path = os.path.dirname(os.path.abspath(poscar_path))
    return parent_path

# -- Description: Files the filepath from an input string and parent folderpath.
def findfilepath (parent_folder, fileStr):
    # file_path = glob.iglob(parent_folder + fileStr)
    pathStr = parent_folder + fileStr
    filepath = sorted(glob.glob(pathStr))
    return filepath

# -- Description: Reads the POSCAR metadata from specified address.
def readPOSCAR(address):
    """ Function Description: Reads the address of the input POSCAR file. POSCAR file description (from web), 
    "This file contains the lattice geometry and the ionic positions, optionally also starting velocities and 
    predictor-corrector coordinates for a MD-run."
"""
    file = open(address)
    ind = 0
    for x in file:
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
            iii = 0
            pp=np.zeros(len(x))
            for p in x:
                pp[iii]=AFF.atom_dict[p]
                iii += 1
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
        if ind>8:
            x=x.lstrip()
            x=x.rstrip()
            x=x.split("         ")
            x=np.array(x)
            # print (x)
            p=x.astype(float)
            position[ind-9,1]=p[0]
            position[ind-9,2]=p[1]
            position[ind-9,3]=p[2]
    file.close()
    
    ind = 0
    iii = 0
    for ii in z1:
        position[iii:iii+ii+1,0]=z[ind]
        iii =iii+ii
        ind =ind+1
    return a1,a2,a3,position

# -- Description: Simulates Bragg peaks from input file in loadPOSCAR()
def Bragg_Peaks(a1,a2,a3,positions,theta_x,theta_y,hkl_dimension):
    """ Function Description: Calculates the position of Bragg peaks in reciprocal space 
    using lattice parameters and position of atoms read from the POSCAR file. Two rotation 
    angles are added with respect to x and y axes to adjust the orientation of the single crystal.
    """
    
    # Lattice parameters M matrix in cartesian coordinate(angstrom)
    M=[a1,a2,a3]
    M=np.asarray(M)
    # Rotation Matrix respect to X axis, rotation angle = theta_x
    Rx=np.array([[1,0,0],[0,np.cos(theta_x),-np.sin(theta_x)],[0,np.sin(theta_x),np.cos(theta_x)]])
    # Rotation Matrix respect to Y axis, rotation angle = theta_y
    Ry=np.array([[np.cos(theta_y),0,-np.sin(theta_y)],[0,1,0],[np.sin(theta_y),0,np.cos(theta_y)]])

    # Rotation of the sample
    M=np.matmul(M, Rx)
    M=np.matmul(M, Ry)
    
    # New lattice parameter
    aa1=M[0,:]
    aa2=M[1,:]
    aa3=M[2,:]

    # reciprocal lattice
    volume=np.matmul(aa3,np.cross(aa1,aa2))
    b1=2*np.pi*np.cross(aa2,aa3)/volume
    b2=2*np.pi*np.cross(aa3,aa1)/volume
    b3=2*np.pi*np.cross(aa1,aa2)/volume

    # grid for Miller index
    i =np.linspace(-hkl_dimension,hkl_dimension,2*hkl_dimension+1)
    H,K,L=np.meshgrid(i,i,i)
    
    # The position of Bragg peaks in reciprocal space
    G1=H*b1[0]+K*b2[0]+L*b3[0]
    G2=H*b1[1]+K*b2[1]+L*b3[1]
    G3=H*b1[2]+K*b2[2]+L*b3[2]
    
    ss=np.size(positions)/4
    ss=int(ss)
    
    # load atomic form factor table
    AF=AFF.AFF()
    
    # calculate the atomic form factor
    ii =np.linspace(0,ss-1,ss)
    ii =ii.astype(int)
    q2=G1*G1+G2*G2+G3*G3
    F= 0
    for j in ii:
        x = np.searchsorted(AF[:,0],positions[j,0])
        fq= 0
        # first formula at http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
        fq=fq+AF[x,1]*np.exp(-AF[x,2]*q2/16/pow(np.pi,2))
        fq=fq+AF[x,3]*np.exp(-AF[x,4]*q2/16/pow(np.pi,2))
        fq=fq+AF[x,5]*np.exp(-AF[x,6]*q2/16/pow(np.pi,2))
        fq=fq+AF[x,7]*np.exp(-AF[x,8]*q2/16/pow(np.pi,2))
        fq=fq+AF[x,9]
        # position of atom in real space cartesian coordinate(angstrom)
        RR=positions[j,1]*aa1+positions[j,2]*aa2+positions[j,3]*aa3
        F=F+fq*np.exp(1j*(G1*RR[0]+G2*RR[1]+G3*RR[2]))
    F=np.abs(F)
    F=pow(F,2)
    BPeaks=np.concatenate((G1,G2,G3,F), axis= 0)

    Mqxy = pow(G1*G1+G2*G2,0.5)
    Mqz = pow(G3*G3,0.5)
    FMiller = F

    return BPeaks, Mqxy, Mqz, FMiller

# -- Description: Loads the POSCAR file from the specified path and returns the simulated Bragg peaks.
def loadPOSCAR(poscar_folder, fileStr, BPeakParams):
    
    theta_x,theta_y, hkl_dimension = BPeakParams # unpack input params

    poscar_path = findfilepath (poscar_folder, fileStr)[0] # build the .vasp path
    print ("Loaded POSCAR file from path: " + poscar_path)

    a1, a2, a3, positions = readPOSCAR(poscar_path) # extract relevant metadata by reading the POSCAR
    BPeaks, Mqxy, Mqz, FMiller = Bragg_Peaks(a1, a2, a3, positions, theta_x, theta_y, hkl_dimension)

    return BPeaks, Mqxy, Mqz, FMiller

# -- Description: Complete function that integrates the above functions to process a CIF simulation and output an intensity map.
def intensitymap (poscar_folder, fileStr, BPeakParams, crystParams, imgParams):
    # unpack variables
    theta_x, theta_y, hkl_dimension = BPeakParams
    sigma_theta, sigma_phi, sigma_r = crystParams
    resolutionx, qxymax, qzmax, qzmin = imgParams

    # map the image space based on input parameters
    resolutionz = int(resolutionx/qxymax*qzmax)
    gridx, gridz = np.meshgrid(np.linspace(-qxymax,qxymax,resolutionx),np.linspace(0,qzmax,resolutionz))
    
    # BPeaks = loadPOSCAR(poscar_folder, fileStr, BPeakParams)
    BPeaks, Mqxy, Mqz, FMiller = loadPOSCAR(poscar_folder, fileStr, BPeakParams)

    t1_start = 0
    t1_stop = 0
    t1_start = time.process_time()

    iMiller = hkl_dimension*2+1
    G1 = BPeaks[0:iMiller,:,:]+np.finfo(float).eps
    G2 = BPeaks[iMiller:2*iMiller,:,:]+np.finfo(float).eps
    G3 = BPeaks[2*iMiller:3*iMiller,:,:]+np.finfo(float).eps
    F = BPeaks[3*iMiller:4*iMiller,:,:]
    
    Eye=np.ones((iMiller,iMiller,iMiller))
    # The positions(r0,theta0,phi0) of Bragg peaks in spherical coordinates.
    theta0=np.pi/2-np.arctan(G3/np.sqrt(pow(G2,2)+pow(G1,2)))
    phi0=np.ones((iMiller,iMiller,iMiller))
    i =np.arange(iMiller)
    for k1 in i:
        for k2 in i:
            for k3 in i:
                if G1[k1,k2,k3]>0:
                    phi0[k1,k2,k3]=np.arcsin(G2[k1,k2,k3]/np.sqrt(pow(G2[k1,k2,k3],2)+pow(G1[k1,k2,k3],2)))
                else:
                    phi0[k1,k2,k3]=np.pi+np.arcsin(G2[k1,k2,k3]/np.sqrt(pow(G2[k1,k2,k3],2)+pow(G1[k1,k2,k3],2)))
                if abs(G2[k1,k2,k3])<0.2:
                    if abs(G1[k1,k2,k3])<0.2:
                        phi0[k1,k2,k3]= 0
                    
    r0=np.sqrt(pow(G1,2)+pow(G2,2)+pow(G3,2))

    # The positions(r,theta,phi) of image plane in spherical coordinates.
    ix,iy=gridx.shape
    I0=np.ones((ix,iy))
    ix=np.arange(ix)
    iy=np.arange(iy)
    for x in ix:
        for y in iy:
            theta=np.pi/2-np.arctan(gridz[x,y]/abs(gridx[x,y]))
            r=np.sqrt(pow(gridx[x,y],2)+pow(gridz[x,y],2))
            if gridx[x,y]>0:
                phi = 0
            else:
                phi =np.pi
            phi =phi*Eye
            phid =abs(phi-phi0)
            phid =abs(abs(phid-np.pi)-np.pi)
            I1=np.exp(-0.5*pow(theta*Eye-theta0,2)/sigma_theta/sigma_theta)
            I2=np.exp(-0.5*phid*phid/sigma_phi/sigma_phi)
            I3=np.exp(-0.5*pow(r*Eye-r0,2)/sigma_r/sigma_r)
            Intensity=I1*I2*I3*F
            I0[x,y]=np.sum(Intensity)
    
    t1_stop = time.process_time()
    print('CPU Time: ')
    print(t1_stop - t1_start,' s')

    return I0, BPeaks, Mqxy, Mqz, FMiller

# -- Description: Plots and formats the simulated scattering from a 
# def plotIntensity(intensity_map, savepath, plotParams):
#     plt.close('all')
#     plt.pause(0.01)
#     gc.collect()

#     os.chdir(savepath)

#     samplename, qxymax, qzmax, xmin, xmax, ymin, ymax, cmin, cmax, cmap, cblabel, cbsize, scaleLog, header, headerfontsize, xfontsize, yfontsize, tickfontsize, autosave, imgdpi, ext, colorscale, fsize, hklcutoff  = plotParams

#     contrastmin = np.percentile(intensity_map, cmin)
#     contrastmax = np.percentile(intensity_map, cmax)
#     extent = -qxymax, qxymax, 0, qzmax

#     if scaleLog == True:
#         intensity_map = np.log(intensity_map + 1)

#     fig, ax = plt.subplots(figsize=fsize)
#     img = ax.imshow(intensity_map,
#                      interpolation='nearest',
#                      vmax=colorscale*intensity_map.max(), vmin=intensity_map.min(),
#                      cmap=cmap,
#                      extent=extent,
#                      origin='lower',
#                      aspect='auto')

#     plt.tick_params(axis='both', which='major', labelsize=tickfontsize) # Image tick parameters
    
#     plt.title(header, fontsize=headerfontsize)
#     plt.xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)', fontsize=xfontsize)
#     plt.xlim(xmin, xmax)

#     plt.ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)', fontsize=yfontsize)
#     plt.ylim(ymin, ymax)

#     if cmin > cmax:
#         cmin = 0
#         cmax = 100
#         print ("Invalid contrast scaling limits, setting cmin = 0, cmax = 100.")
    
#     if cmin < 0:
#         cmin = 0
#         print("Invalid cmin contrast limit, setting cmin = 0.")
    
#     if cmax > 100:
#         cmax = 100
#         print("Invalid cmax contrast limit, setting cmax = 100.")

#     plt.tight_layout()
#     plt.colorbar(img)

#     if autosave == True:
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
#         imgname = samplename + " " + timestamp + ext
#         imgpath = os.path.join(savepath, imgname)
#         plt.savefig(imgpath, dpi=imgdpi) # save the image if desired
#         print("Image Saved: " + str(imgpath))

#     return

def plotIntensity(intensity_map, savepath, plotParams):
    plt.close('all')
    plt.pause(0.01)
    gc.collect()

    os.chdir(savepath)

    samplename, qxymax, qzmax, xmin, xmax, ymin, ymax, cmin, cmax, cmap, cblabel, cbsize, scaleLog, header, headerfontsize, xfontsize, yfontsize, tickfontsize, autosave, imgdpi, ext, colorscale, fsize, hklcutoff  = plotParams

    contrastmin = np.percentile(intensity_map, cmin)
    contrastmax = np.percentile(intensity_map, cmax)
    extent = -qxymax, qxymax, 0, qzmax

    if scaleLog == True:
        intensity_map = np.log(intensity_map + 1)

    fig, ax = plt.subplots(figsize=fsize)
    img = ax.imshow(intensity_map,
                     interpolation='nearest',
                     vmax=colorscale*intensity_map.max(), vmin=intensity_map.min(),
                     cmap=cmap,
                     extent=extent,
                     origin='lower',
                     aspect='auto')

    plt.tick_params(axis='both', which='major', labelsize=tickfontsize) # Image tick parameters
    
    plt.title(header, fontsize=headerfontsize)
    plt.xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)', fontsize=xfontsize)
    plt.xlim(xmin, xmax)

    plt.ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)', fontsize=yfontsize)
    plt.ylim(ymin, ymax)

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

    plt.tight_layout()
    plt.colorbar(img)

    if autosave == True:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        imgname = samplename + " " + timestamp + ext
        imgpath = os.path.join(savepath, imgname)
        plt.savefig(imgpath, dpi=imgdpi) # save the image if desired
        print("Image Saved: " + str(imgpath))

    # save a second version without labels, ticks, borders
    fig2, ax2 = plt.subplots(figsize=fsize)
    img2 = ax2.imshow(intensity_map,
                     interpolation='nearest',
                     vmax=colorscale*intensity_map.max(), vmin=intensity_map.min(),
                     cmap=cmap,
                     extent=extent,
                     origin='lower',
                     aspect='auto')

    ax2.axis('off')  # removes axis

    if autosave == True:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        imgname = samplename + " " + timestamp + "_nolabels" + ext
        imgpath = os.path.join(savepath, imgname)
        plt.savefig(imgpath, dpi=96, bbox_inches='tight', pad_inches=0) # save the image if desired
        print("Image Without Labels Saved: " + str(imgpath))

    return


def plotIntensityGrid(intensity_maps, savepath, plotParams, rotation_sets):
    assert isinstance(savepath, str), f"Expected savepath to be a string, but it was {type(savepath)}"

    plt.close('all')
    plt.pause(0.01)
    gc.collect()
    
    os.chdir(savepath)

    samplename, qxymax, qzmax, xmin, xmax, ymin, ymax, cmin, cmax, cmap, cblabel, cbsize, scaleLog, header, headerfontsize, xfontsize, yfontsize, tickfontsize, autosave, imgdpi, ext, colorscale, fsize, hklcutoff  = plotParams
    
    fig, axs = plt.subplots(2, 3, figsize=fsize)

    extent = -qxymax, qxymax, 0, qzmax

    for i, ax in enumerate(axs.flatten()):
        intensity_map = intensity_maps[i]
        
        if scaleLog == True:
            intensity_map = np.log(intensity_map + 1)
        
        img = ax.imshow(intensity_map,
                         interpolation='nearest',
                         vmax=colorscale*intensity_map.max(), vmin=intensity_map.min(),
                         cmap=cmap,
                         extent=extent,
                         origin='lower',
                         aspect='auto')

        ax.tick_params(axis='both', which='major', labelsize=tickfontsize) # Image tick parameters
        # ax.set_title(f'{header}, Rotation: {rotation_sets[i]}', fontsize=headerfontsize)
        # ax.set_title('f{rotation_sets[i]}', fontsize=headerfontsize)

        rot_x, rot_y = rotation_sets[i]

        # Convert the rotation values to fractions of pi
        fraction_x = fractions.Fraction(rot_x / np.pi).limit_denominator()
        fraction_y = fractions.Fraction(rot_y / np.pi).limit_denominator()

        # Create a function to format the fractions appropriately
        def format_fraction(fraction):
            if fraction.numerator == 0:
                return '0'
            elif fraction.denominator == 1:
                return 'π'
            else:
                return f'{fraction.numerator}π/{fraction.denominator}'

        # Format the fractions as strings
        rot_x_str = format_fraction(fraction_x)
        rot_y_str = format_fraction(fraction_y)

        ax.set_title(f'Rot. X: {rot_x_str}, Rot. Y: {rot_y_str}', fontsize=headerfontsize)

        rot_x, rot_y = rotation_sets[i]
        # ax.set_title(f'Rotation X: {rot_x}, Rotation Y: {rot_y}', fontsize=headerfontsize)
        ax.set_xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)', fontsize=xfontsize)
        ax.set_xlim(xmin, xmax)
        ax.set_ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)', fontsize=yfontsize)
        ax.set_ylim(ymin, ymax)

        fig.colorbar(img, ax=ax)

    plt.tight_layout()
    # plt.title(header, fontsize=headerfontsize)

    if autosave == True:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        imgname = samplename + " " + timestamp + ext
        imgpath = os.path.join(savepath, imgname)
        plt.savefig(imgpath, dpi=imgdpi) # save the image if desired
        print("Image Saved: " + str(imgpath))

    return

# -- Description:  Formats the split map of simulation/real data.
def format_splitmap(intensity_map_ax2, intensity_map_ax1, ax1_qxy, ax1_qz, savepath, plotParams):
    """Show both simulated data and real data loaded from a CSV in parallel. Split the image panel with two axes to do so."""
    plt.close('all')
    plt.pause(0.01)
    gc.collect()

    samplename, qxymax, qzmax, xmin, xmax, ymin, ymax, cmin, cmax, cmap, cblabel, cbsize, scaleLog, header, headerfontsize, xfontsize, yfontsize, tickfontsize, autosave, imgdpi, ext, colorscale, fsize, hklcutoff  = plotParams

    os.chdir(savepath)
    
    # -- Setup Subplot Grid Dimensions
    fig = plt.figure(figsize=(12,8))
    griddim = gridspec.GridSpec(8,8) # Defines image dimensions specifications.
    fig.subplots_adjust(bottom=0.5, left=0.025, top = 1, right=0.975) # Defines subplot image margins
    ax1 = plt.subplot(griddim[1:7, 1:4]) # axis 1 (sim data)
    ax2 = plt.subplot(griddim[1:7, 4:7]) # axis 2 (real data)

    
    # -- Axis (1) Parameters
    ax1_title = 'Data'
    ax1.set_title(ax1_title,  fontsize=headerfontsize) 
    ax1.set_xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)', fontsize=xfontsize)
    ax1.set_xlim(-qxymax, 0)
    ax1.set_ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)', fontsize = yfontsize)
    ax1.set_ylim(0, qzmax)
    ax1.tick_params(axis='both', which='major', labelsize = 12)

    ax1_extent = (np.min(ax1_qxy),np.max(ax1_qxy),np.min(ax1_qz),np.max(ax1_qz))
    ax1_min = np.percentile(intensity_map_ax1, cmin) # Plot q-space detector image (color scaling minimum)
    ax1_max = np.percentile(intensity_map_ax1, cmax) # Plot q-space detector image (color scaling maximum)

    ax1.imshow(intensity_map_ax1, # Create reciprocal space map with qr Ewald sphere correction applied.
                # interpolation='nearest',
                norm=matplotlib.colors.Normalize(vmin=ax1_min,vmax=ax1_max),
                cmap='turbo', # color map formatting
                extent=ax1_extent, # extent defines the visual bounds of the image.
                origin = 'lower',
                aspect='auto')
    
    # -- Axis (2) Parameters
    ax2_title = header
    ax2.set_title(ax2_title,  fontsize=headerfontsize) 
    # ax2.set_xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)', fontsize=xfontsize)
    ax2.set_xlabel(None)
    ax2.set_xlim(0, qxymax)
    # ax2.set_ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)', fontsize = yfontsize)
    ax2.set_ylabel(None)
    ax2.set_ylim(0, qzmax)
    ax2.tick_params(axis='x', which='major', labelsize = 12)
    # ax2.xaxis.set_tick_params(labelbottom=False)
    ax2.yaxis.set_tick_params(labelleft=False)

    if scaleLog == True:
        intensity_map_ax2 = np.log(intensity_map_ax2 + 1)
    
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

    # contrastmin = np.percentile(intensity_map, cmin)
    # contrastmax = np.percentile(intensity_map, cmax)
    # simplot_intensitymap = ax1.imshow(intensity_map,

    ax2_extent = -qxymax, qxymax, 0, qzmax
    ax2.imshow(intensity_map_ax2,
                    #  norm=matplotlib.colors.Normalize(vmin=contrastmin,vmax=contrastmax),
                     interpolation='nearest',
                     vmax=colorscale*intensity_map_ax2.max(), vmin=intensity_map_ax2.min(),
                     cmap='turbo',
                     extent=ax2_extent,
                     origin='lower',
                     aspect='auto')

    
    # -- Plot Parameters
    # plt.title(header, fontsize = headerfontsize)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.colorbar(reciprocalmap, ax = ax1)
    plt.xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)', fontsize=xfontsize)

    if autosave == True:
        timestamp = time.strftime("%Y%m%d_%H%M%S") # Create a time string, with Year, Month, Day + '_' + Hour, Minute, Second. This is appended to the save_path to give the analyzed/reduced data a unique identifier.
        imgname = samplename + '_simplot_splitmap_' + timestamp + ext
        imgpath = os.path.join(savepath, imgname)
        plt.savefig(imgpath, dpi=imgdpi) # save the image if desired
        print("Image Saved: " + str(imgpath))

    return

# -- Description:  Plots the split image of simulation/real data.
def splitmap(datapath, intensity_map, savepath, plotParams):
    # call the CSV loader function for WAXS data
    np_recipmap, np_qxy, np_qz = WAXS.loadcsv_2drecipmap(datapath)
    # call the formatting function
    format_splitmap(intensity_map, np_recipmap, np_qxy, np_qz, savepath, plotParams)

    return

# -- Description:  Find peaks on a 2D reciprocal space map and plots their hkl values.
def peakfinder2d_intensitymap(data, neighborhood_size=5, threshold=0.1, hkllabels=True, qzmax=3, qxymax=3, qzmin=0, colorscale=1):
    
    data_max = ndimage.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)

    data_min = ndimage.minimum_filter(data, neighborhood_size)
    diff1 = ((data_max - data_min) > threshold)
    maxima[diff1 == 0] = 0

    resolutionz,resolutionx=data.shape
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []

    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        y_center = (dy.start + dy.stop - 1)/2

        y_center=resolutionz-y_center
        y_center=qzmax-y_center/resolutionz*(qzmax-qzmin)
        x_center=-qxymax+x_center/resolutionx*qxymax*2

        if abs(x_center)>0.2 and abs(y_center)>0.2:
            x.append(x_center)
            y.append(y_center)

    x = np.array(x)
    y = np.array(y)

    plt.close('all')
    plt.pause(0.01)
    gc.collect() # memory allocation garbage collection

    fig, ax = plt.subplots(figsize=(7,7))

    plt.imshow(data, 
                interpolation='nearest', 
                cmap=cm.turbo,
                origin='lower', 
                extent=[-qxymax, qxymax, 0, qzmax],
                vmax=colorscale*data.max(), vmin=data.min())

    plt.xlabel('q$_{xy}$(1/A)', fontsize=16)
    plt.ylabel('q$_{z}$(1/A)', fontsize=16)
    plt.plot(x,y, 'go')

    # print('peak position')
    if hkllabels == True:
        exp_peak_postions=np.zeros([x.size,3])

        counter=0
        for i in x:
            exp_peak_postions[counter,0]=i
            exp_peak_postions[counter,1]=y[counter]
            exp_peak_postions[counter,2]=np.sqrt(i*i+y[counter]*y[counter])
            counter=counter+1

        column_to_sort_by = 2
        sorted_data = sorted(exp_peak_postions, key=lambda row: row[column_to_sort_by])

        counter=0
        for i in exp_peak_postions:
            print("[","%.3f" % sorted_data[counter][0],",","%.3f" % sorted_data[counter][1],"]",
                  "q=","%.3f" % sorted_data[counter][2])
            counter=counter+1
    return

# -- Description:
def plot_hklindex(intensity_map, savepath, Mqxy, Mqz, FMiller, plotParams, imgParams, BPeakParams):
    """Generates (hkl) Miller indices for an input set of peaks. Takes the input intensity_map in the format of a numpy array 
     with arbitrary dimension (n x m), as we;; as a set of variables that correspond to axes and image formatting parameters.
      Returns a simulated X-ray diffraction image with the specified user input formatting. """

    '''# figure(figsize = (10,8)) # generate figure
    # figure(figsize = fsize) # generate figure
    # colorbar=0.00001
    # contrastmin = np.percentile(intensity_map, cmin)
    # contrastmax = np.percentile(intensity_map, cmax)
    # extent=[-qxymax, qxymax, 0, qzmax],vmax=colorbar*II1.max(), vmin=II1.min() '''
    # -- Plotting Routine
    plt.close('all')
    plt.pause(0.01)
    gc.collect()
    os.chdir(savepath)

    samplename, qxymax, qzmax, xmin, xmax, ymin, ymax, cmin, cmax, cmap, cblabel, cbsize, scaleLog, header, headerfontsize, xfontsize, yfontsize, tickfontsize, autosave, imgdpi, ext, colorscale, fsize, hklcutoff  = plotParams
    resolutionx, qxymax, qzmax, qzmin = imgParams
    hkl_dimension = BPeakParams[2]

    Mindexrange = np.linspace(0, hkl_dimension, hkl_dimension+1)
    Mindexrange = Mindexrange.astype('int')

    simuposi = np.zeros([100,2])
    isimuposi = 0

    fig, ax = plt.subplots(figsize=fsize)
    extent = -qxymax, qxymax, 0, qzmax

    if scaleLog == True:
        intensity_map = np.log(intensity_map + 1)
    
    img = plt.imshow(intensity_map,
                    #  norm=matplotlib.colors.Normalize(vmin=contrastmin,vmax=contrastmax),
                     interpolation='nearest',
                     vmax=colorscale*intensity_map.max(), vmin=intensity_map.min(),
                     cmap='turbo',
                     extent=extent,
                     origin='lower',
                     aspect='auto')

    plt.tick_params(axis='both', which='major', labelsize=tickfontsize) # Image tick parameters
    plt.title(header, fontsize = headerfontsize)
    
    plt.xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)',fontsize=xfontsize)
    plt.xlim(xmin, xmax)

    plt.ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)',fontsize=yfontsize)
    plt.ylim(ymin, ymax)

    plt.tight_layout()
    plt.colorbar(img)

    if autosave == True:
        timestamp = time.strftime("%Y%m%d_%H%M%S") # Create a time string, with Year, Month, Day + '_' + Hour, Minute, Second. This is appended to the save_path to give the analyzed/reduced data a unique identifier.
        imgname = samplename + " " + timestamp + ext
        imgpath = os.path.join(savepath, imgname)
        plt.savefig(imgpath, dpi=imgdpi) # save the image if desired
        print("Image Saved: " + str(imgpath))

    # -- Generate the (h k l) index labels.
    MaxI = 0
    for h in Mindexrange:
        for k in Mindexrange:
            for l in Mindexrange:
                if Mqxy[h,k,l]<qxymax and Mqz[h,k,l]>qzmin and Mqz[h,k,l]<qzmax:
                    MaxI = np.maximum(FMiller[h,k,l], MaxI)
                    
    for h in Mindexrange:
        for k in Mindexrange:
            for l in Mindexrange:
                if Mqxy[h,k,l]<qxymax and Mqz[h,k,l]>qzmin and Mqz[h,k,l]<qzmax:
                    if FMiller[h,k,l] > hklcutoff*MaxI:
                        plt.plot(Mqxy[h,k,l], Mqz[h,k,l], 'ko')
                        simuposi[isimuposi,0]=Mqxy[h,k,l]
                        simuposi[isimuposi,1]=Mqz[h,k,l]
                        isimuposi=isimuposi+1
                        textstr='('+str(h-hkl_dimension)+','+str(l-hkl_dimension)+','+str(-k+hkl_dimension)+')'
                        plt.text(Mqxy[h,k,l]/(2*qxymax)+0.5, (Mqz[h,k,l]-qzmin)/(qzmax-qzmin), textstr, 
                                 transform=ax.transAxes, fontsize=10,verticalalignment='top',color='k')

    return simuposi

def intensitymaptool(poscar_folder, fileStr, BPeakParams, crystParams, imgParams):
    # Unpack variables from input dictionaries
    theta_x, theta_y, hkl_dimension = BPeakParams["theta_x"], BPeakParams["theta_y"], BPeakParams["hkl_dimension"]
    sigma_theta, sigma_phi, sigma_r = crystParams["sigma_theta"], crystParams["sigma_phi"], crystParams["sigma_r"]
    resolutionx, qxymax, qzmax, qzmin = imgParams["resolutionx"], imgParams["qxymax"], imgParams["qzmax"], imgParams["qzmin"]

    # Map the image space based on input parameters
    resolutionz = int(resolutionx/qxymax*qzmax)
    gridx, gridz = np.meshgrid(np.linspace(-qxymax,qxymax,resolutionx),np.linspace(0,qzmax,resolutionz))
    
    print ("Please ensure that POSCAR (.vasp) is exported from CIF in 'Fractional Coordinates'.")
    # Load Bragg peaks, reciprocal space map coordinates and Miller indices from POSCAR file
    BPeaks, Mqxy, Mqz, FMiller = loadPOSCAR(poscar_folder, fileStr, (theta_x, theta_y, hkl_dimension))

    # Record process start time
    t1_start = time.process_time()

    iMiller = hkl_dimension*2+1
    G1 = BPeaks[0:iMiller,:,:]+np.finfo(float).eps
    G2 = BPeaks[iMiller:2*iMiller,:,:]+np.finfo(float).eps
    G3 = BPeaks[2*iMiller:3*iMiller,:,:]+np.finfo(float).eps
    F = BPeaks[3*iMiller:4*iMiller,:,:]

    # Calculate the positions of Bragg peaks in spherical coordinates
    theta0 = np.pi/2 - np.arctan(G3/np.sqrt(G2**2 + G1**2))
    phi0 = np.where(G1 > 0, np.arcsin(G2/np.sqrt(G2**2 + G1**2)), np.pi + np.arcsin(G2/np.sqrt(G2**2 + G1**2)))
    phi0[(abs(G2) < 0.2) & (abs(G1) < 0.2)] = 0
    r0 = np.sqrt(G1**2 + G2**2 + G3**2)

    # Initialize the intensity map
    I0 = np.zeros_like(gridx)

    # Calculate the intensity map for each point in the image plane
    theta, r, phi = np.pi/2 - np.arctan(gridz/np.abs(gridx)), np.sqrt(gridx**2 + gridz**2), (gridx <= 0) * np.pi
    I0 = np.sum(np.exp(-0.5 * ((theta - theta0[..., np.newaxis, np.newaxis])**2 / sigma_theta**2 + 
                               ((phi - phi0[..., np.newaxis, np.newaxis]) % (2 * np.pi) - np.pi)**2 / sigma_phi**2 + 
                               (r - r0[..., np.newaxis, np.newaxis])**2 / sigma_r**2)) * F[..., np.newaxis, np.newaxis], axis=(0, 1, 2))

    # Record process end time and print elapsed time
    t1_stop = time.process_time()
    print('CPU Time: ')
    print(t1_stop - t1_start,' s')

    return I0, BPeaks, Mqxy, Mqz, FMiller

# ------ Unimplemented Functions
# -- Description (ZZ):
def intensity_diffuse(gridx,gridz,BPeaks,sigma_theta,sigma_phi,sigma_r,hkl_dimension):
    iMiller=hkl_dimension*2+1
    idiffuse=2
    G1=BPeaks[0:iMiller,:,:]+np.finfo(float).eps
    G2=BPeaks[iMiller:2*iMiller,:,:]+np.finfo(float).eps
    G3=BPeaks[2*iMiller:3*iMiller,:,:]+np.finfo(float).eps
    F=BPeaks[3*iMiller:4*iMiller,:,:]
    
    Eye=np.ones([iMiller,2,iMiller])

    # The positions(r0,theta0,phi0) of Bragg peaks in spherical coordinates.
    theta0=np.pi/2-np.arctan(G3/np.sqrt(pow(G2,2)+pow(G1,2)))
    phi0=np.ones((iMiller,2,iMiller))
    i =np.arange(iMiller)
    for k1 in i:
        for k2 in np.array([0,1]):
            for k3 in i:
                if G1[k1,k2,k3]>0:
                    phi0[k1,k2,k3]=np.arcsin(G2[k1,k2,k3]/np.sqrt(pow(G2[k1,k2,k3],2)+pow(G1[k1,k2,k3],2)))
                else:
                    phi0[k1,k2,k3]=np.pi+np.arcsin(G2[k1,k2,k3]/np.sqrt(pow(G2[k1,k2,k3],2)+pow(G1[k1,k2,k3],2)))
                if abs(G2[k1,k2,k3])<0.2:
                    if abs(G1[k1,k2,k3])<0.2:
                        phi0[k1,k2,k3]= 0
                    
    r0=np.sqrt(pow(G1,2)+pow(G2,2)+pow(G3,2))

    # The positions(r,theta,phi) of image plane in spherical coordinates.
    ix,iy=gridx.shape
    I0=np.ones((ix,iy))
    ix=np.arange(ix)
    iy=np.arange(iy)
    for x in ix:
        for y in iy:
            theta=np.pi/2-np.arctan(gridz[x,y]/abs(gridx[x,y]))
            r=np.sqrt(pow(gridx[x,y],2)+pow(gridz[x,y],2))
            if gridx[x,y]>0:
                phi = 0
            else:
                phi =np.pi
            phi =phi*Eye
            phid =abs(phi-phi0)
            phid =abs(abs(phid-np.pi)-np.pi)
            I1=np.exp(-0.5*pow(theta*Eye-theta0,2)/sigma_theta/sigma_theta)
            I2=np.exp(-0.5*phid*phid/sigma_phi/sigma_phi)
            I3=np.exp(-0.5*pow(r*Eye-r0,2)/sigma_r/sigma_r)
            Intensity=I1*I2*I3*F
            I0[x,y]=np.sum(Intensity)
    return I0

# -- Description (ZZ):
def diffuse(a1,a2,a3,positions,theta_x,theta_y,hkl_dimension,shift):
    # Lattice parameters M matrix in cartesian coordinate(angstrom)
    M=[a1,a2,a3]
    M=np.asarray(M)
    # Rotation Matrix respect to X axis, rotation angle = theta_x
    Rx=np.array([[1,0,0],[0,np.cos(theta_x),-np.sin(theta_x)],[0,np.sin(theta_x),np.cos(theta_x)]])
    # Rotation Matrix respect to Y axis, rotation angle = theta_y
    Ry=np.array([[np.cos(theta_y),0,-np.sin(theta_y)],[0,1,0],[np.sin(theta_y),0,np.cos(theta_y)]])

    # Rotation of the sample
    M=np.matmul(M, Rx)
    M=np.matmul(M, Ry)
    
    # New lattice parameter
    aa1=M[0,:]
    aa2=M[1,:]
    aa3=M[2,:]

    # reciprocal lattice
    volume=np.matmul(aa3,np.cross(aa1,aa2))
    b1=2*np.pi*np.cross(aa2,aa3)/volume
    b2=2*np.pi*np.cross(aa3,aa1)/volume
    b3=2*np.pi*np.cross(aa1,aa2)/volume

    # grid for Miller index
    i =np.linspace(-hkl_dimension,hkl_dimension,2*hkl_dimension+1)
    H,K,L=np.meshgrid(np.array([0,1]),i,i)

    
    # The position of Bragg peaks in reciprocal space
    G1=H*b1[0]+K*b2[0]+L*b3[0]
    G2=H*b1[1]+K*b2[1]+L*b3[1]
    G3=H*b1[2]+K*b2[2]+L*b3[2]
    print('Gshape')
    print(G1.shape)
    print(G2.shape)
    print(G3.shape)
    
    ss=np.size(positions)/4
    ss=int(ss)
    
    # load atomic form factor table
    AF=AFF.AFF()
    
    # calculate the atomic form factor
    ii =np.linspace(0,ss-1,ss)
    ii =ii.astype(int)
    q2=G1*G1+G2*G2+G3*G3
    F= 0
    for j in ii:
        x = np.searchsorted(AF[:,0],positions[j,0])
        fq= 0
        # first formula at http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
        fq=fq+AF[x,1]*np.exp(-AF[x,2]*q2/16/pow(np.pi,2))
        fq=fq+AF[x,3]*np.exp(-AF[x,4]*q2/16/pow(np.pi,2))
        fq=fq+AF[x,5]*np.exp(-AF[x,6]*q2/16/pow(np.pi,2))
        fq=fq+AF[x,7]*np.exp(-AF[x,8]*q2/16/pow(np.pi,2))
        fq=fq+AF[x,9]
        # position of atom in real space cartesian coordinate(angstrom)
        RR=positions[j,1]*aa1+positions[j,2]*aa2+positions[j,3]*aa3
        F=F+fq*np.exp(1j*(G1*RR[0]+G2*RR[1]+G3*RR[2]))
    F=np.abs(F)
    F=pow(F,2)
    BPeaks=np.concatenate((G1,G2-shift,G3,F), axis= 0)
    return BPeaks

# -- Description (ZZ):
def GIWAXS_peak_finder(data,neighborhood_size,threshold,print_peak_position,qzmax,qxymax,qzmin,colorbar):
    data_max = ndimage.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = ndimage.minimum_filter(data, neighborhood_size)
    diff1 = ((data_max - data_min) > threshold)
    maxima[diff1 == 0] = 0
    resolutionz,resolutionx=data.shape
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        y_center = (dy.start + dy.stop - 1)/2

        y_center=resolutionz-y_center
        y_center=qzmax-y_center/resolutionz*(qzmax-qzmin)
        x_center=-qxymax+x_center/resolutionx*qxymax*2

        if abs(x_center)>0.2 and abs(y_center)>0.2:
            x.append(x_center)
            y.append(y_center)



    x=np.array(x)
    y=np.array(y)

    fig,ax=plt.subplots(figsize=(7,7))

    plt.imshow(data, interpolation='nearest', cmap=cm.jet,
                   origin='lower', extent=[-qxymax, qxymax, 0, qzmax],
                   vmax=colorbar*data.max(), vmin=data.min())

    plt.xlabel('q$_{xy}$(1/A)',fontsize=16)
    plt.ylabel('q$_{z}$(1/A)',fontsize=16)
    plt.plot(x,y, 'go')

    # print('peak position')
    if print_peak_position==True:
        exp_peak_postions=np.zeros([x.size,3])
        counter=0
        for i in x:
            exp_peak_postions[counter,0]=i
            exp_peak_postions[counter,1]=y[counter]
            exp_peak_postions[counter,2]=np.sqrt(i*i+y[counter]*y[counter])
            counter=counter+1

        column_to_sort_by = 2
        sorted_data = sorted(exp_peak_postions, key=lambda row: row[column_to_sort_by])

        counter=0
        for i in exp_peak_postions:
            print("[","%.3f" % sorted_data[counter][0],",","%.3f" % sorted_data[counter][1],"]",
                  "q=","%.3f" % sorted_data[counter][2])
            counter=counter+1
    return

# -- Description: (Unwritten)
def chiint1d_simplot():
    """Integrate the simulated 2D intensity map."""
    return

# -- Description: (Unwritten)
def format_chiint1d_simplot():
    """Format 'matplotlib' plots for 1d integrations of simulated waxs data."""
    return

'''
# ################################
# import numpy as np
# import scipy.fftpack as fftpack
# import scipy.ndimage as ndimage
# import matplotlib.pyplot as plt
# import gc
# import time
# from collections import defaultdict
# from scipy.optimize import minimize

# # %matplotlib widget

# def generate_gaussian_heatmap(height, width, num_gaussians, gaussian_std, amplitude_range):
#     heatmap = np.zeros((height, width))
#     true_peak_centers = []

#     for _ in range(num_gaussians):
#         amplitude = np.random.uniform(amplitude_range[0], amplitude_range[1])
#         center_y, center_x = np.random.randint(0, height), np.random.randint(0, width)
#         true_peak_centers.append((center_y, center_x))
#         y, x = np.mgrid[0:height, 0:width]
#         heatmap += amplitude * np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * gaussian_std ** 2))

#     return heatmap, np.array(true_peak_centers)

# def monte_carlo_peak_finding(heatmap, m, n, gradient_threshold):
#     height, width = heatmap.shape
#     visited = np.zeros_like(heatmap, dtype=bool)

#     peak_centers = []

#     for _ in range(m):
#         start_point = np.random.randint(0, height), np.random.randint(0, width)

#         for _ in range(n):
#             y, x = start_point
#             neighborhood = heatmap[max(y - 1, 0):min(y + 2, height), max(x - 1, 0):min(x + 2, width)]
#             next_point = np.unravel_index(np.argmax(neighborhood), neighborhood.shape)
#             next_point = next_point[0] + max(y - 1, 0), next_point[1] + max(x - 1, 0)

#             if next_point == start_point:
#                 break
#             else:
#                 start_point = next_point

#         y, x = start_point
#         if not visited[y, x]:
#             peak_centers.append(start_point)

#             # Turn off the region around the peak center
#             grad_y, grad_x = np.gradient(heatmap)
#             gradient_magnitude = np.sqrt(grad_y ** 2 + grad_x ** 2)
#             mask = gradient_magnitude < gradient_threshold
#             visited[mask] = True

#     return np.array(peak_centers)

# def fourier_filter_and_find_peaks(heatmap, cutoff_frequency, m, n, gradient_threshold):
#     # Apply the Fourier transform to the heatmap
#     fft_heatmap = fftpack.fft2(heatmap)

#     # Create a low-pass filter in the frequency domain
#     rows, cols = heatmap.shape
#     crow, ccol = int(rows / 2), int(cols / 2)
#     low_pass_filter = np.zeros((rows, cols))
#     low_pass_filter[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 1

#     # Apply the low-pass filter to the heatmap in the frequency domain
#     filtered_fft_heatmap = fft_heatmap * low_pass_filter

#     # Transform the filtered heatmap back to the spatial domain
#     filtered_heatmap = np.real(fftpack.ifft2(filtered_fft_heatmap))

#     # Run the Monte Carlo peak finding algorithm on the filtered heatmap
#     peak_centers = monte_carlo_peak_finding(filtered_heatmap, m, n, gradient_threshold)

#     return peak_centers

# def looped_monte_carlo_peak_finding(heatmap, cutoff_frequency, m, n, gradient_threshold, num_iterations, reproducibility_threshold, edge_removal):
#     gc.collect()

#     def distance(p1, p2):
#         return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

#     peaks_count = defaultdict(int)
#     all_peaks = []

#     for _ in range(num_iterations):
#         np.random.seed(int(time.time() * 1e6) % 2**32)  # Set seed based on the current time
#         peak_centers = fourier_filter_and_find_peaks(heatmap, cutoff_frequency, m, n, gradient_threshold)

#         for peak in peak_centers:
#             peak = tuple(peak)  # Convert numpy array to tuple
#             all_peaks.append(peak)
#             for other_peak in all_peaks:
#                 if distance(peak, other_peak) <= reproducibility_threshold:
#                     peaks_count[other_peak] += 1
#                     break

#     reproducible_peaks = [peak for peak, count in peaks_count.items() if count >= num_iterations / 2]

#     # Remove peaks at image edges
#     reproducible_peaks = [(x, y) for x, y in reproducible_peaks if edge_removal <= x < heatmap.shape[0] - edge_removal and edge_removal <= y < heatmap.shape[1] - edge_removal]

#     return np.array(reproducible_peaks)

# def plot_heatmap_and_peaks(heatmap, peak_centers):
#     plt.figure()
#     plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    
#     if peak_centers.size > 0:
#         plt.scatter(peak_centers[:, 1], peak_centers[:, 0], c='blue', marker='x', s=50)
    
#     plt.colorbar()
#     plt.show()

# def peak_finding_score(true_peak_centers, detected_peak_centers, tolerance):
#     true_peak_centers = set(tuple(p) for p in true_peak_centers)
#     detected_peak_centers = set(tuple(p) for p in detected_peak_centers)
#     score = 0

#     for true_peak in true_peak_centers:
#         if not any(np.linalg.norm(np.array(true_peak) - np.array(detected_peak)) <= tolerance for detected_peak in detected_peak_centers):
#             score += 1
    
#     for detected_peak in detected_peak_centers:
#         if not any(np.linalg.norm(np.array(true_peak) - np.array(detected_peak)) <= tolerance for true_peak in true_peak_centers):
#             score += 1

#     return score

# def optimize_parameters(heatmap, true_peak_centers, initial_parameters, bounds):
#     def objective_function(parameters):
#         cutoff_frequency, m, n, gradient_threshold, num_iterations, reproducibility_threshold, edge_removal = parameters
#         m, n, num_iterations, edge_removal = int(m), int(n), int(num_iterations), int(edge_removal)
#         detected_peak_centers = looped_monte_carlo_peak_finding(heatmap, cutoff_frequency, m, n, gradient_threshold, num_iterations, reproducibility_threshold, edge_removal)
#         score = peak_finding_score(true_peak_centers, detected_peak_centers, tolerance=5)
#         return score

#     result = minimize(objective_function, initial_parameters, bounds=bounds, method='L-BFGS-B')
#     return result.x
'''