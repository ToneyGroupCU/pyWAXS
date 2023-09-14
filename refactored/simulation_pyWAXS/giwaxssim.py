#!keithwhite@Keiths-MacBook-Pro/opt/anaconda3/envs/pyWAXS
# -*- coding: utf-8 -*-
#
#    Project: GIWAXS Simulation Package
#
#    (2022 - 2023), University of Colorado Boulder, Boulder, CO 80305
#
#    Principal author(s): Zihan Zhang (Zihan.Zhang-1@colorado.edu), Keith White (Keith.White@colorado.edu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

"""GIWAXS Simulation

A module used to simulate 2D GIWAXS data, and fit known phases to existing GIWAXS data."""

__author__ = "Zihan Zhang, Keith White"
__contact__ = "keith.white@colorado.edu"
__date__ = "24/04/2023"
__status__ = "production"

# -- Standard Packages
import os, re, sys, gc, time, dask
import numpy as np
import scipy as sp
import glob2 as glob
import pandas as pd

'''
# import logger
# matplotlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import subplots
from matplotlib import cm
from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.pyplot import figure
from matplotlib.gridspec import GridSpec
'''

# -- Specialized Packages
import pydabax

# -- Custom Modules
import AFF
import bokehplotting

# logger = logging.getLogger(__name__)

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

class WAXSSim:
    def __init__(self, poscar_address=None):
        # -- Positional Class Attributes
        self.a = None
        self.b = None
        self.c = None
        self.alpha = None
        self.beta = None
        self.gamma = None

        # -- Simulation Class Attributes
        # Crystallite Centeral Position
        self.theta_x = 0
        self.theta_y = 0

        # Crystallite Orientation Parameters
        self.sigma_theta = 0.02
        self.sigma_phi = 100
        self.sigma_r = 0.01

        # Image Specific Parameters
        self.hkl_dimension = 7
        self.resolutionx = 300
        self.qxymax = 2
        self.qzmax = 2

        # POSCAR Address
        self.poscar_address = poscar_address
        self._positions = None
        self.position = None

        # Simulation Data
        self.BPeaks = None
        self.Mqxy = None
        self.Mqz = None
        self.FMiller = None
        self.I0 = None

        if poscar_address is not None:
            self.get_poscar(poscar_address)

    def get_poscar(self, poscar_address):
        self.clear_Bragg_Peaks() # clear Bragg peaks when loading new poscar address       
        self.poscar_address = poscar_address
        self.a, self.b, self.c, self.position = self.read_poscar()

    def read_poscar(self):
        address = self.poscar_address

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
                p=x.astype(float)
                position[ind-9,1]=p[0]
                position[ind-9,2]=p[1]
                position[ind-9,3]=p[2]
        file.close()

        ind = 0
        iii = 0
        for ii in z1:
            position[iii:iii+ii+1,0]=z[ind]
            iii = iii+ii
            ind = ind+1

        print('POSCAR file has been read into class object -WAXSSim-.')
        return a1, a2, a3, position

    def sim_init(self, theta_x=None, theta_y=None, sigma_theta=None, sigma_phi=None, sigma_r=None, hkl_dimension=None, resolutionx=None, qxymax=None, qzmax=None):
        if theta_x is not None:
            self.theta_x = theta_x
        if theta_y is not None:
            self.theta_y = theta_y
        if sigma_theta is not None:
            self.sigma_theta = sigma_theta
        if sigma_phi is not None:
            self.sigma_phi = sigma_phi
        if sigma_r is not None:
            self.sigma_r = sigma_r
        if hkl_dimension is not None:
            self.hkl_dimension = hkl_dimension
        if resolutionx is not None:
            self.resolutionx = resolutionx
        if qxymax is not None:
            self.hkl_dimension = qxymax
        if qzmax is not None:
            self.resolutionx = qzmax
        print('Parameters initialized.')

    def Bragg_Peaks(self):
        """ Function Description: Calculates the position of Bragg peaks in reciprocal space 
        using lattice parameters and position of atoms read from the POSCAR file. Two rotation 
        angles are added with respect to x and y axes to adjust the orientation of the single crystal.
        """
        if self.a is None or self.b is None or self.c is None or self.position is None:
            print("Numerical values must be assigned to a, b, c, and position before executing Bragg_Peaks().")
            return
    
        a1 = self.a
        a2 = self.b
        a3 = self.c
        positions = self.position
        theta_x = self.theta_x
        theta_y = self.theta_y
        hkl_dimension = self.hkl_dimension

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
        
        self.BPeaks = BPeaks
        self.Mqxy = Mqxy
        self.Mqz = Mqz
        self.FMiller = FMiller

    def clear_Bragg_Peaks(self):
        self.a = None
        self.b = None
        self.c = None
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.BPeaks = None
        self.Mqxy = None
        self.Mqz = None
        self.FMiller = None
        self.I0 = None

        gc.collect()
    
    def intensity_map(self):
        # unpack variables
        hkl_dimension = self.hkl_dimension
        sigma_theta = self.sigma_theta
        sigma_phi = self.sigma_phi
        sigma_r = self.sigma_r
        resolutionx = self.resolutionx
        qxymax = self.qxymax
        qzmax = self.qzmax
        BPeaks = self.BPeaks

        # map the image space based on input parameters
        resolutionz = int(resolutionx/qxymax*qzmax)
        gridx, gridz = np.meshgrid(np.linspace(-qxymax,qxymax,resolutionx),np.linspace(0,qzmax,resolutionz))

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

        self.I0 = I0