# from numba import jit, njit, prange
import dask.array as da
from dask import delayed, compute
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import matplotlib.cm as cm
import pandas as pd
from pathlib import Path
from typing import Union
import os
from contextlib import redirect_stdout


# Custom Imports
import WAXSAFF

class WAXSSim:
    def __init__(self, 
                #  *file_paths: Union[Path, str], 
                 poscarPath: Union[Path, str],
                 simPath: Union[Path, str], 
                 sigma_theta = 0.01, 
                 sigma_phi = 100, 
                 sigma_r = 0.01, 
                 hkl_dimension = 10, 
                 thetaX = np.pi/2, 
                 thetaY = 0,
                 resX = 256,
                 qxyRange = [-3, 3],
                 qzRange = [0, 3], 
                 projectname = None):
        
        self.sigma_theta = sigma_theta # smearing about the theta coordinate
        self.sigma_phi = sigma_phi # smearing about the phi coordinate
        self.sigma_r = sigma_r # smearing about the r coordinate
        self.hkl_dimension = hkl_dimension # extent of (hkl) to calculate 
        self.thetaX = thetaX # initial rotation about x-axis
        self.thetaY = thetaY # initial rotation about y-axis
        
        self.poscarPath = poscarPath # pathlib or string path variable for poscar.
        self.simPath = simPath
        self.projectname = projectname
        
        self.a1, self.a2, self.a3, self.position = self.readPOSCAR(address = poscarPath)

        self.M = [self.a1, self.a2, self.a3] # lattice parameter matrix
        self.M = np.asarray(self.M)
        self.Rx = None # rotation matrix X
        self.Ry = None # romationmatrix Y

        self.volume = None
        self.a1_rotated = None
        self.a2_rotated = None
        self.a3_rotated = None

        self.b1, self.b2, self.b3 = None, None, None
        self.G1, self.G2, self.G3 = None, None, None
        self.FMiller = None
        self.BPeaks, self.M_qxy, self.M_qz, self.FMiller = None, None, None, None

        self.iMiller = None
        self.theta0, self.phi0, self.r0 = None, None, None
        self.centroids = None

        self.qxyRange = qxyRange
        self.qzRange = qzRange
        self.resX = resX

        self.qxyMin = np.min(qxyRange)
        self.qxyMax = np.max(qxyRange)
        self.qzMin = np.min(qzRange)
        self.qzMax = np.max(qzRange)

        self.resZ = int(self.resX/self.qxyMax * self.qzMax)
        self.gridx, self.gridz = np.meshgrid(np.linspace(self.qxyMin, self.qxyMax, self.resX), np.linspace(self.qzMin, self.qzMax, self.resZ))

        self.intensityMap = None

        # Define a DataFrame to store the required information
        # columns = ['path', 'a1', 'a2', 'a3', 'positions', 'bragg_peaks', 'other_bragg_info']
        # self.diffsim_df = pd.DataFrame(columns=columns) # Changed to 'diffsim_df'

    def readPOSCAR(self, address: Union[Path, str]):
        ''' READ POSCAR FILE: Read-in the POSCAR input file.'''
        
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
        file.close()

        ind = 0
        iii = 0
        for ii in z1:
            position[iii:iii+ii+1,0]=z[ind]
            iii=iii+ii
            ind=ind+1
        
        # Storing values in the DataFrame
        # self.data.loc[len(self.data)] = [address, a1, a2, a3, position, None, None]
        # self.diffsim_df.loc[len(self.diffsim_df)] = [address, a1, a2, a3, position, None, None]

        return a1, a2, a3, position

    def BraggPeaks(self):
        """
        # -- BRAGG PEAK CALCULATION: Calculate the position of the Bragg Peaks.
        Description: BraggPeaks function calculate the position of Bragg peaks in reciprocal space using lattice parameters and position of atoms read from the POSCAR file.
        Two rotation angles respect to x and y axis are added to adjust the orientation of the single crystal.
        """

        # Rotation Matrix respect to X axis, rotation angle = thetaX
        self.Rx = np.array([[1,0,0],[0,np.cos(self.thetaX),-np.sin(self.thetaX)],[0,np.sin(self.thetaX),np.cos(self.thetaX)]])

        # Rotation Matrix respect to Y axis, rotation angle = thetaY
        self.Ry = np.array([[np.cos(self.thetaY),0,-np.sin(self.thetaY)],[0,1,0],[np.sin(self.thetaY),0,np.cos(self.thetaY)]])

        # Rotation of the sample
        self.M_rotated = np.matmul(self.M, self.Rx) # rotation w/ Rx matrix.
        self.M_rotated = np.matmul(self.M, self.Ry) # rotation w/ Ry matrix.
        
        # New lattice parameter
        self.a1_rotated = self.M_rotated[0,:]
        self.a2_rotated = self.M_rotated[1,:]
        self.a3_rotated = self.M_rotated[2,:]

        # Reciprocal lattice vectors and volume calculation
        self.volume = np.matmul(self.a3_rotated, np.cross(self.a1_rotated, self.a2_rotated))
        self.b1 = 2*np.pi*np.cross(self.a2_rotated,self.a3_rotated)/self.volume
        self.b2 = 2*np.pi*np.cross(self.a3_rotated,self.a1_rotated)/self.volume
        self.b3 = 2*np.pi*np.cross(self.a1_rotated,self.a2_rotated)/self.volume


        # Generate grid for Miller indices
        i = np.linspace(-self.hkl_dimension,self.hkl_dimension, 2*self.hkl_dimension + 1)
        H,K,L = np.meshgrid(i,i,i)
        
        # The position of Bragg peaks in reciprocal space
        self.G1 = H*self.b1[0] + K*self.b2[0] + L*self.b3[0]
        self.G2 = H*self.b1[1] + K*self.b2[1] + L*self.b3[1]
        self.G3 = H*self.b1[2] + K*self.b2[2] + L*self.b3[2]

        ss = np.size(self.position)/4
        ss = int(ss)
        
        # Load atomic form factor table
        AF = WAXSAFF.AFF()

        ''' # -- dask parallization
            # Generate grid for Miller indices
            i = da.linspace(-self.hkl_dimension, self.hkl_dimension, 2 * self.hkl_dimension + 1)
            H, K, L = da.meshgrid(i, i, i)
            
            # The position of Bragg peaks in reciprocal space
            self.G1 = H * self.b1[0] + K * self.b2[0] + L * self.b3[0]
            self.G2 = H * self.b1[1] + K * self.b2[1] + L * self.b3[1]
            self.G3 = H * self.b1[2] + K * self.b2[2] + L * self.b3[2]

            ss = np.size(self.position)/4
            ss = int(ss)
            
            # Load atomic form factor table
            AF = WAXSAFF.AFF()

            ii = np.linspace(0, ss - 1, ss).astype(int)
            ii_dask = da.from_array(ii, chunks=(100,))  # Chunking for parallelization. Adjust chunk size based on your system.

            self.FMiller = da.map_blocks(self.compute_fq_block, 
                                        ii_dask, 
                                        AF=AF, 
                                        q2=self.G1 ** 2 + self.G2 ** 2 + self.G3 ** 2,
                                        a1_rotated=self.a1_rotated, 
                                        a2_rotated=self.a2_rotated, 
                                        a3_rotated=self.a3_rotated, 
                                        G1=self.G1, 
                                        G2=self.G2, 
                                        G3=self.G3, 
                                        dtype=ii_dask.dtype)
        '''

        # Calculate the atomic form factor
        ii = np.linspace(0, ss - 1, ss)
        ii = ii.astype(int)

        q2 = self.G1 * self.G1 + self.G2 * self.G2 + self.G3 * self.G3

        self.F = 0
        for j in ii:
            x = np.searchsorted(AF[:,0],self.position[j,0])
            fq = 0
            # first formula at http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
            fq = fq + AF[x,1] * np.exp(-AF[x,2] * q2/16/pow(np.pi,2))
            fq = fq + AF[x,3] * np.exp(-AF[x,4] * q2/16/pow(np.pi,2))
            fq = fq + AF[x,5] * np.exp(-AF[x,6] * q2/16/pow(np.pi,2))
            fq = fq + AF[x,7] * np.exp(-AF[x,8] * q2/16/pow(np.pi,2))
            fq = fq + AF[x,9]
            
            # position of atom in real space cartesian coordinate(angstrom)
            RR = self.position[j,1]*self.a1_rotated + self.position[j,2] * self.a2_rotated + self.position[j,3] * self.a3_rotated
            self.F = self.F + fq*np.exp(1j*(self.G1 * RR[0] + self.G2 * RR[1] + self.G3 * RR[2]))
        
        self.F = np.abs(self.F) # structure factor F
        self.FMiller = pow(self.F,2)

        self.Bpeaks = np.concatenate((self.G1,self.G2, self.G3, self.FMiller), axis=0)
        self.M_qxy = pow(self.G1 * self.G1 + self.G2 * self.G2, 0.5)
        self.M_qz = pow(self.G3 * self.G3,0.5), self.FMiller

        # If needed, convert Dask arrays back to NumPy arrays
        # self.Bpeaks, self.M_qxy, self.M_qz, self.FMiller = da.compute(self.Bpeaks, self.M_qxy, self.M_qz, self.FMiller)
        # return self.M_qxy, self.M_qz, self.FMiller
        
        return self.Bpeaks, self.M_qxy, self.M_qz, self.FMiller

    def Intensity(self):
        # INTENSITY CALCULATIONS: Calculate the intensity of the Bragg Peaks from the atomic form factor.
        
        self.iMiller = self.hkl_dimension*2 + 1

        self.G1 = self.Bpeaks[0:self.iMiller,:,:] + np.finfo(float).eps
        self.G2 = self.Bpeaks[self.iMiller:2*self.iMiller,: ,:] + np.finfo(float).eps
        self.G3 = self.Bpeaks[2*self.iMiller:3*self.iMiller,: ,:] + np.finfo(float).eps
        self.FMiller = self.Bpeaks[3*self.iMiller:4*self.iMiller,: ,:]
        
        self.centroids = np.ones((self.iMiller,self.iMiller,self.iMiller))
        
        # The positions(r0,theta0,phi0) of Bragg peaks in spherical coordinates.
        self.theta0 = np.pi/2-np.arctan(self.G3/np.sqrt(pow(self.G2,2)+pow(self.G1,2)))
        self.phi0 = np.ones((self.iMiller, self.iMiller, self.iMiller))
        
        i = np.arange(self.iMiller)

        for k1 in i:
            for k2 in i:
                for k3 in i:
                    if self.G1[k1,k2,k3]>0:
                        self.phi0[k1,k2,k3] = np.arcsin(self.G2[k1,k2,k3]/np.sqrt(pow(self.G2[k1,k2,k3],2) + pow(self.G1[k1,k2,k3],2)))
                    else:
                        self.phi0[k1,k2,k3] = np.pi + np.arcsin(self.G2[k1,k2,k3]/np.sqrt(pow(self.G2[k1,k2,k3],2) + pow(self.G1[k1,k2,k3],2)))
                    if abs(self.G2[k1,k2,k3])<0.2:
                        if abs(self.G1[k1,k2,k3])<0.2:
                            self.phi0[k1,k2,k3]=0
                        
        self.r0 = np.sqrt(pow(self.G1,2)+pow(self.G2,2)+pow(self.G3,2))

        # The positions(r,theta,phi) of image plane in spherical coordinates.
        ix, iy = self.gridx.shape
        self.intensityMap = np.ones((ix,iy))
        ix = np.arange(ix)
        iy = np.arange(iy)

        for x in ix:
            for y in iy:
                theta = np.pi/2-np.arctan(self.gridz[x,y]/abs(self.gridx[x,y]))
                r = np.sqrt(pow(self.gridx[x,y],2) + pow(self.gridz[x,y],2))
                
                if self.gridx[x,y] > 0:
                    phi = 0
                else:
                    phi = np.pi

                phi = phi * self.centroids
                phid = abs(phi - self.phi0)
                phid = abs(abs(phid - np.pi) - np.pi)
                
                I1 = np.exp(-0.5 * pow(theta * self.centroids - self.theta0, 2)/self.sigma_theta/self.sigma_theta)
                I2 = np.exp(-0.5 * phid * phid/self.sigma_phi/self.sigma_phi)
                I3 = np.exp(-0.5 * pow(r * self.centroids - self.r0,2)/self.sigma_r/self.sigma_r)
                Intensity = I1 * I2 * I3 * self.FMiller
                self.intensityMap[x,y] = np.sum(Intensity)

        return self.intensityMap
        
    def da_Intensity(self):
        self.iMiller = self.hkl_dimension * 2 + 1

        # Convert these to Dask arrays
        self.G1 = da.from_array(self.Bpeaks[0:self.iMiller,:,:] + np.finfo(float).eps)
        self.G2 = da.from_array(self.Bpeaks[self.iMiller:2*self.iMiller,:,:] + np.finfo(float).eps)
        self.G3 = da.from_array(self.Bpeaks[2*self.iMiller:3*self.iMiller,:,:] + np.finfo(float).eps)
        self.FMiller = da.from_array(self.Bpeaks[3*self.iMiller:4*self.iMiller,:,:])

        self.centroids = da.ones((self.iMiller, self.iMiller, self.iMiller))
        
        # The positions(r0,theta0,phi0) of Bragg peaks in spherical coordinates.
        self.theta0 = np.pi / 2 - da.arctan(self.G3 / da.sqrt(self.G2 ** 2 + self.G1 ** 2))
        
        # Compute phi0 using the static method
        self.phi0 = da.ones((self.iMiller, self.iMiller, self.iMiller), dtype=self.G1.dtype)
        self.phi0 = da.map_blocks(self.compute_phi0_block, self.phi0, self.G1, self.G2, dtype=self.phi0.dtype)
        
        self.r0 = da.sqrt(self.G1 ** 2 + self.G2 ** 2 + self.G3 ** 2)

        # The positions(r,theta,phi) of image plane in spherical coordinates.
        ix, iy = self.gridx.shape
        intensityMap_numpy = np.zeros((ix, iy))  # Temporary NumPy array to hold results
        ix = np.arange(ix)
        iy = np.arange(iy)

        for x in ix:
            for y in iy:
                theta = np.pi / 2 - np.arctan(self.gridz[x, y] / abs(self.gridx[x, y]))
                r = np.sqrt(self.gridx[x, y] ** 2 + self.gridz[x, y] ** 2)
                
                if self.gridx[x, y] > 0:
                    phi = 0
                else:
                    phi = np.pi

                phi = phi * self.centroids.compute()  # Convert Dask array to NumPy array before using in NumPy operations
                phid = abs(phi - self.phi0.compute())
                phid = abs(abs(phid - np.pi) - np.pi)
                
                I1 = np.exp(-0.5 * ((theta * self.centroids.compute() - self.theta0.compute()) ** 2) / self.sigma_theta ** 2)
                I2 = np.exp(-0.5 * (phid ** 2) / self.sigma_phi ** 2)
                I3 = np.exp(-0.5 * ((r * self.centroids.compute() - self.r0.compute()) ** 2) / self.sigma_r ** 2)
                Intensity = I1 * I2 * I3 * self.FMiller.compute()
                
                intensityMap_numpy[x, y] = np.sum(Intensity)

        self.intensityMap = da.from_array(intensityMap_numpy)  # Convert back to Dask array
        return self.intensityMap

    def mapblock_Intensity(self):
        ''' INTENSITY CALCULATIONS: Calculate the intensity of the Bragg Peaks from the atomic form factor.
        '''
        self.iMiller = self.hkl_dimension * 2 + 1
        
        # Convert these to Dask arrays for parallel computation.
        self.G1 = da.from_array(self.Bpeaks[0:self.iMiller,:,:] + np.finfo(float).eps)
        self.G2 = da.from_array(self.Bpeaks[self.iMiller:2*self.iMiller,:,:] + np.finfo(float).eps)
        self.G3 = da.from_array(self.Bpeaks[2*self.iMiller:3*self.iMiller,:,:] + np.finfo(float).eps)
        self.FMiller = da.from_array(self.Bpeaks[3*self.iMiller:4*self.iMiller,:,:])

        self.centroids = da.ones((self.iMiller, self.iMiller, self.iMiller))
        
        # Replace these calculations with Dask-enabled operations.
        self.theta0 = np.pi / 2 - da.arctan(self.G3 / da.sqrt(self.G2 ** 2 + self.G1 ** 2))
        
        self.phi0 = da.ones((self.iMiller, self.iMiller, self.iMiller))
        self.phi0 = da.map_blocks(self.compute_phi0_block, self.phi0, self.G1, self.G2, dtype=self.phi0.dtype)
        
        self.r0 = da.sqrt(self.G1 ** 2 + self.G2 ** 2 + self.G3 ** 2)
        
        # Convert grid to Dask arrays
        self.gridx_dask = da.from_array(self.gridx)
        self.gridz_dask = da.from_array(self.gridz)
        
        ix, iy = self.gridx.shape
        self.intensityMap = da.zeros((ix, iy), dtype=float)
        
        # Replace nested loops with da.map_blocks
        self.intensityMap = da.map_blocks(self.compute_intensity_block, self.intensityMap, 
                                          self.gridx, self.gridz, self.centroids, self.phi0,
                                          self.theta0, self.r0, self.FMiller, self.sigma_theta,
                                          self.sigma_phi, self.sigma_r, dtype=self.intensityMap.dtype)

        # Convert the final Dask array to a NumPy array if needed.
        self.intensityMap = self.intensityMap.compute()

        return self.intensityMap

    def dask_Intensity(self):
        self.iMiller = self.hkl_dimension * 2 + 1
        
        # Keep these as Dask arrays.
        self.G1 = da.from_array(self.Bpeaks[0:self.iMiller,:,:] + np.finfo(float).eps)
        self.G2 = da.from_array(self.Bpeaks[self.iMiller:2*self.iMiller,:,:] + np.finfo(float).eps)
        self.G3 = da.from_array(self.Bpeaks[2*self.iMiller:3*self.iMiller,:,:] + np.finfo(float).eps)
        self.FMiller = da.from_array(self.Bpeaks[3*self.iMiller:4*self.iMiller,:,:])

        self.centroids = da.ones((self.iMiller, self.iMiller, self.iMiller))
        self.theta0 = np.pi / 2 - da.arctan(self.G3 / da.sqrt(self.G2 ** 2 + self.G1 ** 2))
        self.phi0 = da.map_blocks(self.compute_phi0_block, da.ones((self.iMiller, self.iMiller, self.iMiller)), self.G1, self.G2)
        self.r0 = da.sqrt(self.G1 ** 2 + self.G2 ** 2 + self.G3 ** 2)

        ix, iy = self.gridx.shape
        self.intensityMap = np.zeros((ix, iy), dtype=float)

        delayed_results = []

        for x in range(ix):
            for y in range(iy):
                delayed_result = delayed(self.compute_intensity_block)(self.intensityMap[x, y], self.gridx[x, y], self.gridz[x, y], 
                                                                    self.centroids, self.phi0, self.theta0, self.r0, 
                                                                    self.FMiller, self.sigma_theta, self.sigma_phi, self.sigma_r)
                delayed_results.append((x, y, delayed_result))
                
        computed_results = compute(*[res[2] for res in delayed_results])

        for idx, (x, y, _) in enumerate(delayed_results):
            self.intensityMap[x, y] = computed_results[idx]

        return self.intensityMap

    @staticmethod
    def compute_phi0_block(block, G1_block, G2_block):
        iMiller = block.shape[0]
        i = np.arange(iMiller)
        phi0_block = np.ones_like(block)
        for k1 in i:
            for k2 in i:
                for k3 in i:
                    if G1_block[k1, k2, k3] > 0:
                        phi0_block[k1, k2, k3] = np.arcsin(G2_block[k1, k2, k3] / np.sqrt(G2_block[k1, k2, k3]**2 + G1_block[k1, k2, k3]**2))
                    else:
                        phi0_block[k1, k2, k3] = np.pi + np.arcsin(G2_block[k1, k2, k3] / np.sqrt(G2_block[k1, k2, k3]**2 + G1_block[k1, k2, k3]**2))

                    if abs(G2_block[k1, k2, k3]) < 0.2:
                        if abs(G1_block[k1, k2, k3]) < 0.2:
                            phi0_block[k1, k2, k3] = 0
        return phi0_block

    @staticmethod
    def compute_intensity_block(intensity_block, gridx_block, gridz_block, centroids, phi0, theta0, r0, FMiller, sigma_theta, sigma_phi, sigma_r):
        intensity_block_copy = np.copy(intensity_block)
        ix, iy = intensity_block_copy.shape
        
        for x in range(ix):
            for y in range(iy):
                theta = np.pi / 2 - np.arctan(gridz_block[x, y] / abs(gridx_block[x, y]))
                r = np.sqrt(gridx_block[x, y] ** 2 + gridz_block[x, y] ** 2)
                
                if gridx_block[x, y] > 0:
                    phi = 0
                else:
                    phi = np.pi

                phi = phi * centroids
                phid = np.abs(phi - phi0)
                phid = np.abs(np.abs(phid - np.pi) - np.pi)

                I1 = delayed(np.exp)(-0.5 * ((theta * centroids - theta0) ** 2) / sigma_theta ** 2)
                I2 = delayed(np.exp)(-0.5 * (phid ** 2) / sigma_phi ** 2)
                I3 = delayed(np.exp)(-0.5 * ((r * centroids - r0) ** 2) / sigma_r ** 2)
                
                Intensity = delayed(lambda i1, i2, i3, fm: i1 * i2 * i3 * fm)(I1, I2, I3, FMiller)
                
                intensity_block_copy[x, y] = np.sum(compute(Intensity)[0])
        
        return intensity_block_copy
        
    @staticmethod
    def compute_fq_block(ii_block, AF, q2, a1_rotated, a2_rotated, a3_rotated, G1, G2, G3):
        F = 0
        for j in ii_block:
            x = np.searchsorted(AF[:, 0], j[0])
            fq = 0
            fq += AF[x, 1] * np.exp(-AF[x, 2] * q2/16/np.pi**2)
            fq += AF[x, 3] * np.exp(-AF[x, 4] * q2/16/np.pi**2)
            fq += AF[x, 5] * np.exp(-AF[x, 6] * q2/16/np.pi**2)
            fq += AF[x, 7] * np.exp(-AF[x, 8] * q2/16/np.pi**2)
            fq += AF[x, 9]
            RR = j[1] * a1_rotated + j[2] * a2_rotated + j[3] * a3_rotated
            F += fq * np.exp(1j * (G1 * RR[0] + G2 * RR[1] + G3 * RR[2]))
        return np.abs(F)**2

    # Run BraggPeaks method
    def runBragg(self):
        return self.BraggPeaks()

    # Run Intensity method
    def runIntensity(self):
        return self.Intensity()

    # Run both BraggPeaks and Intensity methods sequentially
    def runSimulation(self):
        with open(os.devnull, 'w') as fnull, redirect_stdout(fnull):
            self.runBragg()
            self.runIntensity()
        # return self.runIntensity()

    # Save metadata to a .txt file
    def save_metadata(self, imgpath=None):
        if self.simPath or imgpath:
            save_path = Path(self.simPath) if not imgpath else Path(imgpath).parent
            metadata_filename = f"{self.projectname}_{timestamp}.txt" if imgpath else f"{self.projectname}_metadata.txt"
            metadata_file = save_path / metadata_filename
            
            with open(metadata_file, 'w') as f:
                for key, value in self.__dict__.items():
                    if key not in ['intensityMap', 'Bpeaks', 'M_qxy', 'M_qz', 'FMiller']:
                        f.write(f"{key}: {value}\n")
            
            print(f"Metadata saved to {metadata_file}")
        else:
            print("Simulation path not defined. Metadata not saved.")

    # Plot and save the intensityMap as a heatmap
    def plotIntensityMap(self, scaleLog=False, saveFig=False, cmap='turbo', plotname = None):
        if self.intensityMap is None:
            print("Run the simulation first to generate intensityMap.")
            return
        
        plt.close('all')
        if scaleLog:
            intensityMap = np.log(self.intensityMap + 1)
        else:
            intensityMap = self.intensityMap

        plt.imshow(intensityMap, 
                   interpolation='nearest', 
                   cmap=cmap,
                   origin='lower',
                   aspect='auto', 
                   extent=[self.qxyMin, self.qxyMax, self.qzMin, self.qzMax],
                   vmin=np.nanpercentile(self.intensityMap, 10),
                   vmax=np.nanpercentile(self.intensityMap, 99))
        
        plt.colorbar(label='Intensity')
        plt.xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)')
        plt.ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)')

        if plotname is None:
            if self.projectname is None:
                plotname = self.projectname
            else:
                plotname = 'simulation'

        plt.title(f'Intensity Map: {plotname}')

        if saveFig:
            if self.simPath:
                save_path = Path(self.simPath)
                save_path.mkdir(parents=True, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                imgname = f"{plotname}_{timestamp}.png"
                imgpath = save_path / imgname
                plt.savefig(imgpath, dpi=200, bbox_inches='tight', pad_inches=0)
                print(f"Image saved to {imgpath}")
                
                # Save metadata with the same timestamp
                self.save_metadata(imgpath=imgpath)
            else:
                print("Simulation path not defined. Image not saved.")

        plt.show()

    # -- GIWAXS PEAK FINDER:
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

    ''' # - Compute Intensity Cell Blocks
        # @staticmethod
        # def compute_intensity_for_cell(x, y, gridx_block, gridz_block, centroids, phi0, theta0, r0, FMiller, sigma_theta, sigma_phi, sigma_r):
        #     theta = np.pi / 2 - np.arctan(gridz_block[x, y] / abs(gridx_block[x, y]))
        #     r = np.sqrt(gridx_block[x, y] ** 2 + gridz_block[x, y] ** 2)
            
        #     if gridx_block[x, y] > 0:
        #         phi = 0
        #     else:
        #         phi = np.pi

        #     phi = phi * centroids
        #     phid = np.abs(phi - phi0)
        #     phid = np.abs(np.abs(phid - np.pi) - np.pi)

        #     I1 = np.exp(-0.5 * ((theta * centroids - theta0) ** 2) / sigma_theta ** 2)
        #     I2 = np.exp(-0.5 * (phid ** 2) / sigma_phi ** 2)
        #     I3 = np.exp(-0.5 * ((r * centroids - r0) ** 2) / sigma_r ** 2)
        #     Intensity = I1 * I2 * I3 * FMiller
                
        #     return np.sum(Intensity)

        # ----
        # ix, iy = self.gridx.shape
        # self.intensityMap = np.zeros((ix, iy), dtype=float)  # Initialize as a NumPy array

        # delayed_results = []  # Empty list to store delayed objects

        # # Loop through and create delayed objects
        # for x in range(ix):
        #     for y in range(iy):
        #         delayed_result = delayed(self.compute_intensity_for_cell)(
        #             x, y, self.gridx, self.gridz, self.centroids, self.phi0, 
        #             self.theta0, self.r0, self.FMiller, self.sigma_theta, 
        #             self.sigma_phi, self.sigma_r
        #         )
        #         delayed_results.append((x, y, delayed_result))

        # # Compute all delayed objects
        # computed_results = compute(*[res[2] for res in delayed_results])

        # # Assign the computed results back to the intensityMap
        # for idx, (x, y, _) in enumerate(delayed_results):
        #     self.intensityMap[x, y] = computed_results[idx]

        # return self.intensityMap
            
        # @staticmethod
        # def compute_intensity_block(intensity_block, gridx_block, gridz_block, centroids, phi0, theta0, r0, FMiller, sigma_theta, sigma_phi, sigma_r):
        #     intensity_block_copy = np.copy(intensity_block)
        #     ix, iy = intensity_block_copy.shape
        #     for x in range(ix):
        #         for y in range(iy):
        #             theta = np.pi / 2 - np.arctan(gridz_block[x, y] / abs(gridx_block[x, y]))
        #             r = np.sqrt(gridx_block[x, y] ** 2 + gridz_block[x, y] ** 2)
                    
        #             if gridx_block[x, y] > 0:
        #                 phi = 0
        #             else:
        #                 phi = np.pi

        #             phi = phi * centroids
        #             phid = np.abs(phi - phi0)
        #             phid = np.abs(np.abs(phid - np.pi) - np.pi)

        #             I1 = np.exp(-0.5 * ((theta * centroids - theta0) ** 2) / sigma_theta ** 2)
        #             I2 = np.exp(-0.5 * (phid ** 2) / sigma_phi ** 2)
        #             I3 = np.exp(-0.5 * ((r * centroids - r0) ** 2) / sigma_r ** 2)
        #             Intensity = I1 * I2 * I3 * FMiller
                        
        #             intensity_block_copy[x, y] = np.sum(Intensity)
        #     return intensity_block_copy
    '''
   
    ''' # def Intensity(): w/o dask parallization 
    def Intensity(self):
        # INTENSITY CALCULATIONS: Calculate the intensity of the Bragg Peaks from the atomic form factor.
        
        self.iMiller = self.hkl_dimension*2 + 1

        self.G1 = self.Bpeaks[0:self.iMiller,:,:] + np.finfo(float).eps
        self.G2 = self.Bpeaks[self.iMiller:2*self.iMiller,: ,:] + np.finfo(float).eps
        self.G3 = self.Bpeaks[2*self.iMiller:3*self.iMiller,: ,:] + np.finfo(float).eps
        self.FMiller = self.Bpeaks[3*self.iMiller:4*self.iMiller,: ,:]
        
        self.centroids = np.ones((self.iMiller,self.iMiller,self.iMiller))
        
        # The positions(r0,theta0,phi0) of Bragg peaks in spherical coordinates.
        self.theta0 = np.pi/2-np.arctan(self.G3/np.sqrt(pow(self.G2,2)+pow(self.G1,2)))
        self.phi0 = np.ones((self.iMiller, self.iMiller, self.iMiller))
        
        i = np.arange(self.iMiller)

        for k1 in i:
            for k2 in i:
                for k3 in i:
                    if self.G1[k1,k2,k3]>0:
                        self.phi0[k1,k2,k3] = np.arcsin(self.G2[k1,k2,k3]/np.sqrt(pow(self.G2[k1,k2,k3],2) + pow(self.G1[k1,k2,k3],2)))
                    else:
                        self.phi0[k1,k2,k3] = np.pi + np.arcsin(self.G2[k1,k2,k3]/np.sqrt(pow(self.G2[k1,k2,k3],2) + pow(self.G1[k1,k2,k3],2)))
                    if abs(self.G2[k1,k2,k3])<0.2:
                        if abs(self.G1[k1,k2,k3])<0.2:
                            self.phi0[k1,k2,k3]=0
                        
        self.r0 = np.sqrt(pow(self.G1,2)+pow(self.G2,2)+pow(self.G3,2))

        # The positions(r,theta,phi) of image plane in spherical coordinates.
        ix, iy = self.gridx.shape
        self.intensityMap = np.ones((ix,iy))
        ix = np.arange(ix)
        iy = np.arange(iy)

        for x in ix:
            for y in iy:
                theta = np.pi/2-np.arctan(self.gridz[x,y]/abs(self.gridx[x,y]))
                r = np.sqrt(pow(self.gridx[x,y],2) + pow(self.gridz[x,y],2))
                
                if self.gridx[x,y] > 0:
                    phi = 0
                else:
                    phi = np.pi

                phi = phi * self.centroids
                phid = abs(phi - self.phi0)
                phid = abs(abs(phid - np.pi) - np.pi)
                
                I1 = np.exp(-0.5 * pow(theta * self.centroids - self.theta0, 2)/self.sigma_theta/self.sigma_theta)
                I2 = np.exp(-0.5 * phid * phid/self.sigma_phi/self.sigma_phi)
                I3 = np.exp(-0.5 * pow(r * self.centroids - self.r0,2)/self.sigma_r/self.sigma_r)
                Intensity = I1 * I2 * I3 * self.FMiller
                self.intensityMap[x,y] = np.sum(Intensity)

        return self.intensityMap
        '''

# -- Diffuse Code Implementation
    '''
    def diffuse(self, a1,a2,a3,positions,thetaX,thetaY,hkl_dimension,shift):
        # Lattice parameters M matrix in cartesian coordinate(angstrom)
        M=[a1,a2,a3]
        M=np.asarray(M)
        # Rotation Matrix respect to X axis, rotation angle = thetaX
        Rx=np.array([[1,0,0],[0,np.cos(thetaX),-np.sin(thetaX)],[0,np.sin(thetaX),np.cos(thetaX)]])
        # Rotation Matrix respect to Y axis, rotation angle = thetaY
        Ry=np.array([[np.cos(thetaY),0,-np.sin(thetaY)],[0,1,0],[np.sin(thetaY),0,np.cos(thetaY)]])

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
        i=np.linspace(-hkl_dimension,hkl_dimension,2*hkl_dimension+1)
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
        AF=WAXSAFF.AFF()
        
        # calculate the atomic form factor
        ii=np.linspace(0,ss-1,ss)
        ii=ii.astype(int)
        q2=G1*G1+G2*G2+G3*G3
        F=0
        for j in ii:
            x = np.searchsorted(AF[:,0],positions[j,0])
            fq=0
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
        Bpeaks=np.concatenate((G1,G2-shift,G3,F), axis=0)
        return Bpeaks
    
    def intensity_diffuse(self, gridx,gridz,Bpeaks,sigma_theta,sigma_phi,sigma_r,hkl_dimension):
        iMiller=hkl_dimension*2+1
        idiffuse=2
        G1=Bpeaks[0:iMiller,:,:]+np.finfo(float).eps
        G2=Bpeaks[iMiller:2*iMiller,:,:]+np.finfo(float).eps
        G3=Bpeaks[2*iMiller:3*iMiller,:,:]+np.finfo(float).eps
        F=Bpeaks[3*iMiller:4*iMiller,:,:]
        
        Eye=np.ones([iMiller,2,iMiller])
        # The positions(r0,theta0,phi0) of Bragg peaks in spherical coordinates.
        theta0=np.pi/2-np.arctan(G3/np.sqrt(pow(G2,2)+pow(G1,2)))
        phi0=np.ones((iMiller,2,iMiller))
        i=np.arange(iMiller)
        for k1 in i:
            for k2 in np.array([0,1]):
                for k3 in i:
                    if G1[k1,k2,k3]>0:
                        phi0[k1,k2,k3]=np.arcsin(G2[k1,k2,k3]/np.sqrt(pow(G2[k1,k2,k3],2)+pow(G1[k1,k2,k3],2)))
                    else:
                        phi0[k1,k2,k3]=np.pi+np.arcsin(G2[k1,k2,k3]/np.sqrt(pow(G2[k1,k2,k3],2)+pow(G1[k1,k2,k3],2)))
                    if abs(G2[k1,k2,k3])<0.2:
                        if abs(G1[k1,k2,k3])<0.2:
                            phi0[k1,k2,k3]=0
                        
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
                    phi=0
                else:
                    phi=np.pi
                phi=phi*Eye
                phid=abs(phi-phi0)
                phid=abs(abs(phid-np.pi)-np.pi)
                I1=np.exp(-0.5*pow(theta*Eye-theta0,2)/sigma_theta/sigma_theta)
                I2=np.exp(-0.5*phid*phid/sigma_phi/sigma_phi)
                I3=np.exp(-0.5*pow(r*Eye-r0,2)/sigma_r/sigma_r)
                Intensity=I1*I2*I3*F
                I0[x,y]=np.sum(Intensity)
        return I0
    '''