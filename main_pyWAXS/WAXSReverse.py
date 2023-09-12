import numpy as np
import scipy.ndimage as ndimage
import scipy.io
import time as time
# - Matplotlib Imports
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Button
from matplotlib.widgets import RangeSlider
import matplotlib.patheffects as path_effects

## -- Custom Imports -- ##
from WAXSSearch import WAXSSearch # WAXS Peak Search class
from WAXSDiffSim import WAXSDiffSim # WAXS simulation class
import WAXSAFF # atomic form factors for Bragg scattering calculation

class WAXSReverse():
    def __init__(self) -> None:
        
        # experiment data qxy range, we will get this WAXSSearch() class instance at some point and initialize to null range here.
        self.qxy_exp=[0,4]
        # experiment data qz range, we will get this WAXSSearch() class instance at some point and initialize to null range here.
        self. qz_exp=[0,4]
        
        # Bragg peak positions from WAXSSearch() method experiment output, we will get this WAXSSearch() at some point and initialize to null range here.
        Bragg_peaks_exp=[
            [0,1],
            [0,2],
            [0,3],
            [2,0],
            [2,1],
            [2,2],
            [2,3],
            [2.818,0],
            [2.818,1],
            [2.818,2],
            [2.818,3],
            [1.414,0.5],
            [1.414,1.5],
            [1.414,2.5],
            [3.16,0.5],
            [3.16,1.5],
            [3.16,2.5],
            [4.24,0.5],
            [4.24,1.5],
            [4.24,2.5],
            [4,0],
            [4,1],
            [4,2],
            [4,3]
        ]
        hkl=5

        pass


    def getparams():
        # this will eventually be an output from the WAXSSearch class methods.
        # (1) get q_xy
        pass

    def main():
        # total_distance0=99999999
        for i in range(500000):
            b1,b2,b3=self.make_cif(Bragg_peaks_exp)
            qxy,qz=self.Bragg_peaks(b1,b2,b3,hkl)
            qxy=qxy.ravel()
            qz=qz.ravel()
            combined = np.column_stack((qxy, qz))
            j=0
            total_distance=0
            for ii in Bragg_peaks_exp:
                total_distance=total_distance+self.closest_distance(combined,Bragg_peaks_exp[j])
                j=j+1
            j1=0
            for ii in combined:
                if combined[j1,0]<qxy_exp[1]:
                    if combined[j1,1]<qz_exp[1]:
                        total_distance=total_distance+closest_distance(Bragg_peaks_exp,combined[j1])
                j1=j1+1
            if total_distance<total_distance0:
                total_distance0=total_distance
                FT(b1,b2,b3)

    def make_cif(self, Bragg_peaks):
        # b1=[np.random.uniform(-ub, ub),0,np.random.uniform(-ub, ub)]
        b2x=np.random.uniform(-1.5, 1.5)
        b3x=np.random.uniform(-5, 5)
        b3y=np.random.uniform(-5, 5)
        b3z=np.random.uniform(-5, 5)
        b1=[0,0,1]
        b2=[b2x,np.sqrt(2.25-b2x*b2x),0.5]
        b3=[b3x,b3y,b3z]
        return b1,b2,b3

    def find_c(self, array):
        x_values_with_leading_zero = [pair[1] for pair in array if pair[0] == 0]
        if not x_values_with_leading_zero:
            return None
        return min(x_values_with_leading_zero)

    def closest_distance(self, array, point):
        # Convert the input to numpy arrays if they aren't already
        array = np.array(array)
        point = np.array(point)

        # Compute squared distances
        squared_distances = np.sum((array - point)**2, axis=1)

        # Find the minimum squared distance and take its square root to get the actual distance
        min_distance = np.sqrt(np.min(squared_distances))

        return min_distance

    def Bragg_peaks(self, b1,b2,b3,hkl_dimension):
        # grid for Miller index
        i=np.linspace(-hkl_dimension,hkl_dimension,2*hkl_dimension+1)
        H,K,L=np.meshgrid(i,i,i)

        # The position of Bragg peaks in reciprocal space
        G1=H*b1[0]+K*b2[0]+L*b3[0]
        G2=H*b1[1]+K*b2[1]+L*b3[1]
        G3=H*b1[2]+K*b2[2]+L*b3[2]

        q2=G1*G1+G2*G2+G3*G3
        F=1
        Bpeaks=np.concatenate((G1,G2,G3), axis=0)
        # return Bpeaks,pow(G1*G1+G2*G2,0.5),pow(G3*G3,0.5)
        return pow(G1*G1+G2*G2,0.5),pow(G3*G3,0.5)

    def FT(self, a1,a2,a3):
        # Lattice parameters M matrix in cartesian coordinate(angstrom)
        M=[a1,a2,a3]
        M=np.asarray(M)

        # New lattice parameter
        aa1=M[0,:]
        aa2=M[1,:]
        aa3=M[2,:]

        # reciprocal lattice
        volume=np.matmul(aa3,np.cross(aa1,aa2))
        b1=2*np.pi*np.cross(aa2,aa3)/volume
        b2=2*np.pi*np.cross(aa3,aa1)/volume
        b3=2*np.pi*np.cross(aa1,aa2)/volume
        print(b1,b2,b3)


