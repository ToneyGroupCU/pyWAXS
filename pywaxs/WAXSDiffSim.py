import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import matplotlib.cm as cm
import pandas as pd
from pathlib import Path
from typing import Union

# Custom Imports
import WAXSAFF

class WAXSDiffSim:
    def __init__(self, 
                 *file_paths: Union[Path, str], 
                 sigma1=0, 
                 sigma2=0, 
                 sigma3=0, 
                 hkl_dimension=0, 
                 thetax=0, 
                 thetay=0):
        
        self.sigma1 = sigma1 # smearing about the theta coordinate
        self.sigma2 = sigma2 # smearing about the phi coordinate
        self.sigma3 = sigma3 # smearing about the r coordinate
        self.hkl_dimension = hkl_dimension # extent of (hkl) to calculate 
        self.thetax = thetax # initial rotation about x-axis
        self.thetay = thetay # initial rotation about y-axis
        
        # Define a DataFrame to store the required information
        columns = ['path', 'a1', 'a2', 'a3', 'positions', 'bragg_peaks', 'other_bragg_info']
        self.diffsim_df = pd.DataFrame(columns=columns) # Changed to 'diffsim_df'

        # Initialize with the given file paths
        for path in file_paths:
            self.read_poscar(path)

    # -- READ POSCAR FILE: Read-in the POSCAR input file.
    def read_poscar(self, address: Union[Path, str]):
        f = open(address)
        ind = 0
        for x in f:
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
        f.close()
        ind = 0
        iii = 0

        for ii in z1:
            position[iii:iii+ii+1,0]=z[ind]
            iii=iii+ii
            ind=ind+1
        
        # Storing values in the DataFrame
        self.data.loc[len(self.data)] = [address, a1, a2, a3, position, None, None]
        # self.diffsim_df.loc[len(self.diffsim_df)] = [address, a1, a2, a3, position, None, None]

        # return a1,a2,a3,position

    # -- BRAGG PEAK CALCULATION: Calculate the position of the Bragg Peaks.
    def Bragg_peaks(self, a1, a2, a3, positions, thetax, thetay, hkl_dimension):
        """
        Description: Bragg_peaks function calculate the position of Bragg peaks in reciprocal space using lattice parameters and position of atoms read from the POSCAR file.
        Two rotation angles respect to x and y axis are added to adjust the orientation of the single crystal.
        """
        # Lattice parameters M matrix in cartesian coordinate(angstrom)
        M=[a1,a2,a3]
        M=np.asarray(M)
        # Rotation Matrix respect to X axis, rotation angle = thetax
        Rx=np.array([[1,0,0],[0,np.cos(thetax),-np.sin(thetax)],[0,np.sin(thetax),np.cos(thetax)]])
        # Rotation Matrix respect to Y axis, rotation angle = thetay
        Ry=np.array([[np.cos(thetay),0,-np.sin(thetay)],[0,1,0],[np.sin(thetay),0,np.cos(thetay)]])

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
        H,K,L=np.meshgrid(i,i,i)
        
        # The position of Bragg peaks in reciprocal space
        G1=H*b1[0]+K*b2[0]+L*b3[0]
        G2=H*b1[1]+K*b2[1]+L*b3[1]
        G3=H*b1[2]+K*b2[2]+L*b3[2]
        
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
        Bpeaks=np.concatenate((G1,G2,G3,F), axis=0)
        return Bpeaks,pow(G1*G1+G2*G2,0.5),pow(G3*G3,0.5),F

    # -- INTENSITY CALCULATIONS: Calculate the intensity of the Bragg Peaks from the atomic form factor.
    def intensity(self, gridx, gridz, Bpeaks, sigma1, sigma2, sigma3, hkl_dimension):
        iMiller=hkl_dimension*2+1
        G1=Bpeaks[0:iMiller,:,:]+np.finfo(float).eps
        G2=Bpeaks[iMiller:2*iMiller,:,:]+np.finfo(float).eps
        G3=Bpeaks[2*iMiller:3*iMiller,:,:]+np.finfo(float).eps
        F=Bpeaks[3*iMiller:4*iMiller,:,:]
        
        Eye=np.ones((iMiller,iMiller,iMiller))
        # The positions(r0,theta0,phi0) of Bragg peaks in spherical coordinates.
        theta0=np.pi/2-np.arctan(G3/np.sqrt(pow(G2,2)+pow(G1,2)))
        phi0=np.ones((iMiller,iMiller,iMiller))
        i=np.arange(iMiller)
        for k1 in i:
            for k2 in i:
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
                I1=np.exp(-0.5*pow(theta*Eye-theta0,2)/sigma1/sigma1)
                I2=np.exp(-0.5*phid*phid/sigma2/sigma2)
                I3=np.exp(-0.5*pow(r*Eye-r0,2)/sigma3/sigma3)
                Intensity=I1*I2*I3*F
                I0[x,y]=np.sum(Intensity)
        return I0

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

# -- Diffuse Code Implementation
    '''
    def diffuse(self, a1,a2,a3,positions,thetax,thetay,hkl_dimension,shift):
        # Lattice parameters M matrix in cartesian coordinate(angstrom)
        M=[a1,a2,a3]
        M=np.asarray(M)
        # Rotation Matrix respect to X axis, rotation angle = thetax
        Rx=np.array([[1,0,0],[0,np.cos(thetax),-np.sin(thetax)],[0,np.sin(thetax),np.cos(thetax)]])
        # Rotation Matrix respect to Y axis, rotation angle = thetay
        Ry=np.array([[np.cos(thetay),0,-np.sin(thetay)],[0,1,0],[np.sin(thetay),0,np.cos(thetay)]])

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
    
    def intensity_diffuse(self, gridx,gridz,Bpeaks,sigma1,sigma2,sigma3,hkl_dimension):
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
                I1=np.exp(-0.5*pow(theta*Eye-theta0,2)/sigma1/sigma1)
                I2=np.exp(-0.5*phid*phid/sigma2/sigma2)
                I3=np.exp(-0.5*pow(r*Eye-r0,2)/sigma3/sigma3)
                Intensity=I1*I2*I3*F
                I0[x,y]=np.sum(Intensity)
        return I0
    '''