import os, re, sys, gc, time
import numpy as np
from numpy import unravel_index # used to grab amax index from np_array

import scipy as sc
from scipy.spatial.distance import cdist

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap

import random
from random import seed

# %matplotlib widget

# -- Functionality Description
# mcpointgen() - Generate pseudo-random set of points.
# checkneighborhood() - For each point, return the local maxima.
# checkneighborhood_inverted() - Take the same data matrix, reverse the topology and repeat to return the local maxima.
# get_group_statistics() - Takes the set of points from the checkneighborhood() function and groups them according to an input proximity value. Returns the a dictionary of group statistics.
# remove_common_sets() - Removes index pairs found in both the regular and inverted local max data sets. These pairs are saddle points found from both local minima and maxima approach methods - they are not true peaks.
# mcpeakfinder() - Wraps the functions above to find peaks, compress into two groups, and remove saddle points. Returns the post-process indices.

# Notes for further improvements: 
    # kd tree - algorithm for sorting points 
    # import scipy.spatial.KDTree
    # sphere point picking
    # steepest ascent methods/saddle point approximation

def normalize_array(arr):
    max_val = np.max(arr)
    normalized_arr = arr / max_val
    return normalized_arr

# -- Description: Outputs a seeded set number of pseduo-random indices over the define grid space parameters.
def mcpointgen(seednum=1, genPoints=1000, pixeldimX=1000, pixeldimY=1000):
    """
    Seeded pseudo-random number generator for use with 2D Monte-Carlo peak finder.
    Returns an array of shape (genPoints, 2) containing randomly generated (x, y) coordinates.

    Parameters:
    seednum : int, optional
        Seed value used to produce the random numbers. Default is 1.
    genPoints : int, optional
        Number of random points generated on the rangeX x rangeY grid. Default is 1000.
    pixeldimX : int, optional
        Pixel dimensions in X over data set. Default is 1000.
    pixeldimY : int, optional
        Pixel dimensions in Y over data set. Default is 1000.

    Returns:
    np_mcrand : numpy array
        Array of shape (genPoints, 2) containing randomly generated (x, y) coordinates.
    """

    np.random.seed(seednum)
    np_mcrand = np.random.randint(0, pixeldimX, size=(genPoints, 2), dtype=int)
    return np_mcrand

# -- Description: Returns a single peak position it finds by walking in direction of increasing intensity.
def checkneighborhoodmax(data_nparray, pixindexX, pixindexY, radius=5, pixeldimX=1000, pixeldimY=1000, threshold=0.005):
    """Checks the pixel neighborhood for the local maximum with respect to the input pixel indices (x,y). 
    Returns the coordinates of the maximum pixel.
    Input parameters:
    data_nparray : numpy array
        input numpy array, indexed values will be typecasted as needed.
    pixindexX : int
        centered pixel index in x w/ respect to pixel array
    pixindexY : int
        centered pixel index in y w/ respect to pixel array
    radius : int
        radius about the pixel center (non-inclusive) that 
        defines the neighborhood parameter space to check for
        the maxima in this range. 
    threshold : float64
        decimal val (0-1), defines the percentage difference 
        needed for moving to a different pixel as origin."""

    pixminX = int(pixindexX - radius) # defines the boxed pixel neighborhood x-index minimum (x,y)
    pixmaxX = int(pixindexX + radius) # defines the boxed pixel neighborhood x-index maximum (x,y)

    pixminY = int(pixindexY - radius) # defines the boxed pixel neighborhood y-index minimum (x,y)
    pixmaxY = int(pixindexY + radius) # defines the boxed pixel neighborhood y-index maximum (x,y)

    # pixrangeX = (pixmaxX - pixminX) + 1
    # pixrangeY = (pixmaxY - pixminY) + 1

    origincoord = (pixindexX, pixindexY) # store the origin coordinates as a tuple to make them immutable

    loop = True
    counter = 0
    while loop == True:
        counter += 1
        # if statement fail-safes in case we hit the edge of the img array
        if pixmaxX > pixeldimX:
            pixmaxX = pixeldimX-1 # -1 because indexing from 0
        
        if pixmaxY > pixeldimY:
            pixmaxY = pixeldimY-1 # -1 because indexing from 0
        
        if pixminX < 0:
            pixminX = 0
        
        if pixminY < 0:
            pixminY = 0

        centerpix_intensity = data_nparray[pixindexX, pixindexY] # grab the center pixel intensity
        subset_nparray = data_nparray[pixminX:pixmaxX, pixminY:pixmaxY] # create the subset array

        temp_origincoord = list(unravel_index(subset_nparray.argmax(), subset_nparray.shape)) # get the indices of the subset array maximum
        temp_centerpix_intensity = subset_nparray[int(temp_origincoord[0]), int(temp_origincoord[1])] # get the intensity of the subset array maximum
        
        d_intensity = temp_centerpix_intensity - centerpix_intensity # calculate the delta intensity between current origin and the new candidate origin (should be positive above threshold to make the move)
        # print ('d_intensity = ', str(d_intensity))
    
        d_threshold = threshold * centerpix_intensity # threshold default = 0.05 (5% of center pixel intensity)
        # print ('d_threshold = ', str(d_threshold))

        # if the center pixel intensity is less than or equal to (threshold default = 5%) percent smaller than the next pixel, stop the search
        if d_intensity <= d_threshold:
            loop = False # kill the loop
            if d_intensity > 0: # if the intensity is greater, reassign the points to the final position (because why not)
                pixindexX = pixminX + int(temp_origincoord[0]) # set the new index bounds in X
                pixindexY = pixminY + int(temp_origincoord[1]) # set the new index bounds in Y
        else: # otherwise, rewrite the pixel index center to repeat the sequence
            pixindexX = pixminX + int(temp_origincoord[0]) # set the new index bounds in X
            pixindexY = pixminY + int(temp_origincoord[1]) # set the new index bounds in Y

            pixminX = int(pixindexX - radius) # change the pixel neighborhood zone x-min
            pixmaxX = int(pixindexX + radius) # change the pixel neighborhood zone x-max

            pixminY = int(pixindexY - radius) # change the pixel neighborhood zone y-min
            pixmaxY = int(pixindexY + radius) # change the pixel neighborhood zone y-max

    # return the pixel indices of the greater array
    return pixindexX, pixindexY

# -- Description: Returns a single peak position it finds by walking in direction of decreasing intensity.
def checkneighborhoodmin(data_nparray, pixindexX, pixindexY, radius=5, pixeldimX=1000, pixeldimY=1000, threshold=0.005):
    """Checks the pixel neighborhood for the local maximum with respect to the input pixel indices (x,y). 
    Returns the coordinates of the maximum pixel.
    Input parameters:
    data_nparray : numpy array
        input numpy array, indexed values will be typecasted as needed.
    pixindexX : int
        centered pixel index in x w/ respect to pixel array
    pixindexY : int
        centered pixel index in y w/ respect to pixel array
    radius : int
        radius about the pixel center (non-inclusive) that 
        defines the neighborhood parameter space to check for
        the maxima in this range. 
    threshold : float64
        decimal val (0-1), defines the percentage difference 
        needed for moving to a different pixel as origin."""

    pixminX = int(pixindexX - radius) # defines the boxed pixel neighborhood x-index minimum (x,y)
    pixmaxX = int(pixindexX + radius) # defines the boxed pixel neighborhood x-index maximum (x,y)

    pixminY = int(pixindexY - radius) # defines the boxed pixel neighborhood y-index minimum (x,y)
    pixmaxY = int(pixindexY + radius) # defines the boxed pixel neighborhood y-index maximum (x,y)

    # pixrangeX = (pixmaxX - pixminX) + 1
    # pixrangeY = (pixmaxY - pixminY) + 1

    origincoord = (pixindexX, pixindexY) # store the origin coordinates as a tuple to make them immutable

    loop = True
    counter = 0
    while loop == True:
        counter += 1
        # if statement fail-safes in case we hit the edge of the img array
        if pixmaxX > pixeldimX:
            pixmaxX = pixeldimX-1 # -1 because indexing from 0
        
        if pixmaxY > pixeldimY:
            pixmaxY = pixeldimY-1 # -1 because indexing from 0
        
        if pixminX < 0:
            pixminX = 0
        
        if pixminY < 0:
            pixminY = 0

        centerpix_intensity = data_nparray[pixindexX, pixindexY] # grab the center pixel intensity
        subset_nparray = data_nparray[pixminX:pixmaxX, pixminY:pixmaxY] # create the subset array

        temp_origincoord = list(unravel_index(subset_nparray.argmin(), subset_nparray.shape)) # get the indices of the subset array minimum
        temp_centerpix_intensity = subset_nparray[int(temp_origincoord[0]), int(temp_origincoord[1])] # get the intensity of the subset array minimum
        
        # d_intensity = temp_centerpix_intensity - centerpix_intensity # calculate the delta intensity between current origin and the new candidate origin (should be positive above threshold to make the move)
        # print ('d_intensity = ', str(d_intensity))
    
        d_intensity =  centerpix_intensity - temp_centerpix_intensity

        d_threshold = threshold * centerpix_intensity # threshold default = 0.05 (5% of center pixel intensity)
        # print ('d_threshold = ', str(d_threshold))

        # if the center pixel intensity is less than or equal to (threshold default = 5%) percent smaller than the next pixel, stop the search
        if d_intensity <= d_threshold:
            loop = False # kill the loop
            if d_intensity > 0: # if the intensity is greater, reassign the points to the final position (because why not)
                pixindexX = pixminX + int(temp_origincoord[0]) # set the new index bounds in X
                pixindexY = pixminY + int(temp_origincoord[1]) # set the new index bounds in Y
        else: # otherwise, rewrite the pixel index center to repeat the sequence
            pixindexX = pixminX + int(temp_origincoord[0]) # set the new index bounds in X
            pixindexY = pixminY + int(temp_origincoord[1]) # set the new index bounds in Y

            pixminX = int(pixindexX - radius) # change the pixel neighborhood zone x-min
            pixmaxX = int(pixindexX + radius) # change the pixel neighborhood zone x-max

            pixminY = int(pixindexY - radius) # change the pixel neighborhood zone y-min
            pixmaxY = int(pixindexY + radius) # change the pixel neighborhood zone y-max

    # return the pixel indices of the greater array
    return pixindexX, pixindexY

# -- Description: Inverts the input array to look for local maxima. Repeats with the checkneighborhood() function will be thrown out.
def checkneighborhood_inverted(data_nparray, pixindexX, pixindexY, radius=5, pixeldimX=1000, pixeldimY=1000, threshold=0.005):
    """Checks the pixel neighborhood for the local maximum with respect to the input pixel indices (x,y).
    Inverts the input array by multiplying it by -1 before processing.
    Returns the coordinates of the maximum pixel.
    Input parameters:
    data_nparray : numpy array
        input numpy array, indexed values will be typecasted as needed.
    pixindexX : int
        centered pixel index in x w/ respect to pixel array
    pixindexY : int
        centered pixel index in y w/ respect to pixel array
    radius : int
        radius about the pixel center (non-inclusive) that 
        defines the neighborhood parameter space to check for
        the maxima in this range. 
    threshold : float64
        decimal val (0-1), defines the percentage difference 
        needed for moving to a different pixel as origin."""

    # Invert the input array by multiplying it by -1
    data_nparray_inverted = -1 * data_nparray

    # Call the original checkneighborhood function with the inverted data array
    pixindexX_inverted, pixindexY_inverted = checkneighborhoodmax(data_nparray_inverted, pixindexX, pixindexY, radius, pixeldimX, pixeldimY, threshold)

    return pixindexX_inverted, pixindexY_inverted, data_nparray_inverted

# -- Description: Finds peaks located on an input image array using the pseudo-randomly seeded points and neighborhood() functionality.
def mc_findpeaks(data_nparray, radius=5, seednum=1, genPoints=1000, pixeldimX=1000, pixeldimY=1000, threshold=0.005):

    np_mcrand = mcpointgen(seednum, genPoints, pixeldimX, pixeldimY) # generate array of random points
    np_mcrand_length = int(list(np.shape(np_mcrand))[0]) # get length of the np_mcrand array (should be genPoints input)

    locmax_indices = np.zeros([int(list(np.shape(np_mcrand))[0]), int(list(np.shape(np_mcrand))[1])]) # local max array
    locmin_indices = np.zeros([int(list(np.shape(np_mcrand))[0]), int(list(np.shape(np_mcrand))[1])]) # local min array

    i = 0
    for i in range(0, np_mcrand_length-1):
        pixindexX = int(np_mcrand[i,0]) # grab the seeded pseudo-random index x in (x,y)
        pixindexY = int(np_mcrand[i,1]) # grab the seeded pseudo-random index y in (x,y)

        locmaxX, locmaxY = checkneighborhoodmax(data_nparray, pixindexX, pixindexY, radius=radius, pixeldimX=pixeldimX, pixeldimY=pixeldimY, threshold=threshold) # get the regular local maxima indices
        locminX, locminY = checkneighborhoodmin(data_nparray, pixindexX, pixindexY, radius=radius, pixeldimX=pixeldimX, pixeldimY=pixeldimY, threshold=threshold) # get the regular local minima indices
        # invlocmaxX, invlocmaxY, invdata_nparray = checkneighborhood_inverted(data_nparray, pixindexX, pixindexY, radius=radius, pixeldimX=pixeldimX, pixeldimY=pixeldimY, threshold=threshold) # get the inverted local maxima indices

        locmax_indices[i, 0] = locmaxX # store normal local max X-coordinate
        locmax_indices[i, 1] = locmaxY # store normal local max Y-coordinate

        locmin_indices[i, 0] = locminX # store the local max X-coord of the inverted image
        locmin_indices[i, 1] = locminY # store the local max Y-coord of the inverted image

    # return locmax_indices, invlocmax_indices, invdata_nparray
    return locmax_indices, locmin_indices

# -- Description: Groups coordinate pairs based on proximity and returns a dictionary of statistical information on the pairwise groupings.
def get_group_statistics(intensity_data, indices, proximity):
    # Compute pairwise distances between indices
    distances = cdist(indices, indices)

    # Group indices that are closer than proximity threshold
    groups = []
    for i in range(len(indices)):
        group = np.where(distances[i] < proximity)[0]
        groups.append(group)

    # Remove duplicate groups and sort by size
    unique_groups = []
    for group in groups:
        if set(group) not in unique_groups:
            unique_groups.append(set(group))
    unique_groups.sort(key=len, reverse=True)

    # Compute statistics for each group
    group_stats = {}
    group_indices = []
    for i, group in enumerate(unique_groups):
        group_indices_in_array = indices[list(group)].astype(int)
        group_intensity = intensity_data[group_indices_in_array[:, 0], group_indices_in_array[:, 1]]
        group_max_intensity_index = np.argmax(group_intensity)
        group_max_intensity = group_intensity[group_max_intensity_index]
        group_max_intensity_indices = group_indices_in_array[group_max_intensity_index]
        group_size = len(group)
        group_stats[f"group_{i+1}"] = {"max_intensity": group_max_intensity,
                                       "max_intensity_indices": group_max_intensity_indices,
                                       "size": group_size}
        group_indices.append(group_max_intensity_indices)

    # Convert grouped indices from list of tuples to numpy array
    grouped_indices = np.array(group_indices)

    return group_stats, grouped_indices

# -- Description: Groups coordinate pairs together based on proximity.
def group_within_proximity(array1, array2, proximity):
    # Compute pairwise distances between the two arrays
    distances = cdist(array1, array2)

    # Group elements of array2 that are closer than proximity threshold to any element of array1
    groups = []
    for i in range(len(array1)):
        group = np.where(distances[i] < proximity)[0]
        groups.append(group)

    # Remove duplicates from group indices and sort by size
    unique_groups = []
    for group in groups:
        if set(group) not in unique_groups:
            unique_groups.append(set(group))
    unique_groups.sort(key=len, reverse=True)

    # Compute final array by selecting only unique elements of array2 that are not within proximity of any element of array1
    final_array = []
    for i in range(len(array2)):
        is_within_proximity = False
        for group in unique_groups:
            if i in group:
                is_within_proximity = True
                break
        if not is_within_proximity:
            final_array.append(array2[i])

    return np.array(final_array)

# -- Description: Checks local minima/maxima against one another to eliminate values that fall into bifurcations.
def compare_local_extrema(array1, array2, proximity):
    # Get grouped indices for array1 and array2
    # _, grouped_indices1 = get_group_statistics(intensity_data, array1, proximity)
    # _, grouped_indices2 = get_group_statistics(intensity_data, array2, proximity)
    
    # Filter indices in grouped_indices2 that are within proximity of indices in grouped_indices1
    filtered_indices = []
    for i in range(len(array2)):
        keep = True
        for j in range(len(array1)):
            dist = np.sqrt((array2[i][0]-array1[j][0])**2 + (array2[i][1]-array1[j][1])**2)
            if dist <= proximity:
                keep = False
                break
        if keep:
            filtered_indices.append(array2[i])
    
    # Convert filtered indices to numpy array
    filtered_array2 = np.array(filtered_indices)
    
    return filtered_array2

# -- Description: Takes the local max/inverted local max indices and removes common sets using the symmetric difference method.
def remove_common_sets(array1, array2, return_from_array1=True):
    """
    Takes two input arrays of dimensions (n, 2) and (m, 2), in which n does not necessarily = m in dimensionality. 
    Takes the symmetric difference of these sets and returns the input 'array1' with the matching sets from 'array2' removed.

    Note: No pairs or elements from the input 'array2' will be returned, only the pairs from 'array1' with matching pairs from
    'array2' removed.
    
    Args:
    array1 (np array) of dim (n, 2) : The size of the input array - should be the output locmax_indices.
    array2 (np array) of dim (m, 2) : The size of the input array - should be the output invlocmax_indices."""

    # Convert the arrays to sets of tuples
    set1 = set(map(tuple, array1))
    set2 = set(map(tuple, array2))

    # Find the symmetric difference between the two sets
    diff = set1.symmetric_difference(set2)

    # In set theory, the symmetric difference of two sets is the set of elements 
    # that are in either of the sets but not in their intersection. In other words, 
    # it is the set of elements that belong to one of the sets but not both. 
    # The symmetric difference is denoted by the symbol "∆" or "⊕". For example, 
    # if A = {1, 2, 3} and B = {3, 4, 5}, then A ∆ B = {1, 2, 4, 5}.

    # Find the elements that are in array1 or array2, depending on the option
    if return_from_array1:
        unique_pairs = np.array(list(set1.intersection(diff)))
    else:
        unique_pairs = np.array(list(set2.intersection(diff)))

    return unique_pairs

# -- Description: Finds all peaks using a Monte-Carlo pseudo-random number generator method and set theory to group points based on pairwise distances.
def mcpeakfinder(data_nparray, radius=5, seednum=2, genPoints=2000, pixeldimX=1000, pixeldimY=1000, threshold=0.001, prox1=4, prox2=35, proxextrema=20, proxfinal=10):
    # array = random_gaussians(size=1000, num_gaussians=30, max_intensity=1) # used to generate array of 

    data_nparray = normalize_array(data_nparray)
    locmax_indices, locmin_indices = mc_findpeaks(data_nparray, radius=radius, seednum=seednum, genPoints=genPoints, pixeldimX=pixeldimX, pixeldimY=pixeldimY, threshold=threshold)

    # find the local maxima.
    locmax_stats, locmax_group = get_group_statistics(data_nparray, locmax_indices, proximity=prox1) # fine local gap grouping
    locmax_stats, locmax_group = get_group_statistics(data_nparray, locmax_group, proximity=prox2) # coarse broad gap grouping

    # group the local maxima indices.
    locmin_stats, locmin_group = get_group_statistics(data_nparray, locmin_indices, proximity=prox1) # fine local gap grouping
    locmin_stats, locmin_group = get_group_statistics(data_nparray, locmin_group, proximity=prox2) # coarse broad gap grouping

    # remove the saddle points by finding redundancies in minima and maxima from the sets.
    locmax_group = compare_local_extrema(locmin_group, locmax_group, proximity=proxextrema) # extrema grouping

    # remove common 
    peakindices = remove_common_sets(locmax_group, locmin_group)
    peakindices_stats, peakindices = get_group_statistics(data_nparray, peakindices, proximity=proxfinal) # coarse broad gap grouping
    
    return peakindices, peakindices_stats, data_nparray

# -- Description: Plots the image data along with peak markers.
# def plot_intensity_with_markers(data_nparray, peakindices, plotParams, imgParams, marker='o', color='g'):
def plot_intensity_with_markers(data_nparray, peakindices, marker='o', color='g'):
    
    plt.close('all')

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(data_nparray, cmap='turbo')
    norm = plt.Normalize(vmin=data_nparray.min(), vmax=data_nparray.max()) # Normalize the intensity data
    cmap = get_cmap('turbo') # Define the colormap

    # -----
    # samplename, qxymax, qzmax, xmin, xmax, ymin, ymax, cmin, cmax, cmap, cblabel, cbsize, scaleLog, header, headerfontsize, xfontsize, yfontsize, tickfontsize, autosave, imgdpi, ext, colorscale, fsize, hklcutoff  = plotParams
    # resolutionx, qxymax, qzmax, qzmin = imgParams

    # fig, ax = plt.subplots(figsize = fsize) # Set up the plot
    # cmap = get_cmap('turbo') # Define the colormap
    # extent = -qxymax, qxymax, 0, qzmax

    # ax.imshow(data_nparray,
    #                 #  norm=matplotlib.colors.Normalize(vmin=contrastmin,vmax=contrastmax),
    #                  interpolation='nearest',
    #                  vmax=colorscale*data_nparray.max(), vmin=data_nparray.min(),
    #                  cmap='turbo',
    #                  extent=extent,
    #                  origin='lower',
    #                  aspect='auto')

    # plt.tick_params(axis='both', which='major', labelsize=tickfontsize) # Image tick parameters
    
    # plt.title(header, fontsize = headerfontsize)
    
    # plt.xlabel('$\mathregular{q_{xy}}$ ($\AA^{-1}$)',fontsize=xfontsize)
    # plt.xlim(xmin, xmax)

    # plt.ylabel('$\mathregular{q_z}$ ($\AA^{-1}$)',fontsize=yfontsize)
    # plt.ylim(ymin, ymax)

    # -----
    # Plot each group of indices as a scatter plot with a marker size proportional to group size
    for group in np.split(peakindices, np.unique(peakindices[:, 0], return_index=True)[1][1:]):
        group_intensity = data_nparray[group[:, 0], group[:, 1]]
        # color = cmap(norm(np.mean(group_intensity)))
        size = 10 + 100 * len(group) / len(peakindices)
        ax.scatter(group[:, 1], group[:, 0], s=size, marker=marker, color=color)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax) # Add a colorbar

    plt.show()# Show the plot
    # -----
    return

def plotIntensitywPeaks(intensity_map, peakindices, savepath, Mqxy, Mqz, FMiller, plotParams, imgParams, BPeakParams, size = 2, marker='o', color='g'):

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

    norm = plt.Normalize(vmin=intensity_map.min(), vmax=intensity_map.max()) # Normalize the intensity data

    fig, ax = plt.subplots(figsize=fsize)
    # figure(figsize = (10,8)) # generate figure
    # figure(figsize = fsize) # generate figure
    # colorbar=0.00001
    contrastmin = np.percentile(intensity_map, cmin)
    contrastmax = np.percentile(intensity_map, cmax)

    # extent=[-qxymax, qxymax, 0, qzmax],vmax=colorbar*II1.max(), vmin=II1.min() 
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
    # plt.colorbar(img)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax) # Add a colorbar


    if autosave == True:
        timestamp = time.strftime("%Y%m%d_%H%M%S") # Create a time string, with Year, Month, Day + '_' + Hour, Minute, Second. This is appended to the save_path to give the analyzed/reduced data a unique identifier.
        imgname = samplename + " " + timestamp + ext
        imgpath = os.path.join(savepath, imgname)
        plt.savefig(imgpath, dpi=imgdpi) # save the image if desired
        print("Image Saved: " + str(imgpath))

    # Plot each group of indices as a scatter plot with a marker size proportional to group size
    for group in np.split(peakindices, np.unique(peakindices[:, 0], return_index=True)[1][1:]):
        group_intensity = intensity_map[group[:, 0], group[:, 1]]
        # color = cmap(norm(np.mean(group_intensity)))
        # size = 10 + 100 * len(group) / len(peakindices)
        size = size
        ax.scatter(group[:, 1], group[:, 0], s=size, marker=marker, color=color)

    # # -- Generate the (h k l) index labels.
    # MaxI = 0
    # for h in Mindexrange:
    #     for k in Mindexrange:
    #         for l in Mindexrange:
    #             if Mqxy[h,k,l]<qxymax and Mqz[h,k,l]>qzmin and Mqz[h,k,l]<qzmax:
    #                 MaxI = np.maximum(FMiller[h,k,l], MaxI)
                    
    # for h in Mindexrange:
    #     for k in Mindexrange:
    #         for l in Mindexrange:
    #             if Mqxy[h,k,l]<qxymax and Mqz[h,k,l]>qzmin and Mqz[h,k,l]<qzmax:
    #                 if FMiller[h,k,l] > hklcutoff*MaxI:
    #                     plt.plot(Mqxy[h,k,l], Mqz[h,k,l], 'ko')
    #                     simuposi[isimuposi,0]=Mqxy[h,k,l]
    #                     simuposi[isimuposi,1]=Mqz[h,k,l]
    #                     isimuposi=isimuposi+1
    #                     textstr='('+str(h-hkl_dimension)+','+str(l-hkl_dimension)+','+str(-k+hkl_dimension)+')'
    #                     plt.text(Mqxy[h,k,l]/(2*qxymax)+0.5, (Mqz[h,k,l]-qzmin)/(qzmax-qzmin), textstr, 
    #                              transform=ax.transAxes, fontsize=10,verticalalignment='top',color='k')

    # return simuposi
    return

# -- Description: Creates a map of randomly distributed Gaussians to test the functionality of the peak finder.
def random_gaussians(size=1000, num_gaussians=5, max_intensity=1):
    """
    Generates a 2D numpy array by adding multiple random Gaussian distributions with the same maximum intensity and a narrower width.

    Args:
    size (int): The size of the square array.
    num_gaussians (int): The number of Gaussian distributions to add to the array.
    max_intensity (float): The maximum intensity of each Gaussian.

    Returns:
    numpy.ndarray: A 2D numpy array of shape (size, size) containing the random data.
    """
    # Create a grid of coordinates for the array
    x, y = np.meshgrid(np.arange(size), np.arange(size))

    # Add multiple random Gaussian distributions to the array
    array = np.zeros((size, size))
    for _ in range(num_gaussians):
        x0, y0 = size * np.random.rand(2)
        x_sigma, y_sigma = size * (0.05 + 0.15*np.random.rand(2))
        amplitude = max_intensity / num_gaussians
        array += amplitude * np.exp(-((x-x0)**2/(2*x_sigma**2) + (y-y0)**2/(2*y_sigma**2)))

    return array

# -- Description: Plot formatting for the randomly generated Gaussians.
def plot_gaussians(array, points=None):
    """
    Plots a 2D numpy array using the 'imshow()' function with the 'turbo' colormap, and plots a set of points on top of the plot if provided.

    Args:
    array (numpy.ndarray): A 2D numpy array to plot.
    points (numpy.ndarray): A 2D numpy array of shape (n, 2) containing row and column indices to plot as points overlaid on the plot.

    Returns:
    None
    """
    plt.close('all')
    # Plot the random data using the 'imshow()' function from matplotlib with the 'turbo' colormap
    plt.imshow(array, cmap='turbo')
    plt.colorbar()
    # If points are provided, plot them as red points on the plot
    if points is not None:
        plt.plot(points[:, 1], points[:, 0], 'ro', markersize=2)
        # plt.plot(points[i][:, 1], points[i][:, 0], 'ro', markersize=2)


    # Display the plot
    plt.show()
    return

'''
# def get_group_statistics(data_nparray, indices, proximity):
#     """
#     Groups the given indices by proximity and computes statistics for each group based on the intensity values at those indices.

#     Parameters:
#     intensity_data (np.ndarray): A numpy array of shape (n, m) containing the intensity values.
#     indices (np.ndarray): A numpy array of shape (n, 2) containing the indices to group.
#     proximity (float): The maximum distance between two indices to be considered part of the same group.

#     Returns:
#     group_stats (list): A list of dictionaries, where each dictionary contains the indices of the maximum intensity value and other statistics for that group.
#     """

#     n = len(indices)
#     groups = []

#     # Iterate over each index and find its neighbors within the proximity threshold
#     for i in range(n):
#         group = [i]
#         for j in range(i+1, n):
#             if np.linalg.norm(indices[i] - indices[j]) < proximity:
#                 group.append(j)
#         groups.append(group)

#     # Merge overlapping groups
#     merged_groups = []
#     while groups:
#         group = groups.pop(0)
#         for other_group in groups[:]:
#             if any(index in other_group for index in group):
#                 group.extend(other_group)
#                 groups.remove(other_group)
#         merged_groups.append(list(set(group)))

#     # Compute statistics for each group
#     group_stats = []
#     for group in merged_groups:
#         group_indices = indices[group]
#         group_intensity = data_nparray[tuple(group_indices.T)]
#         max_index = group_intensity.argmax()
#         max_intensity = group_intensity[max_index]
#         mean_intensity = group_intensity.mean()
#         num_points = len(group)
#         intensity_var = group_intensity.var()

#         stats_dict = {
#             'indices': group_indices[max_index],
#             'max_intensity': max_intensity,
#             'mean_intensity': mean_intensity,
#             'num_points': num_points,
#             'intensity_var': intensity_var
#         }

#         group_stats.append(stats_dict)

#     return group_stats
'''

'''
# -- Description:
# def phid1(theta00,data,exp_peak_postions):
#     thetax=np.pi/2*0
#     thetay=np.pi/2*theta00
#     hkl_dimension=5

#     # sigma_theta, if you need a small number for single crystal, input~0.01, if you need infinity, input~1000
#     sigma1=0.03

#     # sigma_phi
#     sigma2=1000

#     # sigma_r, use this to tune the peak linewidth
#     sigma3=0.03

#     # settings for imagine plane
#     resolutionx=300
#     qxymax=2
#     qzmax=2
#     qzmin=0.2

#     resolutionz=int(resolutionx/qxymax*qzmax)
#     gridx,gridz=np.meshgrid(np.linspace(-qxymax,qxymax,resolutionx),np.linspace(qzmin,qzmax,resolutionz))
#     dirr = ''
#     filename = 'S-NPB'
#     address = dirr + filename +'.vasp'
#     a1,a2,a3,positions=diff.read_poscar(address) # example (graphite)
#     print(thetay)
#     Bpeaks,Mqxy,Mqz,I_miller = diff.Bragg_peaks(a1,a2,a3,positions,thetax,thetay,hkl_dimension)
#     colorbar=0.9

#     fsize=(30,30)
#     qrange=[-qxymax, qxymax, qzmin, qzmax]
#     Mindexrange=np.linspace(0,hkl_dimension,hkl_dimension+1)
#     Mindexrange=Mindexrange.astype('int')

#     Cutoff_I=0.01
#     simuposi=Mindexing(data,colorbar,fsize,qrange,Mindexrange,I_miller,Cutoff_I,Mqxy,Mqz)
#     plt.plot(exp_peak_postions[:,0],exp_peak_postions[:,1], 'go')
    
#     return simuposi
'''

'''
# def mcpointgen(seednum=1, genPoints=1000, pixeldimX=1000, pixeldimY=1000):
#     """Monte-Carlo Style Pseudo-Random Point Generator: Seeded pseudo-random number generator for use with 2D Monte-Carlo peak finder. 
#     Takes the following inputs:
#     seednum : int
#         seed value used to produce the random numbers.
#     genPoints : int
#         generated points, number of random points generated on the rangeX x rangeY grid.
#     pixeldimX : int
#         pixel dimensions in X over data set
#     pixeldimY : int 
#         pixel dimensions in Y over data set
#     """

    # seednum = int(seednum)
    # seed(seednum) # seed the random number generator
    
    # genPoints = int(genPoints)
    # np_mcrand = np.zeros([genPoints, 2]) # initialize empty monte carlo array

    # i = 0
    # for i in range (0, genPoints):
    #     randX = random.randint(0, pixeldimX-1) # generate random X 
    #     randY = random.randint(0, pixeldimY-1)

    #     np_mcrand[i,0] = int(randX)
    #     np_mcrand[i,1] = int(randY)

    # np_mcrand = mcpointgen(seednum=1, genPoints=1000, pixeldimX=1000, pixeldimY=1000)
    # # print (np_mcrand)
    # index = 0
    # index = int(index)
    # print (list(np.shape(np_mcrand))[index])
    # return np_mcrand
    '''