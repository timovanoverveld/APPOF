# Function definitions

# Imports
import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy.optimize as optimization
from scipy.signal import find_peaks
from scipy import ndimage
import os
import fnmatch
import json

##################
# Image handling #
##################

# Read the image from the given path, open it and crop it.
def readcropimage(path,bitdepth=8):
    #Read image
    if bitdepth==8:
        image = cv2.imread(path,cv2.CV_8UC1)
        image = image*16
    elif bitdepth==16:
        image = cv2.imread(path,cv2.CV_16UC1)

    # Crop image
    image = image[bounds[0]:-1-bounds[1],bounds[2]:-1-bounds[3]]
    return image


# Threshold the input
def thresholdimage(image,bordered=True):
    kernel = np.ones((3,3),np.uint8)
    _, threshold = cv2.threshold(image,thresholdvalue,255,cv2.THRESH_BINARY)
    threshold = cv2.morphologyEx(threshold,cv2.MORPH_OPEN,kernel,iterations=1)

    # Invert the image
    threshold = ~np.uint8(threshold)

    # Add a border of zeros
    if bordered:
        threshold = np.pad(threshold, pad_width=1, mode='constant', constant_values=0)

    return threshold


# Background
def background(image,iters=1):
    kernel = np.ones((3,3),np.uint8)
    background = cv2.dilate(image,kernel,iterations=iters)
    return background


# Foreground
def foreground(image,dttype='L1'):
    # Calculate the distance transform
    if dttype=='L1': # as (|x1-x2|+|y1-y2|)
        disttrans = cv2.distanceTransform(image,cv2.DIST_L1,0)
    elif dttype=='C': # as max(|x1-x2|,|y1-y2|)
        disttrans = cv2.distanceTransform(image,cv2.DIST_C,0)

    # disttrans is used for comparisons and obtaining values of the distance transform
    # centers is defined as disttrans, but in the coming loops sections are altered and put to zero.
    centers = disttrans*1    #*1 otherwise if centers is adjusted, so is disttrans

    # Loop over the values in distancetransform, starting with the highest value
    for k in range(int(np.max(disttrans)),0,-1):
        # markers contains the patches of same value in the distance transform
        q = np.where(disttrans==k,1,0)
        q = np.uint8(q)
        _, markers = cv2.connectedComponents(q)

        # Loop over the patches
        for qi in range(1,np.max(markers)+1,1):
            px,py = np.where(markers == qi)

            # Loop over the pixels in the patch and check if there are any higher neighbours. If yes, then it is not a local maximum
            for i in range(0,np.size(px),1):

                # For all neighbours of the pixel, check if there are any values higher than the pixel value (k)
                neighbours = disttrans[px[i]-1:px[i]+2,py[i]-1:py[i]+2]
                if np.any(neighbours>k):
                    centers[px,py] = 0
#                     break

    foreground = np.uint8(centers>0)

#     foreground = cv2.dilate(foreground,np.ones((3,3),np.uint8),iterations=1)
    return foreground

########################################################################

######################
# Object recognition #
######################

def findlines(image, centerpx, binarize=False, gaussianfilter=True):
    # Extract part of the image, remove center for better statistics
    linearea = np.concatenate((image[0:centerpx[0],:],image[centerpx[1]:,:]))

    # Threshold the extracted part of the image to obtain the lines
    threshold = cv2.adaptiveThreshold(linearea,1,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,10)
#     threshold = cv2.adaptiveThreshold(linearea,1,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,25,10)

    # Remove some additional noise caused by shadows
    threshold = cv2.morphologyEx(threshold,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8),iterations=1)
    threshold = ~np.uint8(threshold)

    # Average and normalize to obtain 1D data (Since particles may overlap the lines)
    averages = (np.average(threshold,axis=0)-np.min(threshold))/(np.max(threshold)-np.min(threshold))

    # Threshold the 1D data to obtain binary information
    if binarize == True:
         averages = cv2.threshold(averages,0.1,1,cv2.THRESH_BINARY)[1]

    if gaussianfilter == True:
        averages = ndimage.gaussian_filter1d(averages,sigma=1)

    # Fit lines with a fixed minimum separation distance
    lines, _ = find_peaks(averages, distance=linespacingpx)

    return lines


def findparticles(image):
    # Threshold the image
    threshold = thresholdimage(image,bordered=True)

    # Compute element of picture that are surely background or foreground
    sure_background = background(threshold)
    sure_foreground = foreground(threshold)

    _, thresholdcomponents = cv2.connectedComponents(threshold)

    # Calculate the unknown area between foreground and background
    unknown = cv2.subtract(sure_background,sure_foreground)

    # Setup the markers based on the foreground
    _, markers = cv2.connectedComponents(sure_foreground)
    markers += 1
    markers[unknown==255] = 0

    # Setup the image to be used in the watershed transform
    imagewatershed = cv2.cvtColor(np.uint8(threshold),cv2.COLOR_GRAY2RGB)

    # Do the watershed transform
    markers = cv2.watershed(imagewatershed,markers)

    # Remove the border that was added during the thresholding
    markers_noborder = markers[1:-1,1:-1]

    return markers_noborder


def removeparticles(image,markers,dilate=True,dilatesize=11):
    # Create mask to use in filtering of particles
    mask = np.where(markers>1,1,0)

    # Dilate the mask such that shadows can be removed as well
    if dilate:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilatesize,dilatesize))
        mask = cv2.dilate(np.uint8(mask),dilate_kernel,iterations=1)

    image_noparticles = np.where(mask==1,np.mean(image),image)

    return image_noparticles

def particlesfilter(image,markers):
    #Number of markers (= detected particles+2)
    N = np.max(markers)
    positions = np.empty((N-1,2),dtype=float)

    # Size distribution of markers
    sizes = [np.sum(np.where(markers==j,1,0)) for j in range(2,N,1)]
    meansize = np.mean(sizes)

    # Maximum number of pixels to count as 1 particle
    maxpx = meansize*1.7

    # Minimum number of pixels required to count as particle
    minpx = meansize*0.5

    # New marker set that is appended
    newmarkersset = np.zeros(np.shape(markers),dtype=int)

    imagefiltered = np.zeros(np.shape(image),dtype=int)
#     pointsinmarkers = [[],[]]

    # Loop over the particles, 0 is nothing, 1 is background, so start from 2
    for i in range(2,N+1,1):
        # Create mask to use in filtering of particles
        mask = np.where(markers==i,1,0)

        # If particle < minpx skip it, if >maxpx split it
        if minpx <= np.sum(mask) <= maxpx:
#             print(np.max(newmarkersset))
            newmarkersset[mask!=0] += np.max(newmarkersset)+1
        elif np.sum(mask) > maxpx:
            if verbose: print('Particle',i,'too large:',np.sum(mask),'px. Will be split in probably',int(round(np.sum(mask)/meansize)),'particles.')

            particle = np.where(mask==1,image,0)

            # Coordinates of pixels in (blob of) particles
            coordpx = np.argwhere(particle!=0)
            px = coordpx[:,0]
            py = coordpx[:,1]

            imagefiltered[px,py] = 1
#             pointsinmarkers = np.append(pointsinmarkers,[px,py],axis=1)

            weights = particle[px,py]

            center = [np.average(py,weights=weights),np.average(px,weights=weights)]
            centerintx = int(center[1])
            centerinty = int(center[0])

            pxx = np.linspace(centerintx-1,centerintx+1,3,dtype=int)
            pyy = np.linspace(centerinty-1,centerinty+1,3,dtype=int)
            Px, Py = np.meshgrid(pxx,pyy)
#             image[Px,Py] = np.max(image)

    return imagefiltered #image, image2


def particlepositions(image,markers,weightedaverage=False):
    #Number of markers (=particles+2)
    N = np.max(markers)
    positions = np.empty((N-1,2),dtype=float)

    # Size distribution of markers
    sizes = [np.sum(np.where(markers==j,1,0)) for j in range(2,N,1)]
    meansize = np.mean(sizes)

    # Maximum number of pixels to count as 1 particle
    maxpx = meansize*1.7

    # Minimum number of pixels required to count as particle
    minpx = 10

    # Loop over the particles, 0 is nothing, 1 is background, so start from 2
    for i in range(2,N+1,1):
        # Create mask to use in filtering of particles
        mask = np.where(markers==i,1,0)

        particle = np.where(mask==1,image,0)

        # Coordinates of pixels in particle
        coordpx = np.argwhere(particle!=0)
        px = coordpx[:,0]
        py = coordpx[:,1]

        # Weighted average
        if weightedaverage:
            weights = particle[px,py]
        # Normal average
        else:
            weights = np.ones(np.size(px))

        center = [np.average(py,weights=weights),np.average(px,weights=weights)]

        positions[i-2,:] = center

    return positions

def correctmarkers(image,markers):
    markerscorrection = particlesfilter(image,markers)

    threshold = thresholdimage(image,bordered=False)
    _, thresholdcomponents = cv2.connectedComponents(threshold)

    imagecorrected = image*1

    for i in range(1,np.max(thresholdcomponents),1):
        a = np.where(thresholdcomponents==i,1,0)
    #     b = a*pointsinmarker
        b = a*markerscorrection
        if np.sum(a*b)>0:
            c = foreground(np.uint8(a))
            d = cv2.dilate(np.uint8(c*markerscorrection),np.ones((3,3),np.uint8),iterations=1)
            imagecorrected[d==1] = np.max(imagecorrected)
    markerscorrected = findparticles(imagecorrected)

    return imagecorrected, markerscorrected

############################################################################

##########################
# Coordinate conversions #
##########################

# For all files in a list, call pixrealH and append
def pixHlist(filelist):
    #Allocate empty arrays
    xpix  = np.empty(0,dtype=float)
    Hreal = np.empty(0,dtype=float)

    #Loop over files
    for file in filelist:
        # Obtain the values for a single file, single side
        xp, Hr = pixrealH(file,filelist.index(file))

        # Addvalues to the lists
        xpix  = np.append(xpix, xp)
        Hreal = np.append(Hreal,Hr)

    return xpix, Hreal

# Read lines in pixel values, construct complementary arrays of real line positions and (flat) water height
def pixrealH(file, index=0):
    # Read and filter image
    image = readcropimage(file)

    # Extract pixel values from image
    xpix = findlines(image,centerpx)

    # Water height per found line
    H  = Hlist[index]
    Hreal = float(H)/100*np.ones(np.size(xpix))

    return xpix, Hreal

#Calculate the real line coordinates by clustering the pixel values
def clusterlines(xpix,Nlines=0):
    if Nlines == 0:
        sortedgrad = np.gradient(np.sort(xpix))
        xpeaks, __ = find_peaks(sortedgrad,distance=10)
        Nlines = np.size(xpeaks)

    _, bins = np.histogram(xpix,bins=Nlines,range=(0,1600))

    xreal = np.zeros(np.size(xpix))
    for i in range(0,Nlines,1):
        a = np.where((bins[i]<=xpix) & (xpix<=bins[i+1]))
        xreal[a] = linespacing*i

    return xreal, Nlines

# Fit the camera position using the projected and real positions
def cameraposition(xprojected,xreal,H):
    xin = np.vstack((xprojected,H))
    popt, pcov = optimization.curve_fit(func_flatsurf, xin, xreal)
    perr = np.sqrt(np.diag(pcov))

    xc = [popt[0],perr[0]]
    Hc = [popt[1],perr[1]]
    return xc, Hc


##############################################################################

####################################
# Fitting and function definitions #
####################################

# Fit a polynomial through the data and pass it as a callable function pix2real()
def pixrealfit(xpix, xreal, order):

    # Fit functions through the data
    z = np.polyfit(xpix,xreal,order)
    pix2real = np.poly1d(z)

    return pix2real


# Construct a polynomial Hpolynomial that represents the water surface shape
def Hpolynomial(xl,xp, Hmean=0.1):
    # Choose a set of heights, 1 value for each line on the bottom
    H  = Hmean*np.ones(np.size(xl))

    maxiterations = 5
    tolerance = 1e-8

    #Choose an H, calculate Hpolynomial from that, and choose Hpolynomial(xw) as new value for H and repeat
    for i in range(0,maxiterations,1):
        # xw and H' are both dependent and fixed once H is chosen.
        xw = H2xw(H,xl,xp)
        Hp = H2Hp(H,xl,xp)

        # Find H(x) by fitting
        Hpolynomial = fitH(xw,H,Hp)

        difference = abs(Hpolynomial(xw)-H)
        if np.all(difference <= tolerance):
            break
        else:
            H = Hpolynomial(xw)

    return Hpolynomial

#Convert H to xw
def H2xw(H,xl,xp):
    alpha = (xc[0]-xp)/Hc[0]
    xw = xp + alpha*H
    return xw

# Convert H to H'
def H2Hp(H,xl,xp):
    Hp = np.empty(np.size(H),dtype=float)
    for i in range(0,np.size(Hp),1):
        # Creating 3 valued input data: (H,xl,xp)
        data = (H[i],xl[i],xp[i])
        solution = optimization.root(F,0.0,args=data,tol=1e-6)
        Hp[i] = solution.x
    return Hp


# Function for flat surface
def func_flatsurf(xin, xc, Hc):
    x = xin[0,:]
    H = xin[1,:]
    y = x + H*(xc-x)/Hc * (1-n/np.sqrt(1+(1-n**2) * ((xc-x)/Hc)**2 ))
    return y


# The complicated function linking H and H' to xl and xp
def F(Hp,*data):
    H, xl, xp = data
    alpha = (xc[0]-xp)/Hc[0]
    A = (   n*(Hp+alpha) - Hp*np.sqrt((Hp**2+1)*(alpha**2+1)-n**2*(Hp+alpha)**2))
    B = (Hp*n*(Hp+alpha) +    np.sqrt((Hp**2+1)*(alpha**2+1)-n**2*(Hp+alpha)**2))
    f = (xp-xl) + H * (alpha - A/B)
    return f


# With the info on xw, H and H', fit a polynomial of arbitrary order through them
def fitH(xw,H,Hp,order=5):
    N = np.size(H) # Number of lines

    #Left hand side (matrix A)
    A = np.zeros((2*N,order))

    for i in range(0,N,1): # Loop over lines
        for j in range(0,order,1): # Loop over number of variables in polynomial
            #Points
            A[i,-j-1] = xw[i]**j

            #Derivatives
            A[N+i,-j-1] = j*xw[i]**(j-1)

    # Right hand side (Vector b)
    b = np.concatenate((H,Hp))

    # solution = np.linalg.solve(A,b)
    coeffs = np.linalg.lstsq(A,b,rcond=None) #rcond just to silence a FutureWarning
    polynomial = np.poly1d(coeffs[0])

    return polynomial

# Calculate xreal based on H (polynomial function) and xprojected (discrete data)
def projected2real(xp,H,Hp):
    #Calculate intersection point
    data = (xp,H)
    xw = optimization.root(intersection,xp,args=data)
    xw = xw.x

    #Calculate real positions
    Hxw  = H(xw)
    Hpxw = Hp(xw)
    xreal = xprojected2xreal(xp,Hxw,Hpxw)
    return xreal

# Calculate the root, which corresponds to the intersection point of sightline and water surface
def intersection(x,*data):
    xp, H = data
    f = H(x) - Hc[0]/(xc[0]-xp)*(x-xp)
    return f

# Computing xreal based on xprojected, H(xw) and H'(xw)
def xprojected2xreal(xp,H,Hp):
    alpha = (xc[0]-xp)/Hc[0]
    A = (   n*(Hp+alpha) - Hp*np.sqrt((Hp**2+1)*(alpha**2+1)-n**2*(Hp+alpha)**2))
    B = (Hp*n*(Hp+alpha) +    np.sqrt((Hp**2+1)*(alpha**2+1)-n**2*(Hp+alpha)**2))
    xreal = xp + H * (alpha - A/B)
    return xreal
