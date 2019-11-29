#!/usr/bin/env python
# coding: utf-8

# Timo van Overveld, 2019

"""A unittest to check whether the camera position is consistently found"""
# Input: settings_cameraposition.txt, H0.tif, H7.tif
# Output: Camera position

# Imports
import numpy as np
import json
import unittest
from images2positions_functions import *

############################################################################

class TestCalibration(unittest.TestCase):

    def test_cameraposition(self):
        # Reading settings file
        f = open('settings_cameraposition.txt', 'r')
        settings = json.loads(f.read())
       
        # Check if settings exists
        self.assertTrue(settings)

        # refraction indices
        nair   = settings['nair']
        nwater = settings['nwater']
        n      = nair/nwater

        # Line spacing (real world) [m]
        linespacing = settings['linespacing']
        linespacingpx = settings['linespacingpx']

        # Cropping bounds (top, bottom, left, right)
        bounds = settings['bounds']

        # Central pixels to exclude
        centerpx = settings['centerpx']

        # Fitting order (C*x^order+...)
        warpingorder = settings['warpingorder']

        ################################### 
        # Warping
        calAlist = 'H0.tif' 

        # Camera position
        calBlist = ['H0.tif','H7.tif']
   
        Hlist = np.asarray([0,7])
        #############################################

        # Camera warping
        # Linking the pixels to the real world coordinates, because the camera warps the image. This script gives the transformation from pixels to real world coordinates for a water height of 0.
       
        #Obtain line positions
        xpix, Hreal = pixHlist(calAlist,Hlist,bounds=bounds,centerpx=centerpx,linespacingpx=linespacingpx)
        xreal, Nlines = clusterlines(xpix,linespacing,Nlines=17)
        
        # Fit the warping function through the data
        pix2realx = pixrealfit(xpix, xreal, warpingorder)
        ###########################

        # Camera position
        # Finding the camera position by using known flat water heights.

        xpix, H = pixHlist(calBlist,Hlist,bounds=bounds,centerpx=centerpx,linespacingpx=linespacingpx)
        xprojected = pix2realx(xpix)
        xreal, Nlines = clusterlines(xpix,Nlines=Nlines,linespacing=linespacing)
     
        xc, Hc = cameraposition(xprojected,xreal,H,n)

        #################################
        # xc should be 0.35000
        # Hc should be 1.82484
        self.assertAlmostEqual(xc[0],0.35000,places=5)
        self.assertAlmostEqual(Hc[0],1.82484,places=5)

if __name__ == '__main__':
    unittest.main()
