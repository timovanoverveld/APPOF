#!/usr/bin/env python
# coding: utf-8

# Timo van Overveld, 2020

"""Sort the files (image, positions, trajectories) into their respective directories."""

# Imports
import numpy as np
import json
import os
import argparse
import fnmatch
import matplotlib.pyplot as plt

############################################################################
def sortfiles():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', action='store_true', help='Sort subdirectories')
    args = parser.parse_args()
   
    # Directories
    basedir = os.getcwd()
    
    imagedir = basedir + '/images/'
    positionsdir = basedir + '/positions/' 
    trajectorydir = basedir + '/trajectories/'


#############################################################################
    #Check if we are in a measurement directory (should include a .strm file)
    if any(File.endswith('strm') for File in os.listdir('.')):

        # Check for directories
        if not os.path.exists(imagedir):
            os.makedirs(imagedir)
        if not os.path.exists(positionsdir):
            os.makedirs(positionsdir)
        if not os.path.exists(trajectorydir):
            os.makedirs(trajectorydir)
        
        # Put loose files in correct directories

        # Allocation
        ilist = np.empty(0,dtype=int)
        plist = np.empty(0,dtype=int)
        tlist = np.empty(0,dtype=int)

        # Find all images files (.tif)
        for (dirpath, dirnames, filenames) in os.walk(basedir):
            i = np.asarray(fnmatch.filter(filenames,'*.tif*'))
            ilist = np.append(ilist,i)
            
            p = np.asarray(fnmatch.filter(filenames,'*[0-9].dat'))
            plist = np.append(plist,p)
            
            t = np.asarray(fnmatch.filter(filenames,'*[x,y].dat'))
            tlist = np.append(tlist,t)
            break


        if np.size(ilist) > 0:
            for i in ilist:
                os.rename(i,imagedir+i)
        if np.size(plist) > 0:
            for i in plist:
                os.rename(i,positionsdir+i)
        if np.size(tlist) > 0:
            for i in tlist:
                os.rename(i,trajectorydir+i)
        
        print('Done')
        quit()
    
    else:
        print('Not in a measurement directory')
        quit()
        

if __name__ == "__main__":  
    sortfiles()
