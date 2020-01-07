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
    basedir = './'
    
    imagedir = basedir + 'images/'
    positionsdir = basedir + 'positions/' 
    trajectorydir = basedir + 'trajectories/'

    # Allocation
    flist = np.empty(0,dtype=int)

    # Find all position files (.dat)
    for (dirpath, dirnames, filenames) in os.walk(positionsdir):
        f = np.asarray(fnmatch.filter(filenames,'*.dat*'))
        flist = np.append(flist,f)

    if np.size(flist) == 0:
        print('Searched through ',positionsdir,'\nNo measurement files specified, quitting.')
        quit()
    if verbose: print('User inputs read')

    #############################################
    sortedlist = [int(x.replace('.dat','').replace('.','')) for x in flist]
    sortedlist = np.argsort(sortedlist)
    flistsorted = list(flist[sortedlist])
    
    # Create directories
    if not os.path.exists(measurementdir+'trajectories'):
        os.makedirs(measurementdir+'trajectories')

    if verbose: print('Particle trajectories stored in folder',trajectorydir)

if __name__ == "__main__":  
    sortfiles()
