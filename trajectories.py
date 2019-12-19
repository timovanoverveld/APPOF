#!/usr/bin/env python
# coding: utf-8

# Timo van Overveld, 2019

"""Obtain the particle trajectories by coupling different particle data files."""
# Input: particle position files (.dat)
# Output: particle positions sorted

# Imports
import numpy as np
#import json
#import os
import argparse
#import fnmatch
#from images2positions_functions import *


############################################################################
def trajectories():
    
    print('Hello world')


if __name__ == "__main__":  
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', action='store_true', help='Set verbosity')
    args = parser.parse_args()

    trajectories()


