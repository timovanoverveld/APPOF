#!/usr/bin/env python
# coding: utf-8

# Timo van Overveld, 2020

"""Obtain radial, angular and 2D distribution functions of input data"""
# Input: particle position files (.dat)
# Output: RDF, ADF, 2DDF

# Imports
import numpy as np
import json
import os
import argparse
import fnmatch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

############################################################################
def distributions():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store_true', help='File')
    parser.add_argument('-r', action='store_true', help='Radial distribution function values')
    parser.add_argument('-p', action='store_true', help='Enable plotting')
    args = parser.parse_args()
   
    print('Done')

if __name__ == "__main__":  
    distributions()
