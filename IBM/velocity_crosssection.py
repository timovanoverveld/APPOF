#!/usr/bin/env python
# coding: utf-8

# Timo van Overveld, 2020

"""Visualize IBM velocity cross section"""
# Input: 
# Output: .png images

# Imports
import numpy as np
import os
import csv
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

############################################################################
def velocity_crosssection():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='File')
    parser.add_argument('-d', type=str, help='Directory')
    args = parser.parse_args()
   
    #Initial data
    #TODO make a list of xyz//NUMBER//.txt files
    
    if args.f:
        with open(args.f, newline='') as txt:
            reader = csv.reader(txt)
            variables = next(reader)

        variables[0] = variables[0][-2]

        print('Variables found:', variables)

        data = np.loadtxt(args.f,skiprows=2)

    # User inputs
    x,y,z = [int(x) for x in input('Give X, Y and Z: ').split()]
    
    #Plotting
    fig = plt.figure(figsize=(12,12))
  
    xplot = np.unique(data[:,x])
    yplot = np.unique(data[:,y])
   
    X, Y = np.meshgrid(xplot,yplot)
   
    Z = np.reshape(data[:,z], (np.size(yplot),np.size(xplot)))
    
    plt.contourf(X,Y,Z,levels=100)
    plt.colorbar()
    plt.title(str(variables[z]))
    plt.grid()
    plt.xlabel(variables[x])
    plt.ylabel(variables[y])

    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()

    print('Done')

if __name__ == "__main__":  
    velocity_crosssection()
