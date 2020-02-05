#!/usr/bin/env python
# coding: utf-8

# Timo van Overveld, 2020

"""Visualize IBM velocity cross section trough particles"""
# Input: 
# Output: .png images

# Imports
import numpy as np
import os
import csv
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
import warnings
warnings.filterwarnings("ignore")

############################################################################
def particle_crosssection():
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

    # Variables are always x, y, xc, yc, vxc, vyc
    x = data[:,0]
    y = data[:,1]
    vx = data[:,4]
    vy = data[:,5]

    #Triangulation
    triang = tri.Triangulation(x,y)

    #Plotting
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(211)
    tcf = ax.tricontourf(triang,vx,levels=100)
    fig.colorbar(tcf)
    ax.set_title(variables[-2])
    
    plt.xlabel(variables[0])
    plt.ylabel(variables[1])

    ax = fig.add_subplot(212)
    tcf = ax.tricontourf(triang,vy,levels=100)
    fig.colorbar(tcf)
    ax.set_title(variables[-1])
    
    plt.xlabel(variables[0])
    plt.ylabel(variables[1])
    
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()

    print('Done')

if __name__ == "__main__":  
    particle_crosssection()
