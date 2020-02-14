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
    parser.add_argument('f', type=str, help='File')
#    parser.add_argument('-d', type=str, help='Directory')
    parser.add_argument('-s', action='store_true', help='Save figure')
    parser.add_argument('-l', action='store_true', help='Plot over line')
    parser.add_argument('-n', type=int, nargs='+', help='Which variables to use')
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
    if not args.n:
        x,y,z = [int(x) for x in input('Give X, Y and Z: ').split()]
    else:
        x,y,z = args.n
    
    #Plotting
    fig = plt.figure(figsize=(12,12))
  
    if args.l:
        size = np.shape(data)
        _, idx = np.unique(data[:,x],return_index=True)

        xplot = data[idx,x]
        zplot = data[idx,z]
        plt.plot(xplot,zplot,label='IBM')
        #plt.plot(xplot,np.max(zplot)*4*xplot*(1.-xplot),label='Planar Poiseuille profile')
        
        plt.grid()
        plt.legend()
        plt.xlabel(variables[x])
        plt.ylabel(variables[z])
        plt.ylim(-3e-1,3e-1)

    else:
        xplot = np.unique(data[:,x])
        yplot = np.unique(data[:,y])
   
        X, Y = np.meshgrid(xplot,yplot)
   
        Z = np.reshape(data[:,z], (np.size(yplot),np.size(xplot)))
        
        plt.contourf(X,Y,Z,levels=100)
        plt.colorbar()
        try: 
            plt.title(str(variables[z])+' at timestep '+str(int(args.f[-11:-4])))
        except:
            plt.title(str(variables[z]))
            pass
        plt.grid()
        plt.xlabel(variables[x])
        plt.ylabel(variables[y])

    if args.s:
        plt.savefig(args.f[:-4]+'.png')
    elif not args.s:
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close()

    print('Done')

if __name__ == "__main__":  
    velocity_crosssection()
