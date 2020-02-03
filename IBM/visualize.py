#!/usr/bin/env python
# coding: utf-8

# Timo van Overveld, 2020

"""Visualize IBM output"""
# Input: 
# Output: .png images

# Imports
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

############################################################################
def visualize():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='File')
    parser.add_argument('-d', type=str, help='Directory')
    args = parser.parse_args()
   
    #Initial data
   
    if args.d:
        data = np.loadtxt(args.d+'position_wall_segments.txt')
        iw = data[:,0]
        xw = data[:,1]
        yw = data[:,2]

        data = np.loadtxt(args.d+'position_spheres.txt')
        
        try:
            i  = data[:,0]
            th = data[:,1]
            ph = data[:,2]
            x  = data[:,3]
            y  = data[:,4]
            z  = data[:,5]
        except:
            i  = data[0]
            th = data[1]
            ph = data[2]
            x  = data[3]
            y  = data[4]
            z  = data[5]

    plt.figure(figsize=(12,12))
    plt.plot(xw,yw)
    
    plt.scatter(x,y)

    plt.axes().set_aspect('equal')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')

    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()

    # Loop over particles
    plt.figure(figsize=(12,12))
    for i in range(1,np.size(i)+1,1):
        string = args.d+'particlechar'+str(i).zfill(3)
        print(string)
        data = np.loadtxt(string)

        j = data[:,0]
        time = data[:,1]
    
        # pos = (rc_new_tot-rc_wall)/lref
        x = data[:,2]
        y = data[:,3]
        z = data[:,4]
        
        r = data[:,5]
        th = data[:,6]
        phi = data[:,7]
        
        # Velocities
        u = data[:,8]
        v = data[:,9]
        w = data[:,10]


        plt.plot(x,y)
        plt.scatter(x,y)

    plt.axes().set_aspect('equal')
    plt.grid()
    tplot = np.linspace(0,2*np.pi,100,'k')
    plt.plot(np.cos(tplot),np.sin(tplot))
    plt.xlabel('x')
    plt.ylabel('y')

    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()

    print('Done')

if __name__ == "__main__":  
    visualize()
