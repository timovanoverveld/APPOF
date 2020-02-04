#!/usr/bin/env python
# coding: utf-8

# Timo van Overveld, 2020

"""Visualize IBM channel profiles: velocities and pressures (mean + RMS)"""
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
def channel_profiles():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='File')
    parser.add_argument('-d', type=str, help='Directory')
    args = parser.parse_args()
   
    #Initial data

    if args.d:
        data = np.loadtxt(args.d+'chanprofile')
        z  = data[:,0] #Coordinate from 0 to Lz approximately
        
        z2 = data[:,1] #Unknown quantity

        um = data[:,2] #Mean velocities
        vm = data[:,3]
        wm = data[:,4]

        pm = data[:,5] #Mean pressure

        ur = data[:,6] #RMS velocities
        vr = data[:,7] 
        wr = data[:,8]

        pr = data[:,9] #RMS pressure


    plt.figure(figsize=(12,12))
    
    plt.plot(z,um,label='u_m')
    plt.plot(z,vm,label='v_m')
    plt.plot(z,wm,label='w_m')
    
    plt.plot(z,ur,label='u_r')
    plt.plot(z,vr,label='v_r')
    plt.plot(z,wr,label='w_r')

    plt.plot(z,pm,label='p_m')
    plt.plot(z,pr,label='p_r')
    
    plt.grid()
    plt.xlabel('z along channel')
    plt.ylabel('velocity')
    plt.legend()

    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()

    print('Done')

if __name__ == "__main__":  
    channel_profiles()
