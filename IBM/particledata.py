#!/usr/bin/env python
# coding: utf-8

# Timo van Overveld, 2020

"""Visualize IBM particle data (as a function of time)"""
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
def particledata():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('f', type=str, help='File')
    parser.add_argument('-f', type=str, help='File')
    parser.add_argument('-d', type=str, help='Directory')
    parser.add_argument('-n', type=int, help='Which variables to plot')
    args = parser.parse_args()
   
    #Initial data

    if args.d:
        data = np.loadtxt(args.d+'particlechar001')
    elif args.f:
        data = np.loadtxt(args.f)
    
    names = ['Timestep','Time','x_c','y_c','z_c','r_c','theta_c','phi_c','u_c','v_c','w_c','omega_x','omega_y','omega_z','omega_theta','force_x','force_y','force_z','torque_x','torque_y','torque_z','torque_theta','force_coll_x','force_coll_y','force_coll_z']

    i  = data[:,0] #Time step istap
    t  = data[:,1] #time/tref (Characteristic timescale Tref=lref/uref)

    xc = data[:,2] #Center position [(x,y,z)-(xcwall,ycwall,0)]/lref
    yc = data[:,3]
    zc = data[:,4]

    rc  = data[:,5] #Center position (r, theta, phi)/(lref,1,1)
    thc = data[:,6]
    phc = data[:,7]

    uc = data[:,8] #Center velocities/Uref
    vc = data[:,9]
    wc = data[:,10]

    wx = data[:,11] #Angular velocities omega * tref
    wy = data[:,12]
    wz = data[:,13]

    wth = data[:,14] #Angular velocity in theta? * tref     

    fx = data[:,15] # Total lagrangian force fx / (uref*lref)^2
    fy = data[:,16]
    fz = data[:,17]

    tx = data[:,18] # Torque / (uref^2*lref)
    ty = data[:,19] 
    tz = data[:,20] 

    tth = data[:,21] # Torque in theta? / (uref^2*lref)

    cfx = data[:,22] # Collision force / (uref^2*lref)
    cfy = data[:,23]
    cfz = data[:,24]
    
    #Plotting 
    plt.figure(figsize=(12,8))
    
    if args.n == 0:
        plt.plot(t,xc,label='xc')
        plt.plot(t,yc,label='yc')
        plt.plot(t,zc,label='zc')
        plt.ylabel('Position')
    
    if args.n == 1:
        plt.plot(t,rc,label='r_c')
        plt.plot(t,thc,label='theta_c')
        plt.plot(t,phc,label='phi_c')
        plt.ylabel('Position')
    
    elif args.n == 2:
        plt.plot(t,uc,label='uc')
        plt.plot(t,vc,label='vc')
        plt.plot(t,wc,label='wc')
        plt.ylabel('Velocity')
    
    elif args.n == 3 :
        plt.plot(t,wx,label='wx')
        plt.plot(t,wy,label='wy')
        plt.plot(t,wz,label='wz')
        plt.plot(t,wth,label='wth')
        plt.ylabel('Angular velocity')
    
    elif args.n == 4:
        plt.plot(t,fx,label='fx')
        plt.plot(t,fy,label='fy')
        plt.plot(t,fz,label='fz')
        plt.ylabel('Force')
    
    elif args.n == 5:
        plt.plot(t,tx,label='tx')
        plt.plot(t,ty,label='ty')
        plt.plot(t,tz,label='tz')
        plt.plot(t,tth,label='tth')
        plt.ylabel('Torque')
    
    elif args.n == 6:
        plt.plot(t,cfx,label='cfx')
        plt.plot(t,cfy,label='cfy')
        plt.plot(t,cfz,label='cfz')
        plt.ylabel('Collision force')

    
    plt.grid()
    plt.xlabel('Time')
    plt.legend()

    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()

    print('Done')

if __name__ == "__main__":  
    particledata()
