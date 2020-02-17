#!/usr/bin/env python
# coding: utf-8

# Timo van Overveld, 2020

"""Calculations on Stokes boundary layer"""
# Input: 
# Output: .png images

# Imports
import numpy as np
import os
import csv
import argparse
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

############################################################################
def analytical(x,t,H,A,f,nu,rho):
    alpha = H*np.sqrt(f/nu)
    s = 1j**(3/2)*alpha*(x-H/2)/H
    v = 1j/(rho*f) * A * (1-np.cos(s)/np.cos(1j**(3/2)*alpha/2))
    
    u = np.real(v * np.exp( 1j * (f*t - np.pi/2 )))
    
    return u

def stokes():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('f', type=str, help='File pattern')
    parser.add_argument('n', type=int, nargs='+', help='Which variables to use')
    parser.add_argument('-t', action='store_true', help='Timetrace of maximum value')
    parser.add_argument('-s', action='store_true', help='Save figure')
    args = parser.parse_args()
   
    # User inputs
    a,b = args.n
    
    lref = 5
    uref = 100
    tref = lref/uref

    H = 1*lref
    nu = 1
    mu = 1
    rho = mu/nu
    
    # Pressure amplitude and angular frequency
    A = -4*uref*mu/lref**2
    f = 1
   
    treal = []
    error = []
    vol = []
    
    ana_t = np.empty((0,120), float)
    num_t = np.empty((0,120), float)

    if args.f:
        with open('timestep.txt', newline='') as timetxt:
            time = np.loadtxt(timetxt)
        step = time[:,0]
        t    = time[:,1]*tref

        for fname in os.listdir('.'):
            if fname.startswith(args.f) and fname.endswith('0.txt'):
                print(fname) 
                with open(fname, newline='') as txt:
                    data = np.loadtxt(fname,skiprows=2)

                size = np.shape(data)
                _, idx = np.unique(data[:,a],return_index=True)

                x = data[idx,a]
                y = data[idx,b]
                timestr = fname[10:-4]

                try:
                    tidx = np.argwhere(step==int(timestr))[0]
                except:
                    tidx = 0
                
                treal = np.append(treal,t[tidx])
                
                # Obtain the numerical and analytical profiles
                num_t = np.append(num_t, np.array([y]), axis=0)
               
                anasol = analytical(H*x,t[tidx],H,A,f,nu,rho)
                ana_t = np.append(ana_t, np.asarray([anasol]), axis=0)
                
    #Sorting
    idx  = np.argsort(treal)
    treal = treal[idx]
    ana_t = ana_t[idx,:]
    num_t = num_t[idx,:]
   
    #Plotting 

    #Extreme values
    if args.t:
        fig = plt.figure(figsize=(12,12))
        ax1 = fig.add_subplot(221)
        halfway = int(np.shape(num_t)[1]/2)
        plt.plot(treal,num_t[:,halfway],label='Numerical')
        plt.plot(treal,ana_t[:,halfway]/uref,label='Analytical')
        plt.grid()
        plt.title('Center')
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.legend()
        
        ax2 = fig.add_subplot(223)
        plt.semilogy(treal,abs((ana_t[:,halfway]/uref-num_t[:,halfway])/(ana_t[:,halfway]/uref)))
        plt.grid()
        plt.xlabel('Time')
        plt.ylabel('Relative error')
      
        ax3 = fig.add_subplot(222)
        edgeway = 5
        plt.plot(treal,num_t[:,edgeway],label='Numerical')
        plt.plot(treal,ana_t[:,edgeway]/uref,label='Analytical')
        plt.grid()
        plt.title('Edge')
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.legend()

        ax4 = fig.add_subplot(224)
        plt.semilogy(treal,abs((ana_t[:,edgeway]/uref-num_t[:,edgeway])/(ana_t[:,edgeway]/uref)))
        plt.grid()
        plt.xlabel('Time')
        plt.ylabel('Relative error')
        
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close()

  
    # Profile comparison
    fig = plt.figure(figsize=(24,8))
    ax1 = fig.add_subplot(131)
    
    for i in range(0,100,10):
        plt.plot(x,ana_t[i,:]/uref,label=format(treal[i],'.1f'))
    
    plt.legend()
    plt.xlabel('y/H')
    plt.ylabel('u/u_0')
    plt.ylim(-0.2,0.2)
    plt.grid()
    plt.title('Analytical profiles')
    
    ax2 = fig.add_subplot(132)
    
    for i in range(0,100,10):
        plt.plot(x,num_t[i,:],label=format(treal[i],'.1f'))
    
    plt.legend()
    plt.xlabel('y/H')
    plt.ylabel('u/u_0')
    plt.ylim(-0.2,0.2)
    plt.grid()
    plt.title('Numerical profiles')

    ax3 = fig.add_subplot(133)
    
    yplot = [np.sum(abs((ana_t[t,:]/uref-num_t[t,:])/(ana_t[t,:]/uref*np.size(ana_t[t,:])))) for t in range(1,np.size(treal),1)]
    plt.semilogy(treal[1:],yplot)
    plt.xlabel('Time [s]')
    plt.ylabel('Average relative error')
    plt.grid()
    plt.title('Comparison')
    
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()
    

    print('Done')

if __name__ == "__main__":  
    stokes()
