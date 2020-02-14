#!/usr/bin/env python
# coding: utf-8

# Timo van Overveld, 2020

"""Calculations on Poiseuille profile"""
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
def calculate_error(x,y):
    num = y/np.max(y)
    ana = 4*x*(1-x)
    error = np.sum(np.abs(num-ana)) #Relative
    return error

def volumediff(x,y):
    ana = 2/3
    num = integrate.simps(y/np.max(y),x) 
    return ana, num

def um(m,y,t,H,G,mu,nu):
    um  = np.sin((2*m-1)*np.pi*y/H)
    um *= np.exp(-(2*m-1)**2*np.pi**2*nu*t/H**2)
    um /= (2*m-1)**3
    um *= -4*G*H**2/(np.pi**3*mu)
    return um

def poisseuille():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('f', type=str, help='File pattern')
    parser.add_argument('n', type=int, nargs='+', help='Which variables to use')
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
    G = 4*uref*nu/lref**2
   
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
                error = np.append(error,calculate_error(x,y))
                ana, num = volumediff(x,y)
                vol = np.append(vol,np.abs(num-ana)/ana)

                # Compare to analytical solution at time t
                x_ana = H*x
                u = G/(2*mu)*x_ana*(H-x_ana)
                for m in range(1,1000,1):
                    u += um(m,x_ana,t[tidx],H,G,mu,nu)
                ana_t = np.append(ana_t, np.array([u]), axis=0)
                num_t = np.append(num_t, np.array([y]), axis=0)

    #Sorting
    idx  = np.argsort(treal)
    treal = treal[idx]
    error = error[idx]
    vol = vol[idx]
    ana_t = ana_t[idx,:]
    num_t = num_t[idx,:]
    
    #Plotting
    fig = plt.figure(figsize=(12,12))
   
    plt.plot(treal,error,label='Relative error in datapoints') 
    plt.plot(treal,vol,label='Relative error in integrated volume') 

    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.xlabel('Time [s]')
    plt.ylabel('Relative error')

    if args.s:
        name = args.f[:-4]+'.png'
        print('Saved as',name)
        plt.savefig(args.f[:-4]+'.png')
    elif not args.s:
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close()

    fig = plt.figure(figsize=(24,8))
    ax1 = fig.add_subplot(131)
    
    for i in range(0,100,10):
        plt.plot(x,ana_t[i,:]/uref,label=format(treal[i],'.1f'))
    
    plt.plot(x,G/(2*mu)*x_ana*(H-x_ana)/uref,label='t -> infinity')
    plt.legend()
    plt.xlabel('y/H')
    plt.ylabel('u/u_0')
    plt.grid()
    plt.title('Analytical profiles')
    
    ax2 = fig.add_subplot(132)
    
    for i in range(0,100,10):
        plt.plot(x,num_t[i,:],label=format(treal[i],'.1f'))
    
    plt.legend()
    plt.xlabel('y/H')
    plt.ylabel('u/u_0')
    plt.grid()
    plt.title('Numerical profiles')

    ax3 = fig.add_subplot(133)
    
    yplot = [np.sum(abs(ana_t[t,:]/uref-num_t[t,:])/(ana_t[t,:]/uref*np.size(ana_t[t,:]))) for t in range(1,np.size(treal),1)]
    plt.semilogy(treal[1:],yplot)
    plt.xlabel('Time [s]')
    plt.ylabel('Average relative error')
    plt.grid()
    plt.title('Comparison')
    
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()
    
#    plt.figure()
#    plt.semilogy(abs(ana_t[50,:]/uref-num_t[50,:])/(ana_t[50,:]/uref*np.size(ana_t[50,:])))
#    plt.show()

    print('Done')

if __name__ == "__main__":  
    poisseuille()
