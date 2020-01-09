#!/usr/bin/env python
# coding: utf-8

# Timo van Overveld, 2020

"""Obtain radial, angular and 2D distribution functions of input data"""
# Input: particle position files (.dat)
# Output: RDF, ADF, 2DDF

# Imports
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

############################################################################
def distributions():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='File')
    parser.add_argument('-r', action='store_true', help='Radial distribution function values')
    parser.add_argument('-p', action='store_true', help='Enable plotting')
    args = parser.parse_args()
   
    #Read data

    if args.f:
        X, Y = np.loadtxt(args.f)
    else:
        try:
            X, Y = np.loadtxt('fakepattern.dat')
        except:
            print('No input file found')
            quit()

    N = np.size(X) # Number of particles in domain
    L = 1   # Size of (square) domain
    M = L/2 # Value used for Modulo/shortest distance
    

    # Fill arrays with all distances and angles between particles

    Distances = np.empty((N,N),dtype=float)
    for i in range(0,N,1):
        for j in range(0,N,1):
            # Correct distances for a periodic domain
            dx = abs(X[i]-X[j])
            if dx > L/2:
                dx = L-dx        
            dy = abs(Y[i]-Y[j])
            if dy > L/2:
                dy = L-dy
            
            Distances[i,j] = np.sqrt(dx**2+dy**2)
            
    Angles  = np.empty((N,N),dtype=float)
    for i in range(0,N,1): #Place particle i at origin
        for j in range(0,N,1):
            if i == j:
                Angles[i,j] = 'NaN'
            else:
                dx = X[j]-X[i]     
                dy = Y[j]-Y[i]
    
                Angles[i,j]  = np.arctan2(dy,dx)
    

    # Binwidth, determines the smoothness
    dr   = L/2e2
    dth  = 2*np.pi/2e2
    
    #Integration steps, thus regions [theta-dth/2,theta+dth/2] overlap!
    dxr  = 1e-3
    dxth = 2*np.pi/(5*2**7)
    
    # r-dr/2     r=i*dxr       r+dr/2
    # [-------------------------]
    #  [-------------------------]
    #   [-------------------------]
    
    print('dr   =',format(dr,'.1e'))
    print('dth  =',format(dth,'.1e'))
    print('dxr  =',format(dxr,'.1e'))
    print('dxth =',format(dxth,'.1e'))
    
    rmin = dr/2 # Smallest radius to check (Typically particle radius, dr/2 or just 0)
    r = np.linspace(rmin,M,(M-rmin)/dxr+1) # Radii to calculate g
    
    th  = np.linspace(-np.pi,np.pi,(2*np.pi)/dxth+1) # Radii to calculate g
    
    a = np.empty((N,N,np.size(r)), dtype=float)
    b = np.empty((N,N,np.size(th)),dtype=float)
    
    gr  = np.empty(np.size(r),              dtype=float)
    gth = np.empty(np.size(th),             dtype=float)
    g   = np.empty((np.size(r),np.size(th)),dtype=float)
    

    for i in range(0,np.size(r),1):
        a[:,:,i] = np.logical_and(r[i]-dr/2<Distances,Distances<r[i]+dr/2)

        gr[i] = np.sum(a[:,:,i])*L**2/(N**2*2*np.pi*r[i]*dr)

    for j in range(0,np.size(th),1):
    
        b[:,:,j] = np.logical_and(th[j]-dth/2<Angles,Angles<th[j]+dth/2)
    
        if th[j]-dth/2 < -np.pi:
            b[:,:,j] += np.logical_and(th[j]-dth/2+2*np.pi<Angles,Angles<th[j]+dth/2+2*np.pi)
        if th[j]+dth/2 > np.pi:
            b[:,:,j] += np.logical_and(th[j]-dth/2-2*np.pi<Angles,Angles<th[j]+dth/2-2*np.pi)
    
        gth[j] = np.sum(b[:,:,j])*L**2*2*np.pi/(N**2*dth)

    for i in range(0,np.size(r),1):
        for j in range(0,np.size(th),1):
            c = np.multiply(a[:,:,i],b[:,:,j])
    
            g[i,j] = np.sum(c)*L**2/(N**2*r[i]*dr*dth)
    
    T,R = np.meshgrid(th,r)
    
    print('Average g = ',np.mean(g))
    glog = np.asarray([[np.log10(y) for y in x] for x in g])
    glog = np.where(glog<=-100,-1,glog)
    
    print('Calculations are done')
    
    # Plot
    
    if args.p:
        fig = plt.figure(figsize=(24,16))
        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(234, polar=True)
        ax4 = fig.add_subplot(235, polar=True)
        ax5 = fig.add_subplot(233)
        ax6 = fig.add_subplot(236, polar=True)

        ax1.plot(X,Y,'.')
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$y$')
        ax1.set_xlim(0, L)
        ax1.set_ylim(0, L)
        ax1.set_aspect('equal')
        ax1.set_title('Pattern')
        
        ax2.plot(r,gr,label='dr = '+format(dr,'.1e'))
        ax2.set_xlabel('$r$')
        ax2.set_ylabel('$g(r)$')
        ax2.grid()
        ax2.set_xlim(-0.01,M)
        ax2.legend()
        ax2.set_title('Radial distribution function')
   
        ax3.plot(th,gth,label='dth = '+format(dth,'.1e'))
        #ax3.set_xlabel('$\\theta$')
        #ax3.set_ylabel('$g(\\theta)$')
        ax3.legend()
        ax3.set_title('Angular distribution function')

        im = ax4.contourf(T,R,g,cmap="jet",levels=100)
        #ax4.set_xlabel('$\\theta$')
        #ax4.set_ylabel('$r$')
        ax4.set_title('2D distribution function')
        fig.colorbar(im,ax=ax4)
        
        xdir = np.argmin(abs(th))
        ydir = np.argmin(abs(th-np.pi/2))
        x = np.where(T==th[xdir])
        ax5.plot(R[x],g[x],label='x-direction')
        x = np.where(T==th[ydir])
        ax5.plot(R[x],g[x],label='y-direction')
        ax5.grid()
        ax5.legend()
        ax5.set_title('1D line plots of 2D distribution function')
        
        im = ax6.contourf(T,R,glog,cmap='bwr',levels=100)#, vmin=0, vmax=2)
        #ax6.set_xlabel('$\\theta$')
        #ax6.set_ylabel('$r$')
        ax6.set_title('Log10 of 2D distribution function')
        fig.colorbar(im,ax=ax6)
       
        savename = 'DF_'+args.f[:-4]+'.png'
        print('Plot saved as',savename)
        plt.savefig(savename)
        
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close()

    
    print('Done')

if __name__ == "__main__":  
    distributions()
