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
    parser.add_argument('-f', action='store_true', help='File')
    parser.add_argument('-r', action='store_true', help='Radial distribution function values')
    parser.add_argument('-p', action='store_true', help='Enable plotting')
    args = parser.parse_args()
   
    #Fake data
    N = 120 # Number of particles in domain
    L = 1   # Size of (square) domain
    M = L/2 # Value used for Modulo/shortest distance
    
    rho = N/L**2 # Average particle density (#/m^2)
    
    dist = 'lines'
    noise = True
    if dist == 'random':
        # Random 2D distribution
        X = L*np.random.rand(N) # Random X coordinates
        Y = L*np.random.rand(N) # Random Y coordinates
    
    elif dist == 'square':
        # Square ordered 2D distribution
        X = np.tile(np.linspace(0,L,N/10),11)
        Y = np.sort(X)
    
    elif dist == 'lines':
        # Line ordered 2D distribution
        Nrows = 6 # Number of rows, N should be divisible by this
        if N/Nrows%1 != 0:
            print('Error in number of rows')
            sys.exit()
    
        # Lay down the particles in a grid
        X = np.tile(np.linspace(L*Nrows/(2*N),L*(1-Nrows/(2*N)),N/Nrows),Nrows)
        Y = np.sort(np.tile(np.linspace(L/(2*Nrows),L*(1-1/(2*Nrows)),Nrows),int(N/Nrows)))
    
        print('Interparticle distance =',L/(N/Nrows))
        print('Interline distance =',L/Nrows)
    
    elif dist == 'fluid':
        radius = 0.025 # Sphere radius
        X = L*np.random.rand(1)
        Y = L*np.random.rand(1)
    
        while np.size(X)<N:
            x = L*np.random.rand()
            y = L*np.random.rand()
    
            distances = np.sqrt((X-x)**2+(Y-y)**2)
            if np.all(distances > radius):
                X = np.append(X,x)
                Y = np.append(Y,y)
    
        print(np.size(X))
    
    # Add noise
    if noise:
        A = 1e-2
        B = 1e-2
        X = X + A*2*(np.random.rand(np.size(X))-0.5)
        Y = Y + B*2*(np.random.rand(np.size(Y))-0.5)
    
    plt.figure(figsize=(12,12))
    plt.scatter(X,Y)
    plt.xlim(0,L)
    plt.ylim(0,L)
    plt.grid()
    plt.axes().set_aspect('equal')
    plt.show()


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
    
    # Calculate the combined correlation function in 1 step

    # Binwidth, determines the smoothness
    dr   = L/2e2
    dth  = 2*np.pi/2e2
    
    #Integration steps, thus regions [theta-dth/2,theta+dth/2] overlap!
    dxr  = 1e-2
    dxth = 2*np.pi/(5*2**7)
    
    # r-dr/2     r       r+dr/2
    # [------------------]
    #  [------------------]
    #   [----------------- ]
    
    print('dr   =',format(dr,'.1e'))
    print('dth  =',format(dth,'.1e'))
    print('dxr  =',format(dxr,'.1e'))
    print('dxth =',format(dxth,'.1e'))
    
    rmin = dr/2 # Smallest radius to check (Typically particle radius, dr/2 or just 0)
    r = np.linspace(rmin,M,(M-rmin)/dxr+1) # Radii to calculate g
    
    th  = np.linspace(-np.pi,np.pi,(2*np.pi)/dxth+1) # Radii to calculate g
    
    a = np.empty((N,N,np.size(r)),dtype=float)
    b = np.empty((N,N,np.size(th)),dtype=float)
    g = np.empty((np.size(r),np.size(th)),dtype=float)
    
    for i in range(0,np.size(r),1):
        a[:,:,i] = np.logical_and(r[i]-dr/2<Distances,Distances<r[i]+dr/2)
    
    for j in range(0,np.size(th),1):
    
        b[:,:,j] = np.logical_and(th[j]-dth/2<Angles,Angles<th[j]+dth/2)
    
        if th[j]-dth/2 < -np.pi:
            b[:,:,j] += np.logical_and(th[j]-dth/2+2*np.pi<Angles,Angles<th[j]+dth/2+2*np.pi)
        if th[j]+dth/2 > np.pi:
            b[:,:,j] += np.logical_and(th[j]-dth/2-2*np.pi<Angles,Angles<th[j]+dth/2-2*np.pi)
    
    for i in range(0,np.size(r),1):
        for j in range(0,np.size(th),1):
            c = np.multiply(a[:,:,i],b[:,:,j])
    
            g[i,j] = np.sum(c)/r[i] #rho*N*r[i]*dr*dth)
    
    
    g = g / (rho*N*dr*dth)
    
    print('Calculations are done')
    
    # Plot
    T,R = np.meshgrid(th,r)
    
    print(np.mean(g))
    glog = np.asarray([[np.log10(y) for y in x] for x in g])
    
    glog = np.where(glog<=-100,np.unique(np.sort(glog))[1],glog)
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),figsize=(12,12))
    plt.contourf(T,R,g,cmap="jet",levels=100)
    plt.xlabel('$\\theta$')
    plt.ylabel('$r$')
    plt.ylim(0)
    plt.colorbar()
    # plt.grid()
    plt.show()
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),figsize=(12,12))
    plt.contourf(T,R,glog,cmap='bwr',levels=100,vmin=-np.max(glog),vmax=np.max(glog))
    plt.xlabel('$\\theta$')
    plt.ylabel('$r$')
    plt.ylim(0)
    plt.colorbar()
    # plt.grid()
    plt.show()
    
    plt.figure(figsize=(12,8))
    xdir = np.argmin(abs(th))
    ydir = np.argmin(abs(th-np.pi/2))
    x = np.where(T==th[xdir])
    plt.plot(R[x],g[x],label='x-direction')
    x = np.where(T==th[ydir])
    plt.plot(R[x],g[x],label='y-direction')
    plt.grid()
    plt.legend()
    plt.show()

    
    print('Done')

if __name__ == "__main__":  
    distributions()
