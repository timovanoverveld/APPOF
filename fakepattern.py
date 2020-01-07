#!/usr/bin/env python
# coding: utf-8

# Timo van Overveld, 2020

"""Generate a fake particle pattern used for testing analysis in other scripts"""
# Output: .dat file with particle positions

# Imports
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


############################################################################
def fakepattern():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=int, help='Pattern type: 0 random, 1 square, 2 lines, 3 fluidlike')
    parser.add_argument('-N', type=int, help='Number of particles')
    parser.add_argument('-l', type=int, help='Number of lines')
    parser.add_argument('-n', type=float, help='Add noise to the pattern by specifying an amplitude')
    parser.add_argument('-p', action='store_true', help='Enable plotting')
    parser.add_argument('-f', action='store_true', help='File name')
    args = parser.parse_args()
   
    #Fake data
    N = args.N # Number of particles in domain
   
    L = 1   # Size of (square) domain, always 1
    M = L/2 # Value used for Modulo/shortest distance
    
    if args.t == None:
        print('No pattern type given')
        quit()

    if args.t == 0: #'random'
        # Random 2D distribution
        X = L*np.random.rand(N) # Random X coordinates
        Y = L*np.random.rand(N) # Random Y coordinates
    
    elif args.t == 1: #'square'
        # Square ordered 2D distribution
        X = np.tile(np.linspace(L/(2*np.sqrt(N)),L-L/(2*np.sqrt(N)),N/(np.sqrt(N))),int(np.sqrt(N)))
        Y = np.sort(X)
    
    elif args.t == 2: #'lines'
        # Line ordered 2D distribution
        Nrows = args.l # Number of rows, N should be divisible by this
        if N/Nrows%1 != 0:
            print('Error in number of rows')
            quit()
    
        # Lay down the particles in a grid
        X = np.tile(np.linspace(L*Nrows/(2*N),L*(1-Nrows/(2*N)),N/Nrows),Nrows)
        Y = np.sort(np.tile(np.linspace(L/(2*Nrows),L*(1-1/(2*Nrows)),Nrows),int(N/Nrows)))
    
        print('Interparticle distance =',L/(N/Nrows))
        print('Interline distance =',L/Nrows)
    
    elif args.t == 3: #'fluid'
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
    
    # Add noise
    if args.n:
        A = args.n
        B = args.n
        X = X + A*2*(np.random.rand(np.size(X))-0.5)
        Y = Y + B*2*(np.random.rand(np.size(Y))-0.5)

    # Plotting
    if args.p:
        plt.figure(figsize=(12,12))
        plt.scatter(X,Y)
        plt.xlim(0,L)
        plt.ylim(0,L)
        plt.grid()
        plt.axes().set_aspect('equal')
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close()

    if args.f:
        filename = args.f
    else:
        filename = 'fakepattern.dat'

    # Save pattern data
    np.savetxt(filename,np.asarray([X,Y]))

    print('Pattern stored in',filename)

if __name__ == "__main__":  
    fakepattern()
