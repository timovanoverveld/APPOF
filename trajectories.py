#!/usr/bin/env python
# coding: utf-8

# Timo van Overveld, 2019

"""Obtain the particle trajectories by coupling different particle data files."""
# Input: particle position files (.dat)
# Output: particle positions sorted

# Imports
import numpy as np
import json
import os
import argparse
import fnmatch
import matplotlib.pyplot as plt
#from images2positions_functions import *


############################################################################
def trajectories():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', action='store_true', help='Set verbosity')
    parser.add_argument('-f', help="settings file")
    args = parser.parse_args()
   
    verbose = False
    if args.v:
        verbose = True
        print("verbosity turned on")


    # Reading settings file
    f = open(args.f, 'r')
    settings = json.loads(f.read())

    # Directories
    basedir = settings['basedir']
    measurementdir = basedir + settings['measurementdir']
    positionsdir = measurementdir + 'positions/' 
    
    plots   = settings['plots']
    verbose = settings['verbose']

    if verbose: print('Constants read')

    # Find all position files (.dat)
    # Allocation
    flist = np.empty(0,dtype=int)

    for (dirpath, dirnames, filenames) in os.walk(positionsdir):
        f = np.asarray(fnmatch.filter(filenames,'*.dat*'))
        flist = np.append(flist,f)

    if np.size(flist) == 0:
        print('Searched through ',positionsdir,'\nNo measurement files specified, quitting.')
        quit()
    if verbose: print('User inputs read')

    #############################################
    sortedlist = [int(x.replace('.dat','').replace('.','')) for x in flist]
    sortedlist = np.argsort(sortedlist)
    
    flistsorted = list(flist[sortedlist])
    
    # Number of timesteps
    N_timesteps = np.size(np.asarray(flistsorted))

    # Maximum number of particles
    N_max = 300

    method = 'forwardsimple' #'forwardsimple'
     
    # Create emtpy array to store particle data, 
    # Number of particles X Number of timesteps X coordinates (x,y)
    particlessorted = np.zeros((N_max,N_timesteps,2),dtype=float)
    
    for i in range(0,N_timesteps,1):
        filename = flistsorted[i]
        data = np.loadtxt(positionsdir+filename)
            
        x = np.asarray(data[0])
        y = np.asarray(data[1])
        
        # Number of detected particles
        N = np.count_nonzero(x)
        
        # First file does not have to be sorted
        if flistsorted.index(filename) == 0:
            particlessorted[0:np.size(x),0,0] = x
            particlessorted[0:np.size(y),0,1] = y

        # All other files do have to be sorted
        else:
            # Simple nearest neighbours forward: loop over particles in i-1, find closest particle in i
            if method == 'forwardsimple':
                # Number of particles in previous timestep
                Nprev = np.count_nonzero(particlessorted[:,i-1,0])

                for j in range(0,Nprev,1):
                    # Distance from particle j in step i-1 to all particles in step i
                    distance_fw = np.sqrt((particlessorted[j,i-1,0]-x)**2+(particlessorted[j,i-1,1]-y)**2)
                     
                    # Find shortest distance and store the argument of the particle in the ith step that corresponds to the closest in j
                    idx_fw = np.argsort(distance_fw)[0]
                   
                    # Simplest is to store and overwrite where neccessary
                    particlessorted[j,i,0] = x[idx_fw]
                    particlessorted[j,i,1] = y[idx_fw]
            
            elif method == 'backwardsimple':
                for j in range(0,N,1):
                    # Distance from particle j in step i to all particles in step i-i
                    distance_bw = np.sqrt((particlessorted[:,i-1,0]-x[j])**2+(particlessorted[:,i-1,1]-y[j])**2)
                    #print(x[j],y[j])
                    #print(distance_bw) 
                    # Find shortest distance and store the argument of the particle in the ith step that corresponds to the closest in j
                    idx_bw = np.argsort(distance_bw)[0]
                    
                    # Simplest is to store and overwrite where neccessary
                    particlessorted[idx_bw,i,0] = x[j]
                    particlessorted[idx_bw,i,1] = y[j]

    print(particlessorted[:,0,0])
    print(particlessorted[:,1,0])
    print(particlessorted[:,2,0])
    print(particlessorted[:,3,0])

    # Create directories
    if not os.path.exists(measurementdir+'trajectories'):
        os.makedirs(measurementdir+'trajectories')

    # Saving the data
    savename_x = measurementdir+'trajectories/'+method+'_x.dat'
    savename_y = measurementdir+'trajectories/'+method+'_y.dat'
    np.savetxt(savename_x,particlessorted[:,:,0])
    np.savetxt(savename_y,particlessorted[:,:,1])

    if verbose: print('Particle trajectories stored in',savename_x)
    
    if plots: 
        plt.figure()
        for j in range(0,N_max,1):
            nonzero = np.nonzero(particlessorted[j,:,0])[0]
            plotx = particlessorted[j,nonzero,0]
            ploty = particlessorted[j,nonzero,1]
            plt.plot(plotx,ploty)
        plt.show()
       


        #I = np.linspace(0,N_max-1,N_max,dtype=int)
        #ind_loop = np.linspace(0,N_max-1,N_max,dtype=int) # Indices to include in the calculation
        #Distance_thres = 20 # Distance threshold in pixels
        #    
        #for t in range(0,20,1):
        #    # Choose particle i
        #    idx_fw = np.zeros(N_max,dtype=int)
        #    idx_bw = np.zeros(N_max,dtype=int)
        #
        #    # Loop over particles
        #    for i in ind_loop: #range(0,500,1):
        #        if i >= 0:
        #            # Forward time distances
        #            # Calculate distance to all other particles
        #            Distance_fw = np.sqrt((Particles[i,t,0]-x[t+1,ind_loop])**2+(Particles[i,t,1]-y[t+1,ind_loop])**2)
        #
        #            # Sort this matrix, to find the shortest distance in the next timestep
        #            idx_fw[i] = np.argsort(Distance_fw)[0]
        #
        #            # Backward time distances
        #            # Calculate distance to all other particles
        #            Distance_bw = np.sqrt((x[t+1,i]-Particles[ind_loop,t,0])**2+(y[t+1,i]-Particles[ind_loop,t,1])**2)
        #
        #            # Sort this matrix, to find the shortest distance in the next timestep
        #        #     idx2 = np.argsort(Distance_bw)[0]
        #            idx_bw[i] = np.argsort(Distance_bw)[0]    
        #
        #    # Find the parts where we have closed loops, store by index at t0 and t1
        #    closed_loop_check = idx_bw[idx_fw]-I
        #    a = np.where(closed_loop_check==0)[0] # The positions in idx_fw for which we have found a closed loop
        #    b = idx_fw[a] # The positions they point to
        #
        #    # Add to particles already
        #    Particles[a,t+1,0] = x[t+1,idx_fw[a]]
        #    Particles[a,t+1,1] = y[t+1,idx_fw[a]]
        #
        #    # Check for distances between found particles, if distance is too large then particles are not the same
        #    Distances = np.sqrt((Particles[:,t,0]-Particles[:,t+1,0])**2+(Particles[:,t,1]-Particles[:,t+1,1])**2)
        #    q = np.where(Distances>=Distance_thres)[0]
        #
        #    #First zero in particles: search backwards, find last nonzero 
        #    nz = np.nonzero(Particles[:,t+1,0])[0][-1] # Last nonzero, so new particle should be placed in 1 further (hence the +1)
        #
        ##     print('nz, q',nz,q)
        ##     print(Particles[nz+1:nz+1+np.size(q),t+1,0],Particles[q,t+1,0])
        #
        #    # Put discrepant particles at the back
        #    Particles[nz:nz+np.size(q),t+1,:] = Particles[q,t+1,:]
        #
        #    # Remove particles from previous locations:
        #    Particles[q,t+1,:] = 0
        #
        ##     print(Particles[nz:nz+1+np.size(q),t+1,0])
        
        
        #plt.figure(figsize=(20,10))
        #imax = 20
        #for i in range(0,imax):
        #    plt.scatter(x[i,:],y[i,:],s=5,color=(i/imax,1-i/imax,0))
        #
        #for i in range(0,500,1):
        #    nonzeros = np.argwhere(Particles[i,:,0] !=0 )
        #    if np.size(nonzeros)==0:
        #        break
        #    p1 = int(nonzeros[0])
        #    p2 = int(nonzeros[-1])
        #    plt.plot(Particles[i,p1:p2,0],Particles[i,p1:p2,1],'k',Linewidth=3)
        #
        #plt.xlim(1000,1200)
        #plt.ylim(0,350)
        #
        #plt.show()
        
        #print('done')


if __name__ == "__main__":  
    trajectories()
