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
def intersectwall(wx, wy, X, Y, th):
    labda = ((X-wx[0])/(wx[1]-wx[0])*(wy[1]-wy[0])+wy[0]-Y)/(np.tan(th)-(wy[1]-wy[0])/(wx[1]-wx[0]))
    xinter = labda+X
    yinter = labda*np.tan(th)+Y
    return labda, xinter, yinter


def distributions():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='File')
    parser.add_argument('-r', type=float, help='Radial distribution function values')
    parser.add_argument('-p', action='store_true', help='Enable plotting')
    parser.add_argument('-P', action='store_true', help='Domain is periodic')
    parser.add_argument('-l', action='store_true', help='Log10 colormap')
    parser.add_argument('-L', type=float, help='Maximum correlation/domain size')
    parser.add_argument('-wx', type=float, nargs='+', help='Wall endpoints x-coordinates')
    parser.add_argument('-wy', type=float, nargs='+', help='Wall endpoints y-coordinates')
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

    if np.size(X) == 0:
        print('No input data found in ',args.f)
        quit()

    N = np.size(X) # Number of particles in domain
    
    if args.L:
        L = args.L
    else:
        L = 1 #np.max([np.max(X)-np.min(X),np.max(Y)-np.min(Y)])   # Size of (square) domain
    
    if args.P:
        M = L/2 # Value used for Modulo/shortest distance
    else:
        M = L
   
    if args.wx and args.wy:
        wx = args.wx
        wy = args.wy

    # Fill arrays with all distances and angles between particles

    Distances = np.empty((N,N),dtype=float)
    for i in range(0,N,1):
        for j in range(0,N,1):
            # Calculate absolute distances in x and y directions
            dx = abs(X[i]-X[j])
            dy = abs(Y[i]-Y[j])

            # Correct distances for a periodic domain
            if args.P:
                if dx > L/2:
                    dx = L-dx        
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
    dxr  = 1e-2#1e-4
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
    thsym  = np.linspace(0,np.pi/2,(np.pi/2)/dxth) # Radii to calculate g
    halfsize = int(np.size(th)/2)
    
    a = np.empty((N,N,np.size(r)), dtype=float)
    b = np.empty((N,N,np.size(th)),dtype=float)
    
    gr  = np.empty(np.size(r),              dtype=float)
    gth = np.empty(np.size(th),             dtype=float)
    g   = np.empty((np.size(r),np.size(th)),dtype=float)
    gsym = np.empty((np.size(r),int(np.size(th)/4)),dtype=float)
    

    for i in range(0,np.size(r),1):
        a[:,:,i] = np.logical_and(r[i]-dr/2<Distances,Distances<r[i]+dr/2)

        gr[i] = np.sum(a[:,:,i])*L**2/(N**2*2*np.pi*r[i]*dr)
    
    for j in range(0,np.size(th),1):
    
        area = np.zeros(N,dtype=float)
        
        b[:,:,j] = np.logical_and(th[j]-dth/2<Angles,Angles<th[j]+dth/2)
    
        if th[j]-dth/2 < -np.pi:
            b[:,:,j] += np.logical_and(th[j]-dth/2+2*np.pi<Angles,Angles<th[j]+dth/2+2*np.pi)
        if th[j]+dth/2 > np.pi:
            b[:,:,j] += np.logical_and(th[j]-dth/2-2*np.pi<Angles,Angles<th[j]+dth/2-2*np.pi)
   
        # For all walls
        for i in range(0,np.size(wx),2):
            wallx = wx[i:i+2]
            wally = wy[i:i+2]
            
            # Wall intersections
            labda, xinter, yinter = intersectwall(wallx,wally,X,Y,th[j])
            
            # Angle above and below
            labda_a, x_a, y_a = intersectwall(wallx,wally,X,Y,th[j]+dth/2)
            labda_b, x_b, y_b = intersectwall(wallx,wally,X,Y,th[j]-dth/2)
            
            for i in range(0,N,1):
               # if labda[i]>0 and labda_a[i]>0 and labda_b[i]>0: #Intersections with wall
                if np.min([x_a[i],x_b[i]]) <= xinter[i] <= np.max([x_a[i],x_b[i]]):
                    if np.min([y_a[i],y_b[i]]) <= yinter[i] <= np.max([y_a[i],y_b[i]]):
                        area[i] = abs((X[i]*(y_a[i]-y_b[i])+x_a[i]*(y_b[i]-Y[i])+x_b[i]*(Y[i]-y_a[i]))/2)

        area = np.where(area==0,dth/(2*np.pi),area)

        #gth[j] = np.sum(b[:,:,j])*L**2*2*np.pi/(N**2*dth)
        gth[j] = np.sum(b[:,:,j])*L**2/(N**2*np.mean(area))

    for i in range(0,np.size(r),1):
        for j in range(0,np.size(th),1):
            c = np.multiply(a[:,:,i],b[:,:,j])
    
            g[i,j] = np.sum(c)*L**2/(N**2*r[i]*dr*dth)
    
    
    print('Average g = ',np.mean(g))

    # Calculate the (assumed) symmetric quadrant of g2D
    for i in range(0,int(np.size(th)/4),1):
        gsym[:,i] = (g[:,i]+g[:,-i]+g[:,halfsize+i]+g[:,halfsize-i])/4

    # If logarithmic colormap, calculate the log10()
    if args.l:
        glog = np.asarray([[np.log10(y) for y in x] for x in g])
        glog = np.where(glog<=-100,-np.max(glog),glog)
        
        gsymlog = np.asarray([[np.log10(y) for y in x] for x in gsym])
        gsymlog = np.where(gsymlog<=-100,np.unique(np.sort(gsymlog))[1],gsymlog)
    
    T,R = np.meshgrid(th,r)
    Tsym,Rsym = np.meshgrid(thsym,r)
    
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
        #ax1.set_xlim(0, L)
        #ax1.set_ylim(0, L)
        ax1.set_aspect('equal')
        ax1.set_title('Pattern')

        if args.wx and args.wy:
            for i in range(0,np.size(wx),2):
                ax1.plot(wx[i:i+2],wy[i:i+2],'k')
        
        ax2.plot(r,gr,label='dr = '+format(dr,'.1e'))
        ax2.set_xlabel('$r$')
        ax2.set_ylabel('$g(r)$')
        ax2.grid()
        ax2.set_xlim(-0.01,M)
        ax2.legend()
        ax2.set_title('Radial distribution function')
   
        ax3.plot(th,gth,label='dth = '+format(dth,'.1e'))
        ax3.legend()
        ax3.set_title('Angular distribution function')

        if args.l:
            im = ax4.contourf(T,R,glog,cmap='bwr',levels=100, vmin=np.min(glog), vmax=np.max(glog))
            ax4.set_title('Log10 of 2D distribution function')
        else: 
            im = ax4.contourf(T,R,g,cmap="jet",levels=100)
            ax4.set_title('2D distribution function')
        fig.colorbar(im,ax=ax4)
        
        xdir = np.argmin(abs(thsym))
        ydir = np.argmin(abs(thsym-np.pi/2))
        x = np.where(Tsym==thsym[xdir])
        ax5.plot(Rsym[x],gsym[x],label='x-direction')
        x = np.where(Tsym==thsym[ydir])
        ax5.plot(Rsym[x],gsym[x],label='y-direction')
        ax5.grid()
        ax5.legend()
        ax5.set_title('1D line plots of 2D distribution function')
        
        if args.l:
            im = ax6.contourf(Tsym,Rsym,gsymlog,cmap='bwr',levels=100, vmin=np.min(gsymlog), vmax=np.max(gsymlog))
            ax6.set_title('Log10 of 2D distribution function')
        else: 
            im = ax6.contourf(Tsym,Rsym,gsym,cmap="jet",levels=100)
            ax6.set_title('2D distribution function')
        ax6.set_thetamin(0)
        ax6.set_thetamax(90)
        fig.colorbar(im,ax=ax6)
        
        lastslash = args.f.rfind('/')  
        savename = args.f[0:lastslash+1]+'DF_'+args.f[lastslash+1:-4]+'.png'
        plt.savefig(savename)
        print('Plot saved as',savename)
        
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close()

    
    print('Done')

if __name__ == "__main__":  
    distributions()
