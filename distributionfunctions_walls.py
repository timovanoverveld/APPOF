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
# This function calculates the intersection of two lines. We have a line l=(x0,y0)+labda*(cos(th),sin(th)) and a wall with m=(x1,y1)+mu*(x2-x1,y2-y1). The intersection is considered valid when labda>=0 and 0<=mu<=1
def intersectwall(x0, y0, x1, y1, x2, y2, th):
    labda = ((x0-x1)*(y2-y1)-(y0-y1)*(x2-x1)) / ((x2-x1)*np.sin(th)-(y2-y1)*np.cos(th))

    #Calculate 2 representations of mu, based on x and y component of line m
    mu_x = (labda*np.cos(th)+(x0-x1)) / (x2-x1)
    mu_y = (labda*np.sin(th)+(y0-y1)) / (y2-y1)

    # Replace mu_x if nan of +/- inf (happens when wall is horizontal or vertical)
    mu_x_false = np.logical_or(np.isnan(mu_x),np.isinf(mu_x))
    
    mu = np.where(mu_x_false,mu_y,mu_x)
    
    xinter = x0 + labda*np.cos(th)
    yinter = y0 + labda*np.sin(th)

    return labda, mu, xinter, yinter

# This function calculates the intersection of a circle and a line. We have a line m=(x1,y1)+mu*(x2-x1,y2-y1) and a circle c:(x-x0)^2+(y-y0)^2=r^2. 
def intersectcirclewall(x0, y0, x1, y1, x2, y2, r):

    #Intersection gives A*mu^2+B*mu+C=0
    A = (x2-x1)**2 + (y2-y1)**2
    B = 2*(x1-x0)*(x2-x1) + 2*(y1-y0)*(y2-y1)
    C = (x1-x0)**2 + (y1-y0)**2 - r**2
    
    #Discriminant
    D = B**2-4*A*C
    
    mu1 = (-B-np.sqrt(D)) / (2*A)
    mu2 = (-B+np.sqrt(D)) / (2*A)
   
    xinter1 = x1 + mu1*(x2-x1)
    yinter1 = y1 + mu1*(y2-y1)
    
    xinter2 = x1 + mu2*(x2-x1)
    yinter2 = y1 + mu2*(y2-y1)

    return mu1, xinter1, yinter1, mu2, xinter2, yinter2

def distributions():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='File')
    parser.add_argument('-r', type=float, help='Radial distribution function values')
    parser.add_argument('-p', action='store_true', help='Enable plotting')
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
   
    if args.wx and args.wy:
        wx = args.wx
        wy = args.wy
        if wx[0] != wx[-1] or wy[0] != wy[-1]:
            print('Walls do not form a closed loop')
            quit()

    # Fill arrays with all distances and angles between particles

    Distances = np.empty((N,N),dtype=float)
    for i in range(0,N,1):
        for j in range(0,N,1):
            # Calculate absolute distances in x and y directions
            dx = abs(X[i]-X[j])
            dy = abs(Y[i]-Y[j])

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
    dr   = L/1e2
    dth  = 2*np.pi/5e1
    
    #Integration steps, thus regions [theta-dth/2,theta+dth/2] overlap!
    dxr  = 1e-2#1e-4
    dxth = 2*np.pi/(5*2**6)
    
    # r-dr/2     r=i*dxr       r+dr/2
    # [-------------------------]
    #  [-------------------------]
    #   [-------------------------]
    
    print('dr   =',format(dr,'.1e'))
    print('dth  =',format(dth,'.1e'))
    print('dxr  =',format(dxr,'.1e'))
    print('dxth =',format(dxth,'.1e'))
    
    rmin = dr/2 # Smallest radius to check (Typically particle radius, dr/2 or just 0)
    r = np.linspace(rmin,L,(L-rmin)/dxr+1) # Radii to calculate g
    
    th  = np.linspace(-np.pi,np.pi,(2*np.pi)/dxth+1) # Radii to calculate g
    thsym  = np.linspace(0,np.pi/2,(np.pi/2)/dxth) # Radii to calculate g
    halfsize = int(np.size(th)/2)
    
    a = np.empty((N,N,np.size(r)), dtype=float)
    b = np.empty((N,N,np.size(th)),dtype=float)
    
    gr  = np.empty(np.size(r),              dtype=float)
    gth = np.empty(np.size(th),             dtype=float)
    g   = np.empty((np.size(r),np.size(th)),dtype=float)
    gsym = np.empty((np.size(r),int(np.size(th)/4)),dtype=float)
    
    ###############################################################
    # Radial
    for i in range(0,np.size(r),1):
        a[:,:,i] = np.logical_and(r[i]-dr/2<Distances,Distances<r[i]+dr/2)

        # Give starting values of areas
        area = 2*np.pi*r[i]*dr*np.ones(N,dtype=float)

        # Store single intersections
        intersect = np.zeros((N,np.size(wx)-1),dtype=float)
        
        # Calculate intersections for all walls
        for k in range(0,np.size(wx)-1,1):
            wallx = wx[k:k+2]
            wally = wy[k:k+2]
            
            # Wall intersections
            mu1, xinter1, yinter1, mu2, xinter2, yinter2 = intersectcirclewall(X,Y,wallx[0],wally[0],wallx[1],wally[1],r[i])
            
            # Calculate corresponding angles to intersection points
            theta1 = np.arctan2(yinter1-Y,xinter1-X)
            theta2 = np.arctan2(yinter2-Y,xinter2-X)

            for j in range(0,N,1):
                # 2 Intersections with 1 wall 
                if 0 <= mu1[j] <= 1 and 0 <= mu2[j] <= 1:    #Check for intersection in wall
                    # Segment to disregard
                    angularsegment = np.min([(theta1[j]-theta2[j])%(2*np.pi),(theta2[j]-theta1[j])%(2*np.pi)])
                    # Remove corresponding fraction from area
                    area[j] -= r[i]*dr*angularsegment
                    #if angularsegment > np.pi/1.1:
                    #    print(k,X[j],Y[j],r[i],angularsegment)

                # 1 Intersection with 1 wall: store for later
                if np.logical_xor(0 <= mu1[j] <= 1, 0 <= mu2[j] <= 1):
                    if 0 <= mu1[j] <= 1:
                        intersect[j,k] = theta1[j]
                    elif 0 <= mu2[j] <= 1:
                        intersect[j,k] = theta2[j]
       
        # Choose pseudo-angle:
        thps = 1

        # For the intersections with multiple walls (such as cases with corners) we have the information stored in intersect[:,:], which contains the angles at which the circle intersects with a wall and based on the 2nd index, which wall it intersects with
        # Question now is if the part between theta1 and theta2 lies inside or outside the domain (could be both). For this we consider a pseudo-angle and see if it first intersects with a wall or with the circle at radius r[i].
        for j in range(0,N,1):
            if np.count_nonzero(intersect[j,:])>0:
                # Intersection with circle always at radius r[i]
                
                # Find intersection with wall
                for k in range(0,np.size(wx)-1,1):
                    wallx = wx[k:k+2]
                    wally = wy[k:k+2]
                    
                    labda, mu, xinter, yinter = intersectwall(X[j],Y[j],wallx[0],wally[0],wallx[1],wally[1],thps)
                    if labda >= 0 and 0 <= mu <= 1:
                        #print(j,labda,mu)
                        break
                    #print(xinter,yinter)

                # Distance to wall
                D = np.sqrt((X[j]-xinter)**2 + (Y[j]-yinter)**2)

                theta1, theta2 = intersect[j,np.nonzero(intersect[j,:])[0]]
                
                if D > r[i]:
                    # thps in part that should be kept
                    if np.logical_and(theta1 >= thps, theta2 >= thps):
                        angularsegment = np.max([theta1,theta2])-np.min([theta1,theta2])
                    elif np.logical_xor(theta1 >= thps, theta2 >= thps):
                        angularsegment = 2*np.pi - (np.max([theta1,theta2])-np.min([theta1,theta2]))
                    elif np.logical_and(theta1 <= thps, theta2 <= thps):
                        angularsegment = np.max([theta1,theta2])-np.min([theta1,theta2])
                
                elif D <= r[i]:
                    # thps in part that should be discarded
                    if np.logical_and(theta1 >= thps, theta2 >= thps):
                        angularsegment = 2*np.pi - (np.max([theta1,theta2])-np.min([theta1,theta2]))
                    elif np.logical_xor(theta1 >= thps, theta2 >= thps):
                        angularsegment = np.max([theta1,theta2])-np.min([theta1,theta2])
                    elif np.logical_and(theta1 <= thps, theta2 <= thps):
                        angularsegment = 2*np.pi - (np.max([theta1,theta2])-np.min([theta1,theta2]))

                # Exclude intersected parts from area
                area[j] -= r[i]*dr*angularsegment

        #TODO only do check below if count == 0, for faster calculations
        # Check if circle completely outside domain
        for j in range(0,N,1):
            distance_to_corners = np.empty(np.size(wx)-1,dtype=float)
            for k in range(0,np.size(wx)-1,1):
                distance_to_corners[k] = np.sqrt((X[j]-wx[k])**2+(Y[j]-wy[k])**2)
            
            # Full circle outside of domain, (if domain is convex) so do not count full area
            if np.all(distance_to_corners <= r[i]):
                #print(r[i],distance_to_corners)
                area[j] = 0
        
        #gr[i] = np.sum(a[:,:,i])*L**2/(N**2*2*np.pi*r[i]*dr)
        #gr[i] = np.sum(a[:,:,i])*L**2/(N**2*np.mean(area))
        rho_expected = N/L**2
        rho_measured = np.sum(a[:,:,i])/np.sum(area)
        gr[i] = rho_measured/rho_expected
    
    ######################################################################
    # Angular
    for j in range(0,np.size(th),1):
    
        area = np.zeros(N,dtype=float)
        
        b[:,:,j] = np.logical_and(th[j]-dth/2<Angles,Angles<th[j]+dth/2)
    
        if th[j]-dth/2 < -np.pi:
            b[:,:,j] += np.logical_and(th[j]-dth/2+2*np.pi<Angles,Angles<th[j]+dth/2+2*np.pi)
        if th[j]+dth/2 > np.pi:
            b[:,:,j] += np.logical_and(th[j]-dth/2-2*np.pi<Angles,Angles<th[j]+dth/2-2*np.pi)
   
        # For all walls
        for k in range(0,np.size(wx)-1,1):
            wallx = wx[k:k+2]
            wally = wy[k:k+2]
            
            # Wall intersections
            labda, mu, xinter, yinter = intersectwall(X,Y,wallx[0],wally[0],wallx[1],wally[1],th[j])
             
            # Angle above and below
            labda_a, mu_a, x_a, y_a = intersectwall(X,Y,wallx[0],wally[0],wallx[1],wally[1],th[j]+dth/2)
            labda_b, mu_b, x_b, y_b = intersectwall(X,Y,wallx[0],wally[0],wallx[1],wally[1],th[j]-dth/2)
             
            for i in range(0,N,1):
                if labda[i] > 0 and 0 <= mu[i] <= 1:    #Check for intersection
                    area[i] = abs((X[i]*(y_a[i]-y_b[i])+x_a[i]*(y_b[i]-Y[i])+x_b[i]*(Y[i]-y_a[i]))/2)
            
        area = np.where(area==0,dth/(2*np.pi),area)

        #gth[j] = np.sum(b[:,:,j])*L**2*2*np.pi/(N**2*dth)
        #gth[j] = np.sum(b[:,:,j])*L**2/(N**2*np.mean(area))
        
        rho_expected = N/L**2
        rho_measured = np.sum(b[:,:,j])/np.sum(area)
        gth[j] = rho_measured/rho_expected

    #######################################################################3
    # 2D
    for i in range(0,np.size(r),1):
        for j in range(0,np.size(th),1):
            c = np.multiply(a[:,:,i],b[:,:,j])
            
            # Give starting values of areas
            area = r[i]*dr*dth*np.ones(N,dtype=float)
           
            # Only check if count == 0
            #TODO adapt here 
            # Check for angle th[j] with which wall we intersect
            for k in range(0,np.size(wx)-1,1):
                wallx = wx[k:k+2]
                wally = wy[k:k+2]
                
                # Wall intersections for all particles
                labda, mu, xinter, yinter = intersectwall(X,Y,wallx[0],wally[0],wallx[1],wally[1],th[j])
                for l in range(0,N,1):
                    if labda[l] > 0 and 0 <= mu[l] <= 1:    #Check for intersection
                        distance_to_wall = np.sqrt((X[l]-xinter[l])**2+(Y[l]-yinter[l])**2)
                        if distance_to_wall <= r[i]: # Radius larger, then outside domain
                            area[l] = 0                             
            
            # If inside, then include in the calculation 
            
            # If outside, then don't include

            #g[i,j] = np.sum(c)*L**2/(N**2*r[i]*dr*dth)
            
            rho_expected = N/L**2
            rho_measured = np.sum(c)/np.sum(area)
            g[i,j] = rho_measured/rho_expected

    
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
            for i in range(0,np.size(wx),1):
                ax1.plot(wx[i:i+2],wy[i:i+2],'k')
        
        ax2.plot(r,gr,label='dr = '+format(dr,'.1e'))
        ax2.set_xlabel('$r$')
        ax2.set_ylabel('$g(r)$')
        ax2.grid()
        ax2.set_xlim(-0.01,L)
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
            ax6.set_title('Log10 of 4-Quadrant averaged 2D distribution function')
        else: 
            im = ax6.contourf(Tsym,Rsym,gsym,cmap="jet",levels=100)
            ax6.set_title('4-Quadrant averaged 2D distribution function')
        ax6.set_thetamin(0)
        ax6.set_thetamax(90)
        fig.colorbar(im,ax=ax6)
        
        lastslash = args.f.rfind('/')  
        #savename = args.f[0:lastslash+1]+'DF_'+args.f[lastslash+1:-4]+'.eps'
        savename = args.f[0:lastslash+1]+'DF_'+args.f[lastslash+1:-4]+'.png'
        plt.savefig(savename)#, format='eps')
        print('Plot saved as',savename)
        
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close()

    
    print('Done')

if __name__ == "__main__":  
    distributions()
