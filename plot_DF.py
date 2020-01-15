#!/usr/bin/env python
# coding: utf-8

# Timo van Overveld, 2020

"""Plot the radial, angular and 2D distribution functions of input data"""
# Input: r, theta and g2D (.npz)
# Output: RDF, ADF, 2DDF

# Imports
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

############################################################################
def plot_DF():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='File')
    parser.add_argument('-s', type=str, help='Set of files')
    args = parser.parse_args()
   
    #Read data

    if args.f:
        with np.load(args.f) as data:
            r   = data['arr_0']
            th  = data['arr_1']
            gr  = data['arr_2']
            gth = data['arr_3']
            g   = data['arr_4']

    elif args.s:
        filelist = np.empty(0,dtype=str)
        for file in os.listdir('.'):
            if args.s in file and file.endswith('.npz'):
                filelist = np.append(filelist,file)
        
        print(filelist)
        
        N = np.size(filelist)
        for i in range(0,N,1):
            with np.load(filelist[i]) as data:
                if i == 0: 
                    r   = data['arr_0']
                    th  = data['arr_1']
                    nr  = np.size(r)
                    nth = np.size(th)

                    gr  = np.zeros(nr,dtype=float)
                    gth = np.zeros(nth,dtype=float)
                    g   = np.zeros((nr,nth),dtype=float)
                gr  += data['arr_2']/N
                gth += data['arr_3']/N
                g   += data['arr_4']/N
         
    else:
        print('No input file found')
        quit()
    
    T,R = np.meshgrid(th,r)
    
    dxth = th[5]-th[4]
    halfsize = int(np.size(th)/2)

    thsym  = np.linspace(0,np.pi/2,(np.pi/2)/dxth) # Radii to calculate g
    
    Tsym,Rsym = np.meshgrid(thsym,r)
    
    gsym = np.empty((np.size(r),int(np.size(th)/4)),dtype=float)

    # Calculate the (assumed) symmetric quadrant of g2D
    for i in range(0,int(np.size(th)/4),1):
        gsym[:,i] = (g[:,i]+g[:,-i]+g[:,halfsize+i]+g[:,halfsize-i])/4
    
    #Average

    # Plot
    
    fig = plt.figure(figsize=(24,16))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232, polar=True)
    ax3 = fig.add_subplot(233, polar=True)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235, polar=True)

    ax1.axhline(1,linestyle='--',color='k')
    ax1.plot(r,gr)
    ax1.set_xlabel('$r$')
    ax1.set_ylabel('$g(r)$')
    ax1.grid()
    ax1.set_xlim(0)
    ax1.set_ylim(0)
    ax1.set_title('Radial distribution function')
   
    ax2.plot(th,gth)
    ax2.set_title('Angular distribution function')
    
    levels = np.linspace(0,2,101)
    im = ax3.contourf(T,R,g,cmap="bwr",levels=levels,vmin=0,vmax=2,extend='both')
    ax3.set_title('2D distribution function')
    fig.colorbar(im,ax=ax3,ticks=np.linspace(0,2,11))
    
    ax4.axhline(1,linestyle='--',color='k',label='Uncorrelated system')
    xdir = np.argmin(abs(thsym))
    ydir = np.argmin(abs(thsym-np.pi/2))
    x = np.where(Tsym==thsym[xdir])
    ax4.plot(Rsym[x],gsym[x],label='x-direction')
    x = np.where(Tsym==thsym[ydir])
    ax4.plot(Rsym[x],gsym[x],label='y-direction')
    ax4.grid()
    ax4.set_xlim(0)
    ax4.set_ylim(0)
    ax4.legend()
    ax4.set_title('1D line plots of 2D distribution function')
    
    im = ax5.contourf(Tsym,Rsym,gsym,cmap="bwr",levels=levels,vmin=0,vmax=2,extend='both')
    ax5.set_title('4-Quadrant averaged 2D distribution function')
    ax5.set_thetamin(0)
    ax5.set_thetamax(90)
    fig.colorbar(im,ax=ax5,ticks=np.linspace(0,2,11))
   
    if args.f:
        lastslash = args.f.rfind('/')  
        savename = args.f[0:lastslash+1]+'DF_'+args.f[lastslash+1:-4]+'.png'
    elif args.s:
        savename = args.s+'DF.png'
    plt.savefig(savename)
    print('Plot saved as',savename)
        
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()

    print('Done')

if __name__ == "__main__":  
    plot_DF()
