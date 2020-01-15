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
    args = parser.parse_args()
   
    #Read data

    if not args.f:
        print('No input file found')
        quit()
    elif args.f:
        r, th, gr, gth, g = np.load(args.f)

    #Average

    # Plot
    
    fig = plt.figure(figsize=(24,16))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233, polar=True)
    ax4 = fig.add_subplot(234, polar=True)
    ax5 = fig.add_subplot(235, polar=True)

    ax1.plot(r,gr,label='dr = '+format(dr,'.1e'))
    ax1.set_xlabel('$r$')
    ax1.set_ylabel('$g(r)$')
    ax1.grid()
    ax1.set_xlim(0)
    ax1.set_ylim(0)
    ax1.legend()
    ax1.set_title('Radial distribution function')
   
    ax2.plot(th,gth,label='dth = '+format(dth,'.1e'))
    ax2.legend()
    ax2.set_title('Angular distribution function')

    im = ax3.contourf(T,R,g,cmap="jet",levels=100)
    ax3.set_title('2D distribution function')
    fig.colorbar(im,ax=ax3)
    
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
    
    im = ax5.contourf(Tsym,Rsym,gsym,cmap="jet",levels=100)
    ax5.set_title('4-Quadrant averaged 2D distribution function')
    ax5.set_thetamin(0)
    ax5.set_thetamax(90)
    fig.colorbar(im,ax=ax5)
    
    lastslash = args.f.rfind('/')  
    savename = args.f[0:lastslash+1]+'DF_'+args.f[lastslash+1:-4]+'.png'
    plt.savefig(savename)
    print('Plot saved as',savename)
        
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()

    print('Done')

if __name__ == "__main__":  
    plot_DF()
