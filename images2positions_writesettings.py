#!/usr/bin/env python
# coding: utf-8

# Timo van Overveld, 2019

"""Write the settings of the Images2positions script to a file"""

# Imports
import fnmatch
import json
import argparse

def main():
    """Launcher."""

    # Argument parser
    parser = argparse.ArgumentParser()
    #parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument('-f', help="destination file")
    parser.add_argument('-d', help="store file at given location")
    args = parser.parse_args()

    #if args.verbose: print("verbosity turned on")

    # Adjustable settings
    settings = {
        #Directories
        'basedir' : '/media/timo/Backup Plus/PhD/Experimenten/Jeroen/26-11-2019/',
        'calibrationdir' : 'Calibration/',
        'measurementdir' : 'Wave with particles/',

        # refraction indices
        'nair'   : 1.00027717,
        'nwater' : 1.330,

        # Line spacing
        'linespacing' : 0.025, #[m]
        'linespacingpx' : 75, #[px]

        # Channel width
        'channelwidth' : 0.1, #[m]
        
        # Mean water height during measurements
        'Hmean' : 0.07, #[m]

        # Cropping bounds
        'bounds' : [405, 405, 0, 0], # Top, bottom, left, right
        # Central pixels (no lines present)
        'centerpx' : [70,-60],

        # Thresholding values
        'thresholdvalue' : 720,

        # Fitting order
        'warpingorder' : 3,
        'surfaceshapeorder' : 10,

        'plots'   : True,
        'verbose' : True
    }

    if args.d:
        settingsfile = args.d
    else:
        settingsfile = ''

    if args.f:
        settingsfile += args.f
    else:
        settingsfile += 'settings.txt'

    f = open(settingsfile, 'w+')
    f.write(json.dumps(settings))

    print('Constants stored in',settingsfile)

if __name__ == "__main__":
    main()
