import os 
import numpy as np 
import yaml
import time as timer
import math
import sys


'''
There should be two version of calculations 
1. Slab structure version 
2. Ring structure version 
'''

###############################################
##               Command Center              ##
## Choose your model of simulation structure ##
##  1. Slab                                  ##
##  2. Square (radial distribution)          ##
###############################################
## Structure 

simLength = 1000*1
DiffState = 'steady'
Density1 = 0.001
Density2 = 0.0
ATP = [1e-14,500] # in uM
pathRatios = 0.5
repeat = 5
Type = 'box' # or box

j = 0
for i in np.arange(repeat):
    for n in ATP:
        inputname = 'test'+str(j+1)
        os.system('python3 runner.py -t '+str(simLength)+' '+
                  '-ExtATP '+str(n)+' '+
                  '-cellConc '+str(0)+' '+
                  '-Density1 '+str(Density1)+' '+
                  '-Density2 '+str(Density2)+' '+
                  '-DiffState '+DiffState+' '+
                  '-name '+str(inputname)+' '+
                  '-input '+str(inputname)+' '+
                  '-numOfDeadCell '+str(0)+' '+
                  '-stateVar '+'off'+' '+
<<<<<<< HEAD
                  '-pathRatio '+str(pathRatios)+' '+
                  '-simType '+Type+' '+
                  '-run') # either box or slab (anything other than box is slab structure)
        j += 1
=======
                  '-pathRatio '+str(n)+' '+
                  '-symType '+'slab'+' '+
                  '-run') # either box or slab (anything other than box is slab structure)
        j += 1
 

>>>>>>> 50fcf88ffcbf0820c8bd43436cc704fd1daa068e
