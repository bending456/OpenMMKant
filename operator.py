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
Density1 = 0.00075
Density2 = 0.0
#ATP = [1e-14,500] # in uM
pathRatios = [0.5,0.8,1]
repeat = 1
Type = 'slab' # or box

j = 0
for i in np.arange(repeat):
    for n in pathRatios:
        inputname = 'test'+str(j+1)
        os.system('python3 runner.py -t '+str(simLength)+' '+
                  '-ExtATP '+str(500)+' '+
                  '-cellConc '+str(0)+' '+
                  '-Density1 '+str(Density1)+' '+
                  '-Density2 '+str(Density2)+' '+
                  '-DiffState '+DiffState+' '+
                  '-name '+str(inputname)+' '+
                  '-input '+str(inputname)+' '+
                  '-numOfDeadCell '+str(0)+' '+
                  '-stateVar '+'off'+' '+
                  '-pathRatio '+str(n)+' '+
                  '-symType '+'slab'+' '+
                  '-run') # either box or slab (anything other than box is slab structure)
        j += 1
 

