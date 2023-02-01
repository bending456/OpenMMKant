########################################################
#### Sub functions that need for FeedInForce plugin ####
#### Writtne by Ben chun ###############################
########################################################

from simtk.unit import *
from simtk.openmm import *
from simtk.openmm.app import *
from sys import stdout
import numpy as np
#from FeedInplugin import FeedInForce
from random import seed
from random import randint
import concentration as conc
import random
import scipy
from scipy.optimize import curve_fit
from scipy.integrate import odeint
import yaml 
import time as timer
from scipy.spatial.distance import cdist
import math

'''
[Note]: 02/22/2022 by Ben 
Factoring started

'''

#######################################################################################
###                           Cell/Particle Distributor                             ###
#######################################################################################
'''
[Note] 02/22/2022 by Ben 
Should I create an individual genCellCoord? and plus, indicate the dimension? 
Randomize the migration factor but distrubtion is correlated to the ratio between resting and activated! 
'''
def genCellCoord3D(Density1,               # resting cells in first compartment 
                   Density2,               # num of cells in the second compartment 
                   numOfDeadCell,
                   num_BP,
                   perLine,
                   min_dist_among_cells,
                   resv_dim,               # [maxXpath, minXpath, maxYpath, minYpath]
                   boundary_dim,
                   P2Y_resting, # these two values will determine the migration speed. 
                   P2Y_activated, # instead of having two, maybe we can randomize it but proportional to the ratio between resting and activated. 
                   DiffState,
                   simType
                   ):
    
    [maxXpath, minXpath, maxYpath, minYpath] = resv_dim
    [UnitLength, 
     Indentation, 
     boxFactor, 
     pathFactor, 
     pathWFactor, 
     wallFactor] = boundary_dim
    
    marker = []
    MigMarker = []
    totalmarker = []
    coordtoYaml = {}
    
    # dimensional parameters
    minDist = min_dist_among_cells
    minDistforDead = 10
    bumper = 10
    
    totalNum = 0 
    
    ### Configuring dimension where particles are placed
    UnitL = UnitLength
    # Reservoir Dimension
    if simType == 'box':
        maxX1 = UnitL*boxFactor - bumper
        minX1 = bumper
        maxY1 = UnitL*boxFactor - bumper
        minY1 = bumper
        Area1 = maxX1*maxY1
    else:
        centYpath = (maxYpath + minYpath)/2
        minYbox = centYpath - UnitL*wallFactor/3
        maxYbox = centYpath + UnitL*wallFactor/3
        boxLength = maxYbox - minYbox
        maxX1 = boxLength - bumper 
        minX1 = bumper  
        maxY1 = maxYbox - bumper
        minY1 = minYbox + bumper
        Area1 = maxX1 * boxLength
    
    # Path Dimension - this has been calculated previously
    [maxXpath, minXpath, maxYpath, minYpath] = resv_dim
    
    # Boundary Dimension 
    [UnitLength, Indentation, boxFactor, pathFactor, pathWFactor, wallFactor] = boundary_dim
    
    # Placing cells in the FIRST compartment
    #----------------------------------------------------------------
    # DeadCells First
    '''
    [Note]: 02/22/2022 by Ben 
    Cluster is being placed in orderly fashion such as 
    __________________
                      |
                      |
                      |
                      |
                      |___________________
                        X     X
                           X
                        X     X
                           X
                        X     X
                      ____________________
                      |
                      |
                      |
                      |
    __________________|
    
    *This mode is applicapable only with slab or migration assay model. 
    **Since the location of the individual cluster is pre-determined, the number of dead cells/clusters is also pre-set based on the dimension of path.
    ***When "Square Box" model is being used, the number of Dead Cell is set to be zero 
    
    '''
    numP_DeadCell = 0
    loopcounter1 = 1

    if numOfDeadCell == 0:
        xo_dead_cell = [999999999999999]
        yo_dead_cell = [999999999999999]
    else:
        depthOfpath = maxYpath - minYpath 
        numPer1stRow = math.ceil(depthOfpath/10)
        y1 = np.linspace(5,depthOfpath-5,numPer1stRow) + minYpath
        y2 = np.linspace(10,depthOfpath-10,numPer1stRow-1) + minYpath 
        x = np.linspace(10,100,10) + minXpath
        numOfDeadCell = (numPer1stRow*2 - 1) * 5
        
        loopcounter1 = 0
        for n, x_element in enumerate(x):              
            if x_element%20 == 10:
                for n in np.arange(numPer1stRow):
                    if loopcounter1 == 0:
                        xo = [x_element]
                        yo = [y1[n]]
                        coord = [[x_element,y1[n],0]]
                        loopcounter1 += 1
                    else:
                        xo.append(x_element)
                        yo.append(y1[n])
                        coord.append([x_element,y1[n],0])
                        loopcounter1 += 1
                        
                    numP_DeadCell += 1
                    totalmarker.append('DC')
                    
            elif x_element%20 == 0: 
                for n in np.arange(numPer1stRow-1):
                    xo.append(x_element)
                    yo.append(y2[n])
                    coord.append([x_element,y2[n],0])
                    loopcounter1 += 1
                    numP_DeadCell += 1
                    totalmarker.append('DC')
        
            
        xo_dead_cell = xo
        yo_dead_cell = yo
    #----------------------------------------------------------------
    
    #----------------------------------------------------------------
    # Live Cells 
    ## Compartment (Reservoir #1: All Resting Cells)
    '''
    [Note]: 2/26/22
    Although all the cells are considered to be "resting" or "inactive", which indicates higher rate of migration, 
    their migration capacity should be randomized. - done
    '''
    numP_Comp1 = 0
    loopcounter1 = 1
    NoCell1 = Density1*Area1
    while numP_Comp1 < NoCell1 and loopcounter1 < 200000*NoCell1:
        xpossible = randint(minX1,maxX1)
        ypossible = randint(minY1,maxY1)
        # migration factor randomizer - Using normal distribution 
        mig_factor = P2Y_resting*(1-abs(np.random.normal(loc=0, scale=0.2)))
        if numOfDeadCell == 0 and numP_Comp1 == 0:
            xo = [xpossible]
            yo = [ypossible]
            coord = [[xpossible,ypossible,0]]
            marker.append('resting')
            MigMarker.append(mig_factor) 
            totalmarker.append('RC')
            numP_Comp1 += 1
            continue
            
        distance1 = np.sqrt((np.asarray(xo)-xpossible)**2 + (np.asarray(yo)-ypossible)**2)
            
        if min(distance1) >= minDist: 
            xo.append(xpossible)
            yo.append(ypossible) 
            coord.append([xpossible,ypossible,0])
            marker.append('resting')
            MigMarker.append(mig_factor)
            totalmarker.append('RC')
                
            numP_Comp1 += 1
        
        loopcounter1 = loopcounter1 + 1
            

    #----------------------------------------------------------------
    ## Compartment (Path #2: All Resting Cells or Activated Cells) -> Why is this?
    NoCell2 = (maxXpath - minXpath) * (maxYpath - minYpath) * Density2
    if NoCell2 > 0: # true or false
        numOfCompartment = 5
        LengthOfPath = maxXpath - minXpath + 0.5*UnitL
        UnitLenOfPath = LengthOfPath/numOfCompartment
        
        FixedLength = UnitL*boxFactor
        
        if NoCell2 < 15:
            print("-- Error: Please assign higher density so that more than 15 cells are placed --")
            
        numOfCellUnit = math.floor(NoCell2/15)
        diffNum = NoCell2 - numOfCellUnit*15
        
        density_of_unit = np.linspace(0,Density2,5)

        maxX = UnitLenOfPath*np.linspace(1,5,5) + FixedLength
        minX = UnitLenOfPath*np.linspace(0,4,5) + FixedLength
        num = density_of_unit*UnitLenOfPath*(maxYpath-minYpath)
        num = num.tolist()
        
        if DiffState == 'steady':
            num.reverse()
        
        xo2 = []
        yo2 = []
        
        numP_Comp2 = 0
        for i,n in enumerate(num):
            loopcounter2 = 1
            numP2 = 0
            
            while numP2 < n and loopcounter2 < 200000*n:
                xpossible = randint(math.ceil(minX[i]),math.ceil(maxX[i]))
                ypossible = randint(minYpath+4,maxYpath-4)  
                    
                if numP2 == 0:
                    xo2.append(xpossible)
                    yo2.append(ypossible)
                    coord.append([xpossible,ypossible,0])
                    
                    if DiffState != 'steady':
                        mig_factor = P2Y_activated*(1-abs(np.random.normal(loc=0, scale=0.2)))
                        marker.append('activated')
                        MigMarker.append(mig_factor)
                        totalmarker.append('AC')
                    
                    elif DiffState == 'steady':
                        mig_factor = P2Y_resting*(1-abs(np.random.normal(loc=0, scale=0.2)))
                        marker.append('resting')
                        MigMarker.append(mig_factor)
                        totalmarker.append('RC')
                    
                    numP2 +=1
                    numP_Comp2 += 1
                    continue
                    
                else:
                    distance2 = np.sqrt((np.asarray(xo2)-xpossible)**2 + (np.asarray(yo2)-ypossible)**2)
                
                # measuring distance between dead cell cluster and the cells within the 2nd compartment
                if numOfDeadCell == 0:
                    distance3 = 9999
                else:
                    distance = np.sqrt((np.asarray(xo_dead_cell)-xpossible)**2 + (np.asarray(yo_dead_cell)-ypossible)**2)
                    distance3 = min(distance)
                    
                if min(distance2) >= minDist and distance3 >= minDistforDead/2:
                    xo2.append(xpossible)
                    yo2.append(ypossible)
                    coord.append([xpossible,ypossible,0])
                    
                    if DiffState != 'steady':
                        mig_factor = P2Y_activated*(1-abs(np.random.normal(loc=0, scale=0.2)))
                        marker.append('activated')
                        MigMarker.append(mig_factor)
                        totalmarker.append('AC')
                        
                    elif DiffState == 'steady':
                        mig_factor = P2Y_resting*(1-abs(np.random.normal(loc=0, scale=0.2)))
                        marker.append('resting')
                        MigMarker.append(mig_factor)
                        totalmarker.append('RC')
                    numP2 += 1
                    numP_Comp2 += 1
                            
                loopcounter2 += 1
    else:
        numP_Comp2 = 0
    
    #----------------------------------------------------------------
    
   
    if simType == 'box':
        numP_BC = 0
        WallP = int(UnitL*boxFactor/perLine)
        for n in np.linspace(0,UnitL*boxFactor-perLine,WallP):
            x1 = 0
            y1 = n
            numP_BC += 1 
            coord.append([x1,y1,0])
            totalmarker.append('BC')
            
            x2 = n
            y2 = 0
            numP_BC += 1 
            coord.append([x2,y2,0])
            totalmarker.append('BC')
            
            x3 = UnitL*boxFactor
            y3 = n
            numP_BC += 1 
            coord.append([x3,y3,0])
            totalmarker.append('BC')
            
            x4 = n
            y4 = UnitL*boxFactor
            numP_BC += 1 
            coord.append([x4,y4,0])
            totalmarker.append('BC')
    else:    
        '''
         8                    
      ______               
     |      |      5       |
     |   4  |______________| 6  maxYpath
   9 |       ______________
     |   1  |              | 3  minYpath
     |______|      2       |
         7                    
    
        '''
    ## Boundary Particles
        centYpath = (maxYpath + minYpath)/2
        
        minYbox = centYpath - UnitL*wallFactor/3
        maxYbox = centYpath + UnitL*wallFactor/3
        
        In_p = int(Indentation*wallFactor/perLine)
        Pa_p = int((maxXpath - minXpath)/perLine)
        Re_p = int((minYpath - minYbox)/perLine)
        Re_p2 = int(maxXpath/(perLine*2))
        Wa_p = int((maxYpath + minYpath)*1.5/perLine)
        
    
        #1 - here
        numP_BC = 0
        for n in np.linspace(minYbox,minYpath-perLine,Re_p):
            x = minXpath
            y = n
            numP_BC += 1 
            coord.append([x,y,0])
            totalmarker.append('BC')
        #2
        for n in np.linspace(minXpath,maxXpath-perLine,Pa_p):
            x = n
            y = minYpath
            numP_BC += 1
            coord.append([x,y,0])
            totalmarker.append('BC')
        #3
        for n in np.linspace(minYpath,minYbox,Re_p):
            x = maxXpath
            y = n
            numP_BC += 1
            coord.append([x,y,0])
            totalmarker.append('BC')
        #4 - here
        for n in np.linspace(maxYbox,maxYpath+perLine,Re_p):
            x = minXpath
            y = n
            numP_BC += 1
            coord.append([x,y,0])
            totalmarker.append('BC')
        #5
        for n in np.linspace(minXpath,maxXpath-perLine,Pa_p):
            x = n
            y = maxYpath
            numP_BC += 1
            coord.append([x,y,0])
            totalmarker.append('BC')
        #6
        for n in np.linspace(maxYpath,maxYbox,Re_p):
            x = maxXpath
            y = n
            numP_BC += 1
            coord.append([x,y,0])
            totalmarker.append('BC')
        #7
        for n in np.linspace(0,minXpath-perLine,Re_p2):
            x = n
            y = minYbox
            numP_BC += 1
            coord.append([x,y,0])
            totalmarker.append('BC')
        #8
        for n in np.linspace(0,minXpath-perLine,Re_p2):
            x = n
            y = maxYbox
            numP_BC += 1
            coord.append([x,y,0])
            totalmarker.append('BC')
    
        #9
        for n in np.linspace(minYbox,maxYbox-perLine,Wa_p):
            x = 0
            y = n
            numP_BC += 1
            coord.append([x,y,0])
            totalmarker.append('BC')

    
    #-----------------------------------------------------------------------------
    coord = np.asarray(coord)/10 #<--------- A to nm
    ## Constitution of Cell Population
    Cell_Constitution = {'Compartment1': numP_Comp1,
                         'Compartment2': numP_Comp2,
                         'Dead': numP_DeadCell,
                         'BP' : numP_BC}
    return coord, marker, MigMarker, totalmarker, Cell_Constitution


###########################################################################
###                           PDB Generator                             ###
###########################################################################

def genPDBWrapper(
        pdbFileName,
        nCellsType1,   # generally the diffusing cells 
        nCellsType2=0, # generally the crowders 
        startingPositions=None):
    """
    This is a wrapper for Ben's PDB writer, siunce I needed something simple.
    Later I can revise his code to accept coordinates; currently they're randomized 
    """
    PDBgenNoPBC(pdbFileName,nCellsType1,nCellsType2,0,0,"None",startingPositions=startingPositions)
    

def PDBgenNoPBC(PDBfileName,
                NoCell1, 
                NoCell2,
                numOfDeadCell,
                num_BP,
                DiffState,  
                startingPositions=None):
    '''
    
    [Note]: 11/06/2020 by Ben 
    In this function, having all resting cells or activated cells will not cause any error.
    
    [Parameter Description]
    PDBfileName:    pdb file name
    NoCell1:        a total number of cells in the 1st half of box: should be resting cells
    NoCell2:        a total number of cells in the 2nd half of box: should be activated cells 

    '''    
    # --- Writing pdb file initiated ---
    daFile = PDBfileName
    if ".pdb" not in daFile:
        daFile = daFile+".pdb"
    structure = open(daFile,"w") 
    structure.write("MODEL     1 \n")

    '''
    [------First half of box-------]
    '''
    # determining a total number of resting cells from the overall population
    ## the rest of cells are activated cells
    
    TotalNoCell = NoCell1 + NoCell2 + numOfDeadCell + num_BP
    refnum1 = NoCell1 + NoCell2
    refnum2 = refnum1 + num_BP
    refnum3 = numOfDeadCell + NoCell1
    refnum4 = refnum3 + NoCell2
    
    ## Dead Cell first: There is no distinction between 1st and 2nd compartment for the dead cells 
    for i in np.arange(TotalNoCell):
        # add random positions if starting positions are not defined 
        if startingPositions is None:
          x = randint(0,9)
          y = randint(0,9)
          z = randint(0,9)
        else:
          #print(np.shape(startingPositions))
          x,y,z = startingPositions[i,:]

        # make sure num sig. figs is correct for pdb
        x = format(x,'8.3f') # any changest to this need to be reflected in spacing below 
        y = format(y,'8.3f')
        z = format(z,'8.3f')
        
        if numOfDeadCell == 0:
            if i < NoCell1:
                name = 'RC'
            elif NoCell1 <= i < refnum1 :
                if DiffState =='steady':
                    name = 'RC'
                else:
                    name = 'AC'
            else:
                name = 'BC'
        else:
            if i < numOfDeadCell:
                name = 'DC'
            elif numOfDeadCell <= i < refnum3: 
                name = 'RC'
            elif refnum3 <= i < refnum4:
                if DiffState == 'steady':
                    name = 'RC'
                else:
                    name = 'AC'
            else:
                name = 'BC'

        if i < 9:
            structure.write("ATOM      "+str(int(i+1))+"  "+name+"   "+name+"     "    +str(int(i+1))+"    "+str(x)+""+str(y)+""+str(z)+"  1.00  0.00 \n")
        elif i >= 9 and i < 99:
            structure.write("ATOM     "+ str(int(i+1))+"  "+name+"   "+name+  "    "   +str(int(i+1))+"    "+str(x)+""+str(y)+""+str(z)+"  1.00  0.00 \n")
        elif i >= 99 and i < 999:
            structure.write("ATOM    "+  str(int(i+1))+"  "+name+"   "+name+    "   "  +str(int(i+1))+"    "+str(x)+""+str(y)+""+str(z)+"  1.00  0.00 \n")
        elif i >= 999 and i < 9999:
            structure.write("ATOM   "+   str(int(i+1))+"  "+name+"   "+name+      "  " +str(int(i+1))+"    "+str(x)+""+str(y)+""+str(z)+"  1.00  0.00 \n")
        elif i >= 9999:
            structure.write("ATOM  "+    str(int(i+1))+"  "+name+"   "+name+        " "+str(int(i+1))+"    "+str(x)+""+str(y)+""+str(z)+"  1.00  0.00 \n")
            
    structure.write("ENDMDL")
    structure.close
                   
    return
        

##########################################################################################
### ODE calculation - Intracellular Dynamics 

def intracellularDynamics(odes,
                          stateVariable,
                          marker,
                          ATP,
                          time_state,
                          step_size,
                          P2X_resting,
                          P2Y_resting,
                          P2X_activated,
                          P2Y_activated):
    R = odes['R']
    A = odes['A']
    I = odes['I']
    D = odes['D']
    S1 = odes['S1']
    S2 = odes['S2'] # this should be somewhat different 
    
    # [NOTE]: Assuming that the ATP stimulation is somewhat periodic 
    ATPdynamic = np.asarray(ATP) #*math.cos(0.1*time_state)**2
    ATPdynamic[ATPdynamic < 1e-20] = 1e-20
    
    ## ODE # 1 
    kf1 = 10
    kf2 = 1
    kf3 = 1
    kf4 = 1.5
    kb4 = 0.1
    ks1max = 4
    ks2max = 5
    ksdeg1 = 2
    ksdeg2 = 0.05
    
    dRdt = -kf1*ATPdynamic*R + kf3*I 
    dAdt = kf1*ATPdynamic*R - kf2*A
    dIdt = kf2*A - kf3*I - kf4*I + kb4*D 
    dDdt = kf4*I - kb4*D 
    
    Rnew = R + step_size*dRdt
    Anew = A + step_size*dAdt
    Inew = I + step_size*dIdt
    Dnew = D + step_size*dDdt
    
    
    ## ODE #2
    
    P2X_exp = np.asarray(marker.copy())
    P2X_exp[P2X_exp=='resting'] = int(P2X_resting)       # low
    P2X_exp[P2X_exp=='activated'] = int(P2X_activated)   # high
    
    P2Y_exp = np.asarray(marker.copy())
    P2Y_exp[P2Y_exp=='resting'] = int(P2Y_resting)       # low
    P2Y_exp[P2Y_exp=='activated'] = int(P2Y_activated) # high
    
    P2X_exp = list(map(float,P2X_exp))
    P2Y_exp = list(map(float,P2Y_exp))
        
    RA = A*np.array(P2X_exp)
    RA[RA < 1e-16] = 1e-16
    
    kmax1 = 4
    kmax2 = 5*np.array(P2Y_exp) 
    
    kd11 = 7
    kd12 = 0.25
    kd21 = 0.25
    kd22 = 0.5
    
    n11 = 6
    n12 = 5
    n21 = 5
    n22 = 2
    #print(ATP)
    ks1 = kmax1*(0.2/(1+(kd11/ATPdynamic)**n11) + 0.8/(1+(kd12/RA)**n12)) ## Autocrinic
    ks2 = kmax2/(1+(kd21/ATPdynamic)**n21 + (kd22/RA)**n22) ## Migration 
        
    dS1dt = ks1 - ksdeg1*S1
    dS2dt = ks2 - ksdeg2*S2
    
    S1new = S1 + step_size*dS1dt
    S2new = S2 + step_size*dS2dt 
    
    odes['R'] = Rnew
    odes['A'] = Anew
    odes['I'] = Inew
    odes['D'] = Dnew
    odes['S1'] = S1new
    odes['S2'] = S2new
    
    ################################
    ## -------------------------- ##
    ## State Variable Calculation ##
    ## -------------------------- ##
    ################################
    # Setting up the initial/previous values
        
    stateVarOld = stateVariable
    ## State Variable Constant
    kf1 = 6e-4 #np.random.normal(1e-4,1e-2)   # from R to Ra 
    kf2 = 2e-4 #np.random.normal(0.5e-4,5e-3) # from A to Aa
    kb1 = 4e-5 #np.random.normal(1e-5,1e-3)   # from Ra to R
    kb2 = 2e-5 #np.random.normal(0.5e-5,5e-4) # from Aa to A

    forward = np.asarray(marker.copy())
    backward = np.asarray(marker.copy())
    forward[forward=='resting'] = kf1
    forward[forward=='activated'] = kf2
    backward[backward=='resting'] = kb1
    backward[backward=='activated'] = kb2
    
    forward = list(map(float,forward))
    backward = list(map(float,backward))
    
    sig = 1/(1+(0.5/ATPdynamic)**3)

    stateVarNew = stateVarOld - step_size*(stateVarOld*sig*forward - stateVarOld*backward) 
        
    return odes, stateVarNew


###############################################################################################
###                        Written by Yehven and implemented by Ben                         ###
###############################################################################################

def calcForce(positions,
              numberOfCells,
              numberOfdead,
              OriginOfExternalSignal,
              ExtATP,
              cellConc,
              Diff,
              time_state,
              UnitLength,
              highBC,
              Indentation,
              DispScale,
              searchingRange,
              marker,
              MigMarker,
              totalmarker,
              DiffState,
              ConcByCell,
              odes1,
              odes2,
              step_size,
              stateVar,
              intradyn,
              stateVariable,
              boxFactor,
              pathFactor,
              P2X_resting,
              P2Y_resting,
              P2X_activated,
              P2Y_activated,
              minXpath,
              maxYpath,
              minYpath):

    
    # Do some magic here that will return substrate field-related force acting on each particle
    n = 1
    m = 0
    dummy_coord = np.zeros([numberOfCells,2]) # since it's 2D
    xcoord = []
    ycoord = []
    
    if stateVar == 'off':
        stateVariable = np.ones(numberOfCells)

    for i in enumerate(positions):
        if totalmarker[n-1] == 'RC' or totalmarker[n-1] == 'AC':
            dummy_coord[m][0] = i[1][0].value_in_unit(nanometers)
            dummy_coord[m][1] = i[1][1].value_in_unit(nanometers)
            xcoord.append(i[1][0].value_in_unit(nanometers))
            ycoord.append(i[1][1].value_in_unit(nanometers))
            m += 1
        n += 1
        
    recordedPositions         = dummy_coord
    xcoord                    = np.asarray(xcoord)
    ycoord                    = np.asarray(ycoord)
    
    # Distance among cells
    CelltoCell_r            = conc.DistCelltoCell(recordedPositions)

    # Done
    if DiffState == 'error':
        ConcByOrigin            = conc.ExtConc(recordedPositions,
                                               time_state,
                                               Diff,
                                               ExtATP,
                                               OriginOfExternalSignal,
                                               minXpath,
                                               maxYpath,
                                               minYpath)
    elif DiffState == 'steady':
        ConcByOrigin = np.ones(numberOfCells)*ExtATP

    # What do we have here? We have the list of local ATP concentration 
    # Need all the ODE here
    
    total_conc_at_cell = ConcByCell + ConcByOrigin # This will be the ATP concentration
    if intradyn == 'on':
        if statvar == 'on':
            odesnew1, stateVariable = intracellularDynamics(odes1,
                                                            stateVariable,
                                                            marker,
                                                            total_conc_at_cell,
                                                            time_state,
                                                            step_size,
                                                            P2X_resting,
                                                            P2Y_resting,
                                                            P2X_activated,
                                                            P2Y_activated)
        else:
            odesnew1, stateVariable = intracellularDynamics(odes1,
                                                            stateVariable,
                                                            marker,
                                                            total_conc_at_cell,
                                                            time_state,
                                                            step_size,
                                                            P2X_resting,
                                                            P2Y_resting,
                                                            P2X_activated,
                                                            P2Y_activated)
            stateVariable = 1
    else:
        odesnew1 = 1
        stateVariable = 1
    
    AutoCrineMarker = np.asarray(marker.copy())
    AutoCrineMarker[AutoCrineMarker == 'activated'] = float('%.2f'%(P2X_activated*(1-abs(np.random.normal(loc=0, scale=0.2)))))
    AutoCrineMarker[AutoCrineMarker == 'resting'] = float('%.2f'%(P2X_resting*(1-abs(np.random.normal(loc=0, scale=0.2)))))
    AutoCrineMarker = [float(i) for i in AutoCrineMarker]
  
    # Done
    ConcByCell            = conc.ConcByCell(CelltoCell_r,
                                            odesnew1,
                                            time_state,
                                            Diff,
                                            cellConc,
                                            stateVariable,
                                            AutoCrineMarker)
    
    total_conc_at_cell = ConcByCell + ConcByOrigin
    #print(total_conc_at_cell)
    #print("------------")
    fvX, fvY, fvZ, odesnew2   = conc.forceGen(OriginOfExternalSignal,
                                              recordedPositions,
                                              time_state,
                                              Diff,
                                              searchingRange,
                                              highBC,
                                              UnitLength,
                                              DispScale,
                                              cellConc,
                                              ExtATP,
                                              DiffState,
                                              odesnew1, #
                                              odes2, #
                                              step_size,
                                              stateVariable,#
                                              boxFactor,
                                              pathFactor,
                                              MigMarker,
                                              P2Y_resting,
                                              P2Y_activated,
                                              total_conc_at_cell,
                                              minYpath,
                                              maxYpath,
                                              minXpath,
                                              AutoCrineMarker)    
    
 
       
    return fvX, fvY, fvZ, ConcByCell, odesnew1, odesnew2, ConcByOrigin, stateVariable, xcoord, ycoord

###############################################################################################
###                        Written by Ben and implemented by Ben                            ###
###############################################################################################

def calcForceModified(numberOfdead,
                      num_BP,
                      UnitLength,
                      Indentation,
                      xcoord,
                      ycoord,
                      fvX,
                      fvY,
                      fvZ):
    
    ##########################################
    ## Constraining the motion of particles ##
    ##########################################
    checkerX = 1
    checkerY = 1
    
    # This is where I can make an adjustment for resting and activated cells   
    force_dead = np.zeros([numberOfdead,3])
    force_border = np.zeros([num_BP,3])
    f_x = fvX*checkerX
    f_y = fvY*checkerY
    f_z = np.zeros(len(xcoord))
    force_live = np.transpose(np.vstack([f_x,f_y,f_z]))
    
    forces = np.vstack([force_dead,force_live,force_border])
          
    return forces

###########################################################################
###                           Test RUN - OpenMM                         ###
###########################################################################
def testRunner(filename):
    pdb = PDBFile(filename+'.pdb')
    forcefield = ForceField('/home/bending456/Ben-Code/Modeling-Codes/Codes/OpenMM_Tutorial/Particle_in_box/Particle_Ben.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=HBonds)
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()
    simulation.reporters.append(PDBReporter('output.pdb', 10))
    simulation.reporters.append(StateDataReporter(stdout, 10, step=True, potentialEnergy=True, temperature=True))
    simulation.step(10)
    
    return

