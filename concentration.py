#######################################################
##            Concentration Gradient Calculator      ##
##                 Written By Ben Chun               ##
#######################################################

import numpy as np
from random import seed
from random import randint
from scipy import special 
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
import math
import pickle

############################
#### Diffusion function ####
def func(r,t,D):
    rt = r/(2*np.sqrt(D*t))
    erf = special.erfc(rt)
    return erf
############################

#############################################################################################################################
### Distance of Cell from the location of Damaged Cell
### This will be used to calculate the released substance concentration from the damaged cell at the given cell location 
def DistCelltoOrigin(CellCoord,
                     OriginOfExternalSignal):

    NoOfCell         = np.shape(CellCoord)[0]
    xOrigin          = OriginOfExternalSignal[0]  
    CelltoOrigin_r   = abs(np.transpose(CellCoord)[0] - xOrigin)
    
    return CelltoOrigin_r # array

####################################################################################################################
### Distance among cells 
### This will be used to calculate the released substance concentration from individual cell at the given cell location 
def DistCelltoCell(CellCoord):
    
    CelltoCell_r = cdist(CellCoord,CellCoord)
       
    return CelltoCell_r


####################################################################################################################
### Distance between cells and ref point
### This will be used to calculate the released substance concentration from individual cell at the given cell location 
def DistCelltoPoint(CellCoord,ref):
    
    sqCoord = (CellCoord - ref)**2
    sqCoordx = np.transpose(sqCoord)[0]
    sqCoordy = np.transpose(sqCoord)[1]
    DistList = (sqCoordx + sqCoordy)**0.5
    
    return DistList

#######################################################################
### Concentration of Chemoattractant released from the external source in the path
'''
[Note] 3/6/22 by Ben 
This should be off... 
It sounds like the neighboring cells would contribute to amplifying the motility of cells
but we may approach this as amplifying "migration" rather than "motility".
'''
def ConcByOrigin(CelltoOrigin_r,
                 time_state,
                 Diff,
                 ExtATP,
                 DiffState,
                 OriginOfExternalSignal):
    
    xOri = OriginOfExternalSignal[0]
    if DiffState == 'linear':
        Conc = (1-CelltoOrigin_r/xOri)*ExtATP
    
    elif DiffState == 'steady': # this will be used in the right-handside reservoir
        Conc = np.ones(len(CelltoOrigin_r))*ExtATP
    
    elif DiffState == 'error': # this will be used 
        Conc = func(CelltoOrigin_r,time_state,Diff)*ExtATP
        #Note: once cells enter the right hand side of reservoir, the calculation needs to be switched to "Steady"
    
    return Conc


#######################################################################
### Concentration of Chemoattractant released from the external source in the cell reservoir
'''
[Note]: 02/27/2022 
This function is to differentiate the distribution of attractant with respect to the area due to 
the presence of difference in the openness among area and corner within the structure. 
'''
def ExtConc(CellCoord,
            time_state,
            Diff,
            ExtATP,
            OriginOfExternalSignal,
            minXpath,
            maxYpath,
            minYpath):
    '''
    __________
      Area 3  |
    _ _ _ _ _ A______________
              |
      Area 2       Area 1
    _ _ _ _ _ |______________
      Area 4  B
    __________|
    
    
    '''

    # Take Coordinate 
    RefCoord = CellCoord.copy()
    RefCoordX = np.transpose(RefCoord)[0]
    RefCoordY = np.transpose(RefCoord)[1]
    marker1 = RefCoordX.copy()
    marker2 = RefCoordY.copy()
    marker3 = RefCoordY.copy()
    marker4 = RefCoordY.copy()
    marker5 = RefCoordX.copy()
    marker6 = RefCoordX.copy()
    
    marker1[marker1 <= minXpath/10] = 0
    marker1[marker1 > minXpath/10] = 1
    
    marker2[marker2 > minYpath/10] = 999999
    marker2[marker2 < minYpath/10] = 1
    marker2[marker2 == 999999] = 0
    
    marker3[marker3 <= minYpath/10] = 999999
    marker3[marker3 >= maxYpath/10] = 999999
    marker3[marker3 != 999999] = 1
    marker3[marker3 == 999999] = 0 # all 1 and zeros 
    
    marker4[marker4 > maxYpath/10] = 999999
    marker4[marker4 < maxYpath/10] = 0
    marker4[marker4 == 999999] = 1
    
    marker5[marker5 >= minXpath/10] = 999999
    marker5[marker5 < minXpath/10] = 1
    marker5[marker5 == 999999] = 0
    
    marker6 = marker1.copy() - np.ones(len(marker6))
    marker6[marker6 == -1] = 1 
    
    # Area 1 & 2
    ref1 = OriginOfExternalSignal[0]
    refDist1 = abs(RefCoordX - ref1)
    conc1 = func(refDist1,time_state,Diff)*ExtATP*marker3*marker1
    
    # Area 2 - Source of attractant is the boundary between Area 1 & 2 
    # Assuming diffusion in reservoir would be slighly slower 
    DiffRev = Diff*0.75
    ref2 = minXpath/10
    refDist2 = abs(RefCoordX - ref2)
    refOri1 = abs(ref1 - minXpath)
    refConc1 = func(refOri1,time_state,Diff)*ExtATP
    conc2 = func(refDist2,time_state,DiffRev)*refConc1*marker6*marker3
    
    # Area 3 - Source of attractant is the point A 
    ref3 = [minXpath/10,maxYpath/10]
    refDist3 = DistCelltoPoint(RefCoord,ref3)
    conc3 = func(refDist3,time_state,DiffRev)*refConc1*marker4*marker5
    
    # Area 4 - Source of attractant is the point B 
    ref4 = [minXpath/10,minYpath/10]
    refDist4 = DistCelltoPoint(RefCoord,ref4)
    conc4 = func(refDist4,time_state,DiffRev)*refConc1*marker2*marker5

    Conc = conc1 + conc2 + conc3 + conc4
    
    indexRef = 10

    return Conc

##########################################################################################
### Concentration of Chemoattractant reased from the indivudal cells
def ConcByCell(CelltoCell_r,
               odes, # this is being used to simulate ODE-based autocrine ... We may not want this now.
               # Cells can't convert themselves into another state within the given amount time. 
               time_state,
               DiffRate,
               cellConc,
               stateVariable,
               AutoCrineMarker):
    
    if isinstance(odes, (int,float)):
        ReleaseSig = 1
    else:
        ReleaseSig = odes['S1']
        ReleaseSig[ReleaseSig < 1e-14] = 1e-14
        
    NoOfCell = len(AutoCrineMarker)
    kd = 0.75 # Sensitivity of Autocrine
    Hill = stateVariable*cellConc*np.asarray(AutoCrineMarker)/(1+(kd/ReleaseSig)**5)  
    ConcByCell = np.asarray(Hill)*func(CelltoCell_r,time_state,DiffRate).sum(axis=0)
    
    return ConcByCell


#######################################################################################################
### Generate the force of migration based on the gradient of chemoattractant along wiht Brownian Motion
'''
[Note] 3/6/22 by Ben 
Here is a problem. If we tweak the force applied on the particles, we may accelerate the migration in the second order fashion, 
which we are not quite sure that is the case. 
However, it is true that as ATP or chemoattranctant is diffusing throughout the domain, the cell must gain extra motility and wiggle with much larger margin. 
That is being controlled by the system temperature, not by force. 
Unless we differentiate the chemoattractant and ATP (just boosting motility), this may require major implementation. 
'''
def forceGen(Origin,
             CellCoord,
             time_state,
             DiffRate,
             searchingRange,
             highBC,
             UnitLength,
             DispScale,
             cellConc,
             ExtATP,
             DiffState,
             odes1, #
             odes2, #
             time_step,
             stateVariable, #
             boxFactor,
             pathFactor,
             MigMarker,
             P2Y_resting,
             P2Y_activated,
             total_conc_at_cell,
             minYpath,
             maxYpath,
             minXpath,
             AutoCrineMarker):
    
    RefCoord = np.transpose(CellCoord)[0].copy()
    CellCoord_old = np.transpose(CellCoord)[0]
    NoOfCell = np.shape(CellCoord)[0]
    highBCx, highBCy, highBCz = highBC
   
    SecondReservoir = UnitLength*(boxFactor + pathFactor+1)/10
    RefCoord[RefCoord <= SecondReservoir] = 1
    RefCoord[RefCoord >  SecondReservoir] = 9999999999
    n = 0
    
    if DiffState == 'error' or DiffState == 'linear':
        CellCoord = np.transpose(CellCoord)
        # x
        xtest = np.array([CellCoord[0]-searchingRange,CellCoord[0],CellCoord[0]+searchingRange])
        # y
        ytest = np.array([CellCoord[1]-searchingRange,CellCoord[1],CellCoord[1]+searchingRange])
        
        '''
                  x1y2
                    |
         x0y1 --- x1y1 --- x2y1
                    |
                  x1y0
        '''
        p01 = np.transpose(np.array([xtest[0],ytest[1]]))
        p11 = np.transpose(np.array([xtest[1],ytest[1]])) 
        p21 = np.transpose(np.array([xtest[2],ytest[1]]))
        p10 = np.transpose(np.array([xtest[1],ytest[0]]))
        p12 = np.transpose(np.array([xtest[1],ytest[2]]))
        
        Conc01 = ExtConc(p01,time_state,DiffRate,ExtATP,Origin,minXpath,maxYpath,minYpath)
        Conc11 = ExtConc(p11,time_state,DiffRate,ExtATP,Origin,minXpath,maxYpath,minYpath)
        Conc21 = ExtConc(p21,time_state,DiffRate,ExtATP,Origin,minXpath,maxYpath,minYpath)
        Conc10 = ExtConc(p10,time_state,DiffRate,ExtATP,Origin,minXpath,maxYpath,minYpath)
        Conc12 = ExtConc(p12,time_state,DiffRate,ExtATP,Origin,minXpath,maxYpath,minYpath)
        
        COarray = np.array([Conc01,  #xmy
                            Conc21,  #xpy
                            Conc10,  #xym
                            Conc12,  #xyp
                            Conc11]) #xy

        xcell = CellCoord[0]
        ycell = CellCoord[1]
    
        Conc = np.zeros(NoOfCell)
    
        dirFacX1 = np.zeros(NoOfCell)
        dirFacY1 = np.zeros(NoOfCell)
        
        # without Autocrine 
        if cellConc == 0.0:
            Concxmy =  COarray[0]
            Concxpy =  COarray[1]
            Concxym =  COarray[2]
            Concxyp =  COarray[3]
            Conc  = COarray[4]
            
            dx = (Concxpy - Concxmy)/2
            dy = (Concxyp - Concxym)/2
            
            dirFacX1 = np.random.normal(dx,abs(dx)*2)
            dirFacY1 = np.random.normal(dy,abs(dy)*2)
            
            dirFacX1[abs(dirFacX1) <= 1e-14] = 0
            dirFacX1[dirFacX1 > 1e-14] = 1
            dirFacX1[dirFacX1 < -1e-14] = -1
            dirFacY1[abs(dirFacY1) <= 1e-14] = 0
            dirFacY1[dirFacY1 > 1e-14] = 1
            dirFacY1[dirFacY1 < -1e-14] = -1
            
            deltaX = abs(dx)
            deltaY = abs(dy)
        
        # with Autocrine    
        else:                
            # This is where I should have no diffusion in backward direction 
            xcell_test = xcell.copy()
            dummy1 = np.transpose(np.ones([len(xcell_test),len(xcell_test)])*xcell_test)-xcell_test
            dummy1[dummy1>0] = 1
            dummy1[dummy1<=0] = 1e14
        
            xmy = p01 
            xpy = p21 
            xym = p10 
            xyp = p12 
        
            Coord = np.transpose(CellCoord)
            distxmy = cdist(Coord,xmy)*dummy1
            distxpy = cdist(Coord,xpy)*dummy1
            distxym = cdist(Coord,xym)*dummy1
            distxyp = cdist(Coord,xyp)*dummy1

            Concxmy = ConcByCell(distxmy,odes1,time_state,DiffRate,cellConc,stateVariable,AutoCrineMarker)*RefCoord + COarray[0]
            Concxpy = ConcByCell(distxpy,odes1,time_state,DiffRate,cellConc,stateVariable,AutoCrineMarker)*RefCoord + COarray[1]
            Concxym = ConcByCell(distxym,odes1,time_state,DiffRate,cellConc,stateVariable,AutoCrineMarker)*RefCoord + COarray[2]
            Concxyp = ConcByCell(distxyp,odes1,time_state,DiffRate,cellConc,stateVariable,AutoCrineMarker)*RefCoord + COarray[3]
            
            Concxmy[Concxmy>1000]=1e-15
            Concxpy[Concxpy>1000]=1e-15
            Concxym[Concxym>1000]=1e-15
            Concxyp[Concxyp>1000]=1e-15
                                   
            dx = (Concxpy - Concxmy)/2
            dy = (Concxyp - Concxym)/2
            
            dirFacX1 = np.random.normal(dx,abs(dx))
            dirFacY1 = np.random.normal(dy,abs(dy))
            
            dirFacX1[abs(dirFacX1) <= 1e-14] = 0
            dirFacX1[dirFacX1 > 1e-14] = 1
            dirFacX1[dirFacX1 < -1e-14] = -1
            dirFacY1[abs(dirFacY1) <= 1e-14] = 0
            dirFacY1[dirFacY1 > 1e-14] = 1
            dirFacY1[dirFacY1 < -1e-14] = -1
            
            deltaX = abs(dx)
            deltaY = abs(dy)
                    

    elif DiffState == 'steady':
        CellCoord = np.transpose(CellCoord)
        deltaX = np.zeros(NoOfCell)
        deltaY = np.zeros(NoOfCell)
        Conc = np.ones(NoOfCell)*ExtATP
        dirFacX1 = np.ones(NoOfCell) 
        dirFacY1 = np.ones(NoOfCell)
    
    xrand = np.random.normal(0,0.5,NoOfCell)
    yrand = np.random.normal(0,0.5,NoOfCell)
    
    dirFacX2 = xrand/abs(xrand)
    dirFacY2 = yrand/abs(yrand)
    
    
    ##############################################################
    ## -------------------------------------------------------- ##
    ## Force Conversion as a function of gradient/concentration ##
    ## -------------------------------------------------------- ##
    ##############################################################
    # Setting up the initial/previous values
    '''
    [Note]: 02/27/2022 by Ben 
    Need to add non-ODE calculation
    
    '''
    Ix = odes2['Ix'] 
    Iy = odes2['Iy'] 
    DMx = odes2['DMx'] 
    DMy = odes2['DMy'] 
    
    kd = 2
    signalCoord = CellCoord[0].copy()
    signalCoord[signalCoord > SecondReservoir] = -1
    signalCoord[signalCoord <= SecondReservoir] = 1 
    
    if isinstance(odes1 , (int,float)):
        MigSig = signalCoord
    else:
        MigSig = signalCoord*odes1['S2'] 
    
    MigSig[MigSig < 0] = 1
    
    deltaX[deltaX < 1e-16] = 1e-16
    deltaY[deltaY < 1e-16] = 1e-16
        
    MigSig2 = 0.1/(1+(0.0005/deltaX)**5)
    MigSig3 = 0.1/(1+(0.0005/deltaY)**5)
    
    # Directed migration related constants
    kfD = np.random.normal(0.8,0.4)
    kbD = np.random.normal(0.02,0.04)

    # Undireted migration related constants 
    kfU = np.random.normal(0.2,0.1)
    kbU = np.random.normal(0.01,0.02)
    
    checker = CellCoord[0].copy()
    checker[checker < SecondReservoir] = 1
    checker[checker > SecondReservoir] = 0  

    Inewx = Ix + ((kbD*DMx) - (np.asarray(MigSig)*MigSig2*kfD)*Ix)*time_step 
    Inewy = Iy + ((kbD*DMy) - (np.asarray(MigSig)*MigSig3*kfD)*Iy)*time_step
    DMnewx = DMx + (np.asarray(MigSig)*MigSig2*kfD*Ix - kbD*DMx)*time_step
    DMnewy = DMy + (np.asarray(MigSig)*MigSig3*kfD*Iy - kbD*DMy)*time_step
    FVectorX = DispScale*(DMnewx*dirFacX1*stateVariable*checker*np.asarray(MigMarker))
    FVectorY = DispScale*(DMnewy*dirFacY1*stateVariable*checker*np.asarray(MigMarker))
    FVectorZ = np.zeros(NoOfCell)

    odesNew = {'Ix': Inewx,
               'Iy': Inewy,
               'DMx': DMnewx,
               'DMy': DMnewy}

    return FVectorX, FVectorY, FVectorZ, odesNew 
