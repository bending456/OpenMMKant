#####################################################################################
# This is an example usage of the interface to feed forces into OpenMM with Python. #
#####################################################################################
#import sys

#sys.path.append('../pyscript')

from simtk.unit import *
from simtk.openmm import *
from simtk.openmm.app import *
from sys import stdout
import numpy as np
from FeedInplugin import FeedInForce
from random import seed
from random import randint
import calculator as calc
import time as timer
import os 
import random
import matplotlib.pyplot as plt
import ruamel.yaml


here=os.path.dirname(os.path.abspath(__file__))

'''
[Note]: 3/17/22 by Ben 
Separating path width from the total box size. 

[Note]: 3/6/22 by Ben 
The motility of cells/particles are governed by the system temperature. 
Things should be done. 
1. scale temperature to the experimental concentration. (ATP = [15,40] where 15K is equivalent to 0 ATP and 40K is equivalent to 500 uM ATP.)
2. migration mechanism should be independent from the individual motility driven by the system temperature. 
3. the model should differentiate a compound that amplifies the motility from the compound that induces the directional migration. 

[Note]: 2/26/22 by Ben 
There are three different mode for concentration diffusion calculation 
Steady - Homogeneous distribution of chemoattractant - all cells are exposed to the same concentration of chemoattractant.
Error - Based on error fuction (erf) - Suitable for mimicking the diffusion of chemoattractant from one end to another. 
Linear - self-explanatory 

[Note]: 02/22/2022 by Ben 
Factoring continued

[Note]: 02/21/2022 by Ben 
Factoring started

[Note]: 07/30/2021 by Ben 
The new version should be capable of calculating the physiological responses of individual cells. 
1. Ca transients 
2. Autocrine as function of Ca 
3. Migration as function of Ca and ATP 
4. Inclusion of P2X and P2Y expression in each machinary 

[Note]: 11/06/2020 by Ben 
This simulation is designed for 2D space but it can be also implemented in 3D
'''

## This is where parameters are being stored
'''
2/21/22
Check if all this input parameters are necessary
If anything is deleted or added, it should be updated in the end of script. 
'''
def simulator(ExtATP = 10,                    
              cellConc = 1,                     
              # kinetics related 
              Diff = 10,                       
              DispScale = 200,               
              searchingRange = 0.1,
              DiffState = 'steady',             # steady, error, or linear
              # num of cells related
              Density1 = 0.02,                 
              Density2 = 0.02,
              numOfDeadCell = 50,               # This is redefined by the size of area in path
              # dimension related 
              UnitLength = 50,
              Indentation = 10,
              pathRatio = 1,
              # output related 
              pdbFileName = 'temp',             # generated PDB file name 
              dcdfilename = 'test_output2',     # generated DCD file name 
              simLength = 10000,                # a total number of the simulation steps 
              dumpSize = 100,                   # dump size in the dcd generation process               
              # simulation Mode              
              stateVar = 'on',
              intradyn = 'on',
              testMode = True,
              P2X_resting = 1,
              P2X_activated = 5,
              P2Y_resting = 1,
              P2Y_activated = 0.001,
              DCrepulsive = 0.85,
              frictionCoeff = 0.15,
              simType = 'box' # or 'path'
              ):
    

    # This will generate the random structure that makes no sence but will be processed again
    # Simulation Box Dimension in angstrom
    
    '''
    2/21/22
    [Note]
    Can we eliminate Area2 from Box calculation? 
    
    '''
    # common parameters 
    perLine = 5
    min_dist_among_cells = 2.5
    
    
    ## Box case 
    if simType == 'box':        
        '''
        - Box model - 
        More for homogeneous distribution of ATP
        _________________
        |                |
        |                |
        |       1        |
        |                |
        |                |
        |________________|

        '''
        
        boxFactor = 10
        pathFactor = 0
        pathWFactor = 0
        wallFactor = 0
        # step 1: calculate the area size of simulation box
        Area1 = (UnitLength*boxFactor)**2
        Area2 = 0
        
        # step 2: calculate a number of boundary particles that contain cells within the area of interest
        shape_factor = boxFactor 
        total_boundary = shape_factor*UnitLength*4
        num_BP = int(total_boundary/perLine)
        
        # step 3: calculate the dimension of simulation box (2D)
        xdim = [0,UnitLength*shape_factor]
        ydim = [0,UnitLength*shape_factor]
        zdim = [0,0]
        highBC = np.array([xdim[1],ydim[1],zdim[1]])
        
        # step 4: the following is not required for "slab or migration assay model"
        maxXpath = 0
        minXpath = 0
        maxYpath = 0
        minYpath = 0
        resv_dim = [maxXpath, minXpath, maxYpath, minYpath]
        
        # step 5: assigning the origin of external signal (this is a placeholder for box model)
        dummy = (np.int(UnitLength*(shape_factor+1)))/(10)
        OriginOfExternalSignal = [dummy,0,0] # in nanometer
        
        # step 6: calculating a number of particles (cells)
        if testMode:
            numOfCells1 = 10
            numOfCells2 = 0
        else:
            numOfCells1 = int(Area1*Density1)
            numOfCells2 = int(Area2*Density2)
    
    #-------------------------------------------------------------------------------------------------------------------#
    else:
        '''
        - Migration Assay-   
        _________                         ________
        |        |_______________________|        |
        |   1     _______________________    3    |
        |________|           2           |________|
        <--  l -->
        <--------            l x 6       --------->
        <--------        l x 5  --------->
                                         ^
                                         |
                                         This is the source of external ATP -> double-check with this
        Area 1: Only resting cells 
        Area 2: Activated Cells + Deadcells
        Area 3: Uniform ATP concentration
        
        2/21/22
        [Note]: Dimension for the path can be adjusted based on the experimental model.
        Once the experimental setup is determined, this can be adjusted. 
        
        '''
        # step 1: calculate the area size of simulation box 
        print('SLAB structure')
        boxFactor = 4
        pathFactor = boxFactor*4 #4
        pathWFactor = boxFactor*pathRatio #1.2
        wallFactor = boxFactor*2 #6
        shape_factor = boxFactor + pathFactor 
        Area1 = (UnitLength*boxFactor)**2
        Area2 = (UnitLength*(pathFactor+0.5))*(UnitLength*pathWFactor-2*Indentation)
        
        # step 2: calculate a number of boundary particles that contain cells within the area of interest
        total_boundary = 4*Indentation*wallFactor + pathFactor*UnitLength*2 + 6*UnitLength + UnitLength*2+Indentation*4
        num_BP = int(total_boundary/perLine)
        
        # step 3: calculate the dimension of simulation box (2D)
        xdim = [0,UnitLength*(shape_factor+boxFactor)] # |------>box1<------|---->path<----|------>box2<-------|
        ydim = [0,UnitLength]
        zdim = [0,0]
        highBC = np.array([xdim[1],ydim[1],zdim[1]])
        
        # step 4: migration path dimension 
        maxYpath = round((UnitLength*boxFactor)/2 + UnitLength*pathWFactor/2 - Indentation) # preferrably 10 
        minYpath = round((UnitLength*boxFactor)/2 - UnitLength*pathWFactor/2 + Indentation)
        maxXpath = (maxYpath - minYpath) + UnitLength*pathFactor
        minXpath = (maxYpath - minYpath) + UnitLength*boxFactor - Indentation
        resv_dim = [maxXpath, minXpath, maxYpath, minYpath]
        
        # step 5: assigning the origin of external signal 
        dummy = (np.int(UnitLength*(shape_factor+1)))/(10)
        OriginOfExternalSignal = [dummy,0,0] # in nanometer 
    
        # step 6: calculating a number of particles (cells)
        if testMode:
            numOfCells1 = 10
            numOfCells2 = 0
        else:
            numOfCells1 = int(Area1*Density1)
            numOfCells2 = int(Area2*Density2)
   
    '''
    2/22/22
    [Note]: migration factor with respect to the resting state will be randomized but the distribution will be proportional to 
    the ratio between resting and activated. 
    
    2/21/22
    [Note]: Up to this point, the script has been factored but has not been tested its feasibility, yet
    In genCellCoord3D script, there are so many input variables required. Some are only required for only "Slab". 
    
    
    [Outcomes of genCellCoord3D]
    initPosInNm: coordnates of cells in array format (num_particles by [x,y,z]) 
    marker: a list of markers for resting and activated cells (length of num_particles) 
    
    '''
    boundary_dim = [UnitLength, 
                    Indentation, 
                    boxFactor, 
                    pathFactor, 
                    pathWFactor, 
                    wallFactor]
    
    [initPosInNm, marker, MigMarker,
    totalmarker, Cell_Constitution] = calc.genCellCoord3D(Density1, # ii 
                                                           Density2, # ii
                                                           numOfDeadCell, # ii 
                                                           num_BP, # si
                                                           perLine, # ii 
                                                           min_dist_among_cells, # ii
                                                           resv_dim,
                                                           boundary_dim,
                                                           P2Y_resting, # ii
                                                           P2Y_activated, # ii
                                                           DiffState, # ii
                                                           simType # ii 
                                                           ) # change this to boolean
    
    
    DeadCellRef1 = Cell_Constitution['Dead']
    
    print(Cell_Constitution)
    print('Reservoir 1 & 2, Length of Box: ',UnitLength*2)
    print('Path Width: ',UnitLength*pathWFactor-2*Indentation)
    
    
    ## Compartment Dimension - Reference number to assign specific physical properties of each cell type
    numOfCells1 = int(Cell_Constitution['Compartment1'])
    numOfCells2 = int(Cell_Constitution['Compartment2'])
    numOfDeadCell = int(Cell_Constitution['Dead'])
    num_BP = int(Cell_Constitution['BP'])
    '''
    2/21/22
    [Note]: PDB file is required and it is being generated with random and unreasonable structure in the beginning to initiate the openMM calculation 
    '''
    calc.PDBgenNoPBC(pdbFileName,
                     numOfCells1,
                     numOfCells2,
                     numOfDeadCell,
                     num_BP,
                     DiffState) # Dead cell must be placed in here first
    
    pdb = PDBFile(pdbFileName+'.pdb')        
    
    # Configure Dumping frequency 
    dumpingstep = dumpSize
    dcdReporter = DCDReporter(dcdfilename+'.dcd', dumpingstep)
        
    
    # Configure Simulation LengthF 
    Repeat = simLength
    stepFreq = 1 

    # Parameters determining the dynamics of cells 
    MassOfCell = 50 
    MassOfDead = 1e14
    MassOfBorder = 1e14

    RepulsiveScale = 0.8 # this can determine the distance among cells themselves
    RepulsiveScaleDead = DCrepulsive #0.15 1.5 - 9A
    RepulsiveScaleBorder = 0.75

    ### Simulation parameters
    temperature = 15+50/(1+(500/ExtATP)**3) #0.0    # K   <---- set to 0 to get rid of wiggling for now
    frictionCoeff = frictionCoeff  # 1/ps
    step_size = 0.001    # ps
    
    ##########################################################
    ###              OpenMM Simulation Control             ###
    ##########################################################
    # Create an OpenMM System object
    system = System()

    # Create CustomNonbondedForce. We use it to implement repulsive potential between particles.
    # Only keep repulsive part of L-J potential
    '''
    [Note]: 02/27/2022 by Ben 
    
    Sigma = repulsive force 
    Delta = attractive force switch 
    
    '''
    nonbond = CustomNonbondedForce("(sigma/r)^12-delta*(sigma/r)^6; sigma=0.5*(sigma1+sigma2); delta=0.5*(delta1+delta2)")
    nonbond.addPerParticleParameter("sigma")
    nonbond.addPerParticleParameter("delta")
    # Here we don't use cutoff for repulsive potential, but adding cutoff might speed things up a bit as interaction
    # strength decays very fast.
    #nonbond.setNonbondedMethod(CustomNonbondedForce.NoCutoff)
    nonbond.setCutoffDistance(9)
    # Add force to the System
    system.addForce(nonbond)
    
    # FeedInForce is an OpenMM Force object that serves as the interface to feed forces into the simulation. It stores
    # additional forces that we want to introduce internally and has a method
    # .updateForceInContext(OpenMM::Context context, vector< vector<double> > in_forces). in_forces should be a vector of
    # 3d vectors - one 3d vector per particle and each 3d vector should contain X,Y,Z components of the force.
    # updateForceInContext() will copy forces from in_forces into internal variable that will keep them
    # until updateForceInContext() is called again. Every simulation step OpenMM will add forces stored in FeedInForce to
    # the simulation. You don't need to call updateForceInContext() every step if the forces didn't change - OpenMM will
    # use the forces from the last call of updateForceInContext().
    in_force = FeedInForce()
    # Add in_force to the system
    system.addForce(in_force)

    num_particles = len(pdb.getPositions())
    
    ### Make it 2D 
    energy_expression = 'k * (z^2)'
    force = openmm.CustomExternalForce(energy_expression)
    force.addGlobalParameter('k', 999)
    for particle_index in range(num_particles):
        force.addParticle(particle_index, [])
    system.addForce(force)
    
    for i in range(num_particles):
        #Either Square or Slab without Dead Cells
        if DeadCellRef1 == 0:
            #1st - live cells
            if i < numOfCells1 + numOfCells2:
                system.addParticle(MassOfCell)  # 100.0 is a particle mass
                sigma = RepulsiveScale
                delta = 0
                nonbond.addParticle([sigma,delta])
            #2nd - border particles
            else:
                
                system.addParticle(MassOfBorder)
                sigma = RepulsiveScaleBorder
                delta = 0
                nonbond.addParticle([sigma,delta])
        
        # Only Slab structure
        elif DeadCellRef1 > 0:
            
            #1st - cluster cells
            if i < DeadCellRef1:
                system.addParticle(MassOfDead)
                sigma = RepulsiveScaleDead
                delta = 1
                nonbond.addParticle([sigma,delta])

            #2nd - live cells 
            elif DeadCellRef1 <= i < numOfCells1 + numOfCells2 + DeadCellRef1:

                system.addParticle(MassOfCell)  # 100.0 is a particle mass
                sigma = RepulsiveScale
                delta = 0
                nonbond.addParticle([sigma,delta])
            
            #3rd - border cells 
            else:
                system.addParticle(MassOfBorder)
                sigma = RepulsiveScaleBorder
                delta = 0
                nonbond.addParticle([sigma,delta])
    
    # Create integrator
    integrator = BrownianIntegrator(temperature, frictionCoeff, step_size)

    # Create platform - CPU wouldn't work due to the memory issue. 
    platform = Platform.getPlatformByName('CUDA')

    # Create simulation
    simulation = Simulation(pdb.topology, system, integrator, platform)
    print('Simulation is initiated')
    print('REMARK  Using OpenMM platform %s' % simulation.context.getPlatform().getName())

    # Set initial positions for the particles
    simulation.context.setPositions(initPosInNm)

    # Simulate
    # We can add dcd or other reporters to the simulation to get desired output with
    ## Note: Ben's version 
    simulation.reporters.append(dcdReporter)

    time = 1e-14 # <----- can't start at 0; Otherwise, the calculation blows up 
    
    
    ## Set the initial stateVariable: Starting from 0 for now. 
    #odeiter = 5
    live_num_cells = numOfCells1 + numOfCells2
    stateVariable = np.ones([live_num_cells]) #<---------------------- time to do something about this 
   
    positions = simulation.context.getState(getPositions=True).getPositions()
    
    print("|------ calibration initiated -----|")
    for num_iter in range(1, 100): ### What is this? 
        positions = simulation.context.getState(getPositions=True).getPositions()
        simulation.step(stepFreq)
        state = simulation.context.getState(getEnergy=True, getForces=True)
        time += stepFreq
    
    print("|----- simulation initiated -----|")
    
    time_state = 1e-14
    
    # ODE for the intracellular dynamics <---------- this should be off
    R = np.ones(live_num_cells)
    A = np.ones(live_num_cells)*1e-14
    I = np.ones(live_num_cells)*1e-14
    D = np.ones(live_num_cells)*1e-14
    Signal1 = np.ones(live_num_cells)*1e-14
    Signal2 = np.ones(live_num_cells)*1e-14
    
    odes1 = {'R': R,
             'A': A, 
             'I': I, 
             'D': D, 
             'S1': Signal1,
             'S2': Signal2 }
    
    # ODE for the migration  
    '''
    [Note] 3/6/22 by Ben 
    This should be modified since the indirect migration or motility is simply controlled by "temperature"
    '''
    Ix = np.ones(live_num_cells)*2
    Iy = np.ones(live_num_cells)*2
    DMx = np.zeros(live_num_cells)
    DMy = np.zeros(live_num_cells)
    
    odes2 = { 'Ix': Ix,
              'Iy': Iy,
             'DMx': DMx,
             'DMy': DMy}
    
    dummy_force_x = []  
   
    for num_iter in range(1, Repeat): ### What is this? 
        positions = simulation.context.getState(getPositions=True).getPositions()
        # forces_vec will contain external forces that we want to feed into OpenMM.
        # Get current positions of the particles for concentration field/forces calculation
        ##################################################################################################################
        ###                                    Where All the Magical Things Happen                                     ###
        ##################################################################################################################
        if num_iter == 1:
            ConcByCell = np.zeros(live_num_cells) + 1e-14
        
        stateVariable = 1
        
        [fvX, fvY, fvZ, 
         ConcByCell_new, 
         odesnew1,
         odesnew2,
         ConcbyOrigin, 
         stateVariable_new, 
         xcoord, ycoord]        = calc.calcForce(positions,
                                                 live_num_cells,
                                                 DeadCellRef1,
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
                                                 minYpath)
        
        
        forces_vec                               = calc.calcForceModified(DeadCellRef1,
                                                                          num_BP,
                                                                          UnitLength,
                                                                          Indentation,
                                                                          xcoord,
                                                                          ycoord,
                                                                          fvX, 
                                                                          fvY, 
                                                                          fvZ)
        #stateVariable = stateVariable_new
        odes1 = odesnew1
        odes2 = odesnew2
        ConcByCell = ConcByCell_new
        # Need to monitor 'S1' to determine the proper kd for migration 
        #print(odes1['S1'])
        # Feed external forces into OpenMM
        in_force.updateForceInContext(simulation.context, forces_vec)
        # Advance simulation for 1 steps
        simulation.step(stepFreq)
        state = simulation.context.getState(getEnergy=True, getForces=True)
        time += stepFreq
        time_state += stepFreq
        
   
    if state == 'steady':
        print('from ',num_particles,' of cells, total of ',np.sum(ConcbyOrigin),' is being released by Cells with given conditions')

#!/usr/bin/env python
import sys
import numpy as np
##################################
#
# Revisions
#       10.08.10 inception
#
##################################
#
# Message printed when program run without arguments
#
def helpmsg():
  scriptName= sys.argv[0]
  msg="""
Purpose:

Usage:
"""
  msg+="   -numOfCells [integer number]:  a number of cells in the box \n" 
  msg+="   -lowlimBoxlen [integer number]: low boundary of simulation box in angstrom \n"
  msg+="   -highlimBoxlen [integer number]: high boundary of simulation box in angstrom \n" 
  msg+="   -t [integer number]: a total step of simulation \n" 
  msg+="   -ds [integer number]: dumpting size \n" 
  msg+="   -name [fileName]: output file name \n" 
  msg+="   -input [fileName]: inputfile name- no need to change it \n" 
  msg+="   -CentoR [float]: radius of area where no microglia present \n"
  msg+="   -ExtATP [float]: max concentration of substance from origin \n" 
  msg+="   -cellConc [float]: max concentration of substance from cell \n"
  msg+="   -Diff [float]:diffusion coefficient of substance \n"  
  msg+="   -kd [float]: \n"
  msg+="   -DiffScale [float]: diffusion rate of cells \n" 
  msg+="   -searchingRange [float]: searching range in angstrom \n" 
  msg+="   -restingRatio [float]: ratio of resting cells to activated cells \n"
  msg+="   -restMig [float]: resting cells migration rate factor \n"
  msg+="   -actMig [float]: activated cells migration rate factor \n"
  msg+="   -restAuto [float]: resting cells autocrine factor \n"
  msg+="   -actAuto [float]: activated cells autocrine factor \n"
  msg+="   -shape [string]: slap or square (default=None)\n"
  msg+="   -DiffState [string]: steady or linear or erf (default=None) \n"
  msg+="""


Notes:

"""
  return msg

#
# MAIN routine executed when launching this script from command line
#
if __name__ == "__main__":
  import sys
  msg = helpmsg()
  remap = "none"

  if len(sys.argv) < 2:
      raise RuntimeError(msg)
   
  ExtATP = 1                    
  cellConc = 0.04                     
  # kinetics related 
  Diff = 0.1                       
  DispScale = 50               
  searchingRange = 0.1
  DiffState = 'error'             # steady, error, or linear
  # num of cells related
  Density1 = 0.005                 
  Density2 = 0.005
  numOfDeadCell = 0
  DCrepulsive = 0.85
  # dimension related 
  UnitLength = 30
  Indentation = 5
  pathRatio = 1
  # output related 
  pdbFileName = 'test1'             # generated PDB file name 
  dcdfilename = 'test1'     # generated DCD file name 
  simLength = 50000                # a total number of the simulation steps 
  dumpSize = 1                   # dump size in the dcd generation process 
              
  # simulation Mode              
  stateVar = 'off'
  intradyn = 'off'
  testMode = False

  # Receptor Expression 
  P2X_resting = 1
  P2X_activated = 5
  P2Y_resting = 1
  P2Y_activated = 0.001
  frictionCoeff = 0.85
  simType = 'slab'



  for i,arg in enumerate(sys.argv):
    # calls 'runParams' with the next argument following the argument '-validation'
    if arg=="-ExtATP":
        ExtATP = np.float(sys.argv[i+1])
        
    if arg=="-cellConc":
        cellConc = np.float(sys.argv[i+1])
        
    if arg=="-UnitLength":
        UnitLength = np.int(sys.argv[i+1])
        
    if arg=="-Indentation":
        Indentation = np.int(sys.argv[i+1])
    
    if arg=="-t":
        simLength = np.int(sys.argv[i+1])
        
    if arg=="-ds":
        dumpsize = np.int(sys.argv[i+1])
        
    if arg=="-name":
        dcdfilename = sys.argv[i+1]
    
    if arg=="-input":
        pdbFileName = sys.argv[i+1]
        
    if arg=="-Density1":
        Density1 = np.float(sys.argv[i+1])
    
    if arg=="-Density2":
        Density2 = np.float(sys.argv[i+1])

    if arg=="-Diff":
        Diff = np.float(sys.argv[i+1])
      
    if arg=='-DispScale':
        DispScale = np.float(sys.argv[i+1])
    
    if arg=="-searchingRange":
        searchingRange = np.float(sys.argv[i+1])
                
    if arg=="-DiffState":
        DiffState = sys.argv[i+1]
        
    if arg=="-numOfDeadCell":
        numOfDeadCell = np.int(sys.argv[i+1])
        
    if arg=="-stateVar":
        stateVar = sys.argv[i+1]
    
    if arg=="-testMode":
        testMode = sys.argv[i+1]
        if testMode == 'True':
            testMode = True
        else:
            testMode = False
            
    if arg=='-P2Xr':
        P2X_resting = np.float(sys.argv[i+1])
        
    if arg=='-P2Yr':
        P2Y_resting = np.float(sys.argv[i+1])
        
    if arg=='-P2Xa':
        P2X_activated = np.float(sys.argv[i+1])
        
    if arg=='-P2Ya':
        P2Y_activated = np.float(sys.argv[i+1])
    
    if arg=='-DCsize':
        DCrepulsive = np.float(sys.argv[i+1])
        
    if arg=='-frictionCoeff':
        frictionCoeff = np.float(sys.argv[i+1])
        
    if arg=='-simType':
        simType = sys.argv[i+1]
    
    if arg=='-pathRatio':
        pathRatio = np.float(sys.argv[i+1])
        

    if arg=='-run':
      simulator(ExtATP = ExtATP,                    
          cellConc = cellConc,                     
          # kinetics related 
          Diff = Diff,                       
          DispScale = DispScale,               
          searchingRange = searchingRange,
          DiffState = DiffState,             # steady, error, or linear
          # num of cells related
          Density1 = Density1,                 
          Density2 = Density2,
          numOfDeadCell = numOfDeadCell,
          # dimension related 
          UnitLength = UnitLength,
          Indentation = Indentation,
          pathRatio = pathRatio,
          # output related 
          pdbFileName = pdbFileName,             # generated PDB file name 
          dcdfilename = dcdfilename,     # generated DCD file name 
          simLength = simLength,                # a total number of the simulation steps 
          dumpSize = dumpSize,                   # dump size in the dcd generation process 
              
          # simulation Mode              
          stateVar = stateVar,
          intradyn = intradyn,
          testMode = testMode,
          P2X_resting = P2X_resting,
          P2Y_resting = P2Y_resting,
          P2X_activated = P2X_activated,
          P2Y_activated = P2Y_activated,
          DCrepulsive = DCrepulsive,
          frictionCoeff = frictionCoeff,
          simType = simType
          )
