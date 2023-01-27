# adapted from https://gist.github.com/rmcgibbo/6094172
# code also adapted from runner.py written by Ben Chun

"""Propagating 2D dynamics on the muller potential using OpenMM.
Currently, we just put a harmonic restraint on the z coordinate,
since OpenMM needs to work in 3D. This isn't really a big deal, except
that it affects the meaning of the temperature and kinetic energy. So
take the meaning of those numbers with a grain of salt.
"""

from simtk.unit import kelvin, picosecond, femtosecond, nanometer, dalton
# if kant installastion
import simtk.openmm as mm
# otherwise 
#import openmm as mm
from simtk.openmm import *
from simtk.openmm.app import *

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression

min_per_hour = 60  #
# TODO put into param object
cAttr      = 1.     # conc. of chemoattractant 
diamCell   = 1e-2   # [mm]
diamOccl   = 1e-1   # [mm]
lenDom     = 1e0    # [mm]


# https://demonstrations.wolfram.com/TrajectoriesOnTheMullerBrownPotentialEnergySurface/#more
class CustomForce(mm.CustomExternalForce):
    """OpenMM custom force for propagation on the Muller Potential. Also
    includes pure python evaluation of the potential energy surface so that
    you can do some plotting"""
    aa = [5e-3]       
    bba= [1]   # power for x (1 - linear) 
    bb = [5e-3]
    XX = [0] # need to adjust this to the domain size 
    YY = [0]

    def __init__(self,
        paramDict 
        ):
        
        pD = paramDict
        yPotential = pD["yPotential"]
        xPotential = pD["xPotential"]
 
        # chemoattractant gradient ; assume RHS is maximal ATP
        # c(x=len) = cAttr * ac *x ---> ac = c/lenDom 
        ac    = cAttr/lenDom
        self.aa[0] = -1 * pD["xScale"] * ac # make attractive for U = -xScale * c       
         

        # start with a harmonic restraint on the Z coordinate
        expression = '1000.0 * z^2'

        # any changes here must be made in potential() below too 
        j=0   # TODO superfluous, remove later 

        if yPotential is False:
          self.bb[j]=0.
        if xPotential is False:
          self.aa[j]=0.
             
        # add the terms for the X and Y
        fmt = dict(
                aa=self.aa[j], XX=self.XX[j], bb=self.bb[j], bba=self.bba[j], YY=self.YY[j])

        # y parabola 
        expression += '''+ {bb} * (y - {YY})^4'''.format(**fmt)
        # xExpression / gradient 
        #expression += '''+ {aa} * (x - {XX})^4'''.format(**fmt)
        # my potential for the x direction
        # xPotential = aa*(xParticle-x0)      <--- if bba=1
        expression += '''+ {aa} * (x - {XX})^{bba}  '''.format(**fmt)
  
        print(expression)
                               
        super(CustomForce, self).__init__(expression)

    @classmethod
    # must match 'init' def. above 
    # only used for plotting, so ignores z-direction 
    def potential(cls, x, y):
        "Compute the potential at a given point x,y"
        # use cls., not self. here 
        value = 0
        #for j in range(1):
        j=0
        if 1:
            # y parabola
            #if cls.yPotential:
            value += cls.bb[j] * (y - cls.YY[j])**4
            # x gradient/linear
            #value += cls.aa[j] * (x - cls.XX[j])**4
            value += cls.aa[j] * (x - cls.XX[j])**cls.bba[j]
                
        return value

    @classmethod
    #def plot(cls, ax=None, minx=-1.5, maxx=3.0, miny=0.0, maxy=2, **kwargs):
    def plot(cls, ax=None, minx=-100, maxx=100.0, miny=-100.0, maxy=100, **kwargs):
        "Plot the potential"
        grid_width = max(maxx-minx, maxy-miny) / 200.0
        ax = kwargs.pop('ax', None)
        xx, yy = np.mgrid[minx : maxx : grid_width, miny : maxy : grid_width]
        V = cls.potential(xx, yy)
        # clip off any values greater than 200, since they mess up
        # the color scheme
        if ax is None:
            ax = plt
        ax.contourf(xx, yy, V.clip(max=100), 40, **kwargs)
        ax.colorbar()



##############################################################################
# Global parameters
##############################################################################

class Params():
  def __init__(self):
    paramDict = dict()

    paramDict["nParticles"] = 100  
    paramDict["nCrowders"] = 100  
    paramDict["friction"] = ( 50 / picosecond ) # rescaling to match exptl data PKH  
    paramDict["friction"] = ( 50              ) # rescaling to match exptl data PKH  
    paramDict["timestep"] = 10.0 * femtosecond# 1e-11 s --> * 100 --> 1e-9 [ns] 
                                #    72 s --> &*100 --> 7200 [s] (2hrs)     
    paramDict["nUpdates"] = 1000  # number of cycldes 
    paramDict["xPotential"] = False
    paramDict["yPotential"] = False
    paramDict["xScale"]   = 100.   # scale for chemoattractant gradient 
    paramDict["frameRate"]=   1.  # [1 min/update]
    paramDict["crowderRad"]= 10.  # [um]
    paramDict["outName"]="test"

    # system params (can probably leave these alone in most cases
    paramDict["dim"]    = 200  # [um] dimensions of domain 
    paramDict["nInteg"] = 100  # integration step per cycle
    paramDict["mass"] = 1.0 * dalton
    paramDict["temperature"] = 750 * kelvin

    # store all values 
    self.paramDict = paramDict

# allocate instance 
params = Params()

import yaml
def runBD(
  # each particle is totally independent, propagating under the same potential
  display=False,
  yamlFile=None
  ): 
  if yamlFile is None:
    print("YAML file was not provided - using default parameters.") 
   
  else:
    # load yaml
    with open(yamlFile, 'r') as file:
      auxParams = yaml.safe_load(file)
    
    paramDict = params.paramDict
    for key in auxParams.keys():
      paramDict[key] = auxParams[key]

      if "trajOutName" in key:
          raise RuntimeError(key+" is now deprecated. Use outName instead")
      print("Adding %s="%(key) , auxParams[key])
    
  # place particles 
  # TODO: start w preequilibrated box or get cells from expt 
  nParticles = paramDict["nParticles"] 
  #dist = 10
  #startingPositions = (np.random.rand(nParticles, 3) * np.array([dist,dist,0]) + np.array([-dist/2,-dist/2.,0]))  #
  iCrowder = 0
  #startingPositions[iCrowder,:]=0.

  import lattice 
  print("CROWDER POSITION") 
  crowderPos = np.array([0,0,0.]) 

  print("NOT IMPLEMENTED FULLY")
  lattice.GenerateCrowderLattice(16,dim=20)  # generate 16 crowders


  cellCoords = lattice.GenerateRandomLattice(
          crowderPos, crowderRad=paramDict["crowderRad"], 
          dim = paramDict["dim"],           
          nParticles=(nParticles - 1) ) 
  startingPositions = np.zeros([nParticles,3])
  startingPositions[iCrowder,:] = crowderPos
  startingPositions[1:,:] = cellCoords


  startingPositions[:,2] = 0.
  ###############################################################################
  print("WARNING: taking over the first particle in order to make it a crowder; generalize later")
  
  
  system = mm.System()

  ## define outputs for coordinates
  import calculator as calc 
  trajOutPfx=paramDict["outName"]
  trajOutName = trajOutPfx+".pkl"
  pdbFileName = trajOutPfx+".pdb"
  dcdFileName = trajOutPfx+".dcd"
  # define arbitrary pdb
  calc.genPDBWrapper(pdbFileName,nParticles,startingPositions)
  # add to openmm
  pdb = PDBFile(pdbFileName)

  # Configure dcd                    
  dumpSize = 100 
  dcdReporter = DCDReporter(dcdFileName, dumpSize)



  # define external force acting on particle 
  customforce = CustomForce(paramDict)

  for i in range(nParticles):
      if i != iCrowder:
        system.addParticle(paramDict["mass"])
        customforce.addParticle(i, [])
      else:
        system.addParticle(paramDict["mass"]*1e4)
        # PKH - i don't think I need a custom force for this one 
  
  if paramDict["xPotential"] or paramDict["yPotential"]: 
    print("Adding force") 
    system.addForce(customforce) # <-- PKH should this be added earlier to keep things in z

  # define nonbond force between particles
  nonbond = CustomNonbondedForce("(sigma/r)^12-delta*(sigma/r)^6; sigma=0.5*(sigma1+sigma2); delta=0.5*(delta1+delta2)") # TODO: don't we use geometric avg for this ?
  nonbond.addPerParticleParameter("sigma")
  nonbond.addPerParticleParameter("delta")  
  nonbond.setCutoffDistance(9)
  # Add force to the System
  system.addForce(nonbond)

  # TODO: might need to integrate into loop above, when particles are added to system
  repulsiveScale = 0.1
  for i in range(nParticles):
    if i != iCrowder:
      sigma = repulsiveScale
      delta = 0  # no attraction with other particles of same type 
    else:
      sigma = paramDict["crowderRad"]
      #delta = 50
      delta = 0           
    nonbond.addParticle([sigma,delta])

  
  #integrator = mm.LangevinIntegrator(temperature, friction, timestep)
  integrator = mm.BrownianIntegrator(paramDict["temperature"], paramDict["friction"], paramDict["timestep"])
  simulation = Simulation(pdb.topology, system,integrator) 
  #context = mm.Context(system, integrator)
  
  simulation.context.setPositions(startingPositions)
  simulation.context.setVelocitiesToTemperature(paramDict["temperature"])
  #context.setPositions(startingPositions)
  #context.setVelocitiesToTemperature(paramDict["temperature"])

  # dcd writing
  simulation.reporters.append(dcdReporter)
  
  if display:
    CustomForce.plot(ax=plt.gca())
  
  nUpdates = paramDict["nUpdates"]
  totTime = nUpdates *  paramDict["frameRate"]  # [1 min/update]

  ts = np.arange(nUpdates)/float(nUpdates) * totTime             
  xs = np.reshape( np.zeros( nParticles*nUpdates ), [nParticles,nUpdates])
  ys = np.zeros_like(xs)
  
  #msds = np.zeros( nUpdates ) 
  #x0s = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)

  # minimize to reconcile bad contacts
  #simulation.minimizeEnergy() # don't do this, since it will move the crowders too 

  #
  # START ITERATOR 
  #
  for i in range(nUpdates): 
      # get positions at ith cycle 
      x = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
  
      # get particle's positions 
      xs[:,i] = x[:,0]
      ys[:,i] = x[:,1]
      #print(x[1,:])
      #print("------") 
      #print(x[0:5,:])
      
      # plot 
      if display: 
        #plt.scatter(x[:,0], x[:,1], edgecolor='none', facecolor='k')
        if i==0:
            facecolor ='r'
        else:
            facecolor = 'k'
        plt.scatter(x[0,0], x[0,1], edgecolor='none', facecolor='b')
        plt.scatter(x[1:,0], x[1:,1], edgecolor='none', facecolor=facecolor)
  
      
      
      # integrate 
      #integrator.step(paramDict["nInteg"]) # 100 fs <-->  
      simulation.step( paramDict["nInteg"] ) 
  #
  # END ITERATOR 
  #

  if display:
      plt.show() 

  # package data 

  ar = [ts,xs,ys]
  import pickle as pkl
  if trajOutName is not None:
    if "pkl" not in trajOutName:
      trajOutName+=".pkl"
    file = open(trajOutName, 'wb') 
    pkl.dump(ar,file)        
    file.close()
  
  return ts,xs, ys 

     

#!/usr/bin/env python
import sys


#
# Message printed when program run without arguments 
#
def helpmsg():
  scriptName= sys.argv[0]
  msg="""
Purpose: 
 
Usage:
"""
  msg+="  %s -validation" % (scriptName)
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


  if len(sys.argv) < 2:
      raise RuntimeError(msg)



  #fileIn= sys.argv[1]
  #if(len(sys.argv)==3):
  #  1
  #  #print "arg"

  display=False 
  yamlFile = None 
  
  # Loops over each argument in the command line 
  for i,arg in enumerate(sys.argv):
    # calls 'doit' with the next argument following the argument '-validation'
    if(arg=="-display"):
      display=True
    if(arg=="-yamlFile"):
      yamlFile= sys.argv[i+1]

    #
    # Run modes 
    # 
    if(arg=="-validation"):
      #arg1=sys.argv[i+1] 
      runBD(display=display,yamlFile=yamlFile)
      quit()

    if(arg=="-run"):
      runBD(display=display,yamlFile=yamlFile)
      quit()
  





  raise RuntimeError("Arguments not understood")




