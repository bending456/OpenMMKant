# adapted from https://gist.github.com/rmcgibbo/6094172

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

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# TODO: adj. shape of the potential 
# https://demonstrations.wolfram.com/TrajectoriesOnTheMullerBrownPotentialEnergySurface/#more
class CustomForce(mm.CustomExternalForce):
    """OpenMM custom force for propagation on the Muller Potential. Also
    includes pure python evaluation of the potential energy surface so that
    you can do some plotting"""
    aa = [10]
    bb = [100]
    XX = [-10] 
    YY = [1]

    def __init__(self):
        # start with a harmonic restraint on the Z coordinate
        expression = '1000.0 * z^2'

        for j in range(1):
            # add the muller terms for the X and Y
            fmt = dict(aa=self.aa[j], XX=self.XX[j], bb=self.bb[j], YY=self.YY[j])
            # y parabola 
            expression += '''+ {bb} * (y - {YY})^4'''.format(**fmt)
            # x gradient 
            #expression += '''+ {aa} * x + {XX}'''.format(**fmt)
                               
        super(CustomForce, self).__init__(expression)

    @classmethod
    # must match 'init' def. above 
    def potential(cls, x, y):
        "Compute the potential at a given point x,y"
        value = 0
        for j in range(1):
            # y parabola
            value += cls.bb[j] * (y - cls.YY[j])**4
            # x gradient
            #value += cls.aa[j] * x + cls.XX[j]
                
        return value

    @classmethod
    def plot(cls, ax=None, minx=-1.5, maxx=3.0, miny=0.0, maxy=2, **kwargs):
        "Plot the potential"
        grid_width = max(maxx-minx, maxy-miny) / 200.0
        ax = kwargs.pop('ax', None)
        xx, yy = np.mgrid[minx : maxx : grid_width, miny : maxy : grid_width]
        V = cls.potential(xx, yy)
        # clip off any values greater than 200, since they mess up
        # the color scheme
        if ax is None:
            ax = plt
        ax.contourf(xx, yy, V.clip(max=200), 40, **kwargs)

class MullerForce(mm.CustomExternalForce):
    """OpenMM custom force for propagation on the Muller Potential. Also
    includes pure python evaluation of the potential energy surface so that
    you can do some plotting"""
    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    AA = [-200, -100, -170, 15]
    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]

    def __init__(self):
        # start with a harmonic restraint on the Z coordinate
        expression = '1000.0 * z^2'

        # TODO - do not add x, y terms for expression 
        for j in range(4):
            # add the muller terms for the X and Y
            fmt = dict(aa=self.aa[j], bb=self.bb[j], cc=self.cc[j], AA=self.AA[j], XX=self.XX[j], YY=self.YY[j])
            expression += '''+ {AA}*exp({aa} *(x - {XX})^2 + {bb} * (x - {XX})
                               * (y - {YY}) + {cc} * (y - {YY})^2)'''.format(**fmt)
        super(MullerForce, self).__init__(expression)

    @classmethod
    def potential(cls, x, y):
        "Compute the potential at a given point x,y"
        value = 0
        for j in range(4):
            value += cls.AA[j] * np.exp(cls.aa[j] * (x - cls.XX[j])**2 + \
                cls.bb[j] * (x - cls.XX[j]) * (y - cls.YY[j]) + cls.cc[j] * (y - cls.YY[j])**2)
        return value

    @classmethod
    def plot(cls, ax=None, minx=-1.5, maxx=1.2, miny=-0.2, maxy=2, **kwargs):
        "Plot the Muller potential"
        grid_width = max(maxx-minx, maxy-miny) / 200.0
        ax = kwargs.pop('ax', None)
        xx, yy = np.mgrid[minx : maxx : grid_width, miny : maxy : grid_width]
        V = cls.potential(xx, yy)
        # clip off any values greater than 200, since they mess up
        # the color scheme
        if ax is None:
            ax = plt
        ax.contourf(xx, yy, V.clip(max=200), 40, **kwargs)


##############################################################################
# Global parameters
##############################################################################

class Params():
  def __init__(self):
    self.friction = ( 100 / picosecond ) / 0.0765 # rescaling to match exptl data PKH  

params = Params()

def runBD(
  # each particle is totally independent, propagating under the same potential
  nParticles = 100, 
  mass = 1.0 * dalton,
  temperature = 750 * kelvin,
  friction = 100 / picosecond,   # ps= 1000 fs --> 
  timestep = 10.0 * femtosecond,# 1e-11 s --> * 100 --> 1e-9 [ns] 
                                #    72 s --> &*100 --> 7200 [s] (2hrs)     
  nInteg = 100,  # integration step per cycle
  nUpdates = 1000,  # number of cycldes 
  addForce=False,
  saveTraj=None,
  trajOutName=None,
  display=False
  ): 
  if saveTraj is not None:
    raise RuntimeError("saveTraj is antiquated; use trajOutName instead") 

  friction = params.friction  

  # Choose starting conformations uniform on the grid between (-1.5, -0.2) and (1.2, 2)
  # domain unuts [nm?] --> [mm] (1e9) 
  # TODO: define size of domain 
  # 1600 um  x 1600 um <--> 1.6 mm 
  # for mueller potential 
  #startingPositions = (np.random.rand(nParticles, 3) * np.array([2.7, 1.8, 1])) + np.array([-1.5, -0.2, 0])
  startingPositions = (np.random.rand(nParticles, 3) * np.array([0.25,1,1]) + np.array([1,0,0]))  #
  #)startingPositions[:,1] = 2.
  startingPositions[:,2] = 0.
  ###############################################################################
  
  
  system = mm.System()


  # define force acting on particle 
  customforce = CustomForce()
  for i in range(nParticles):
      system.addParticle(mass)
      customforce.addParticle(i, [])
  
  if addForce:
    print("Adding force") 
    system.addForce(customforce)

  
  #integrator = mm.LangevinIntegrator(temperature, friction, timestep)
  integrator = mm.BrownianIntegrator(temperature, friction, timestep)
  context = mm.Context(system, integrator)
  
  context.setPositions(startingPositions)
  context.setVelocitiesToTemperature(temperature)
  
  if display:
    CustomForce.plot(ax=plt.gca())
  
  
  min_per_hour = 60  #
  ts = np.arange(nUpdates)/float(nUpdates) * 2*min_per_hour 
  xs = np.reshape( np.zeros( nParticles*nUpdates ), [nParticles,nUpdates])
  ys = np.zeros_like(xs)
  
  msds = np.zeros( nUpdates ) 
  x0s = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
  for i in range(nUpdates): 
      # get positions at ith cycle 
      x = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
  
      # get particle's positions 
      xs[:,i] = x[:,0]
      ys[:,i] = x[:,1]
      
      # get msd for each particle 
      # TODO remove this 
      #xmsd = x[:,0] - x0s[:,0]
      #ymsd = x[:,1] - x0s[:,1]
      #sd = xmsd*xmsd + ymsd*ymsd
      #msd = np.mean(sd) 
      #msds[i] = msd 
  
  
      # plot 
      if display: 
        plt.scatter(x[:,0], x[:,1], edgecolor='none', facecolor='k')
  
      
      
      # integrate 
      integrator.step(nInteg) # 100 fs <-->  

  if display:
      plt.show()

  # package data 
  ar = [ts,xs,ys]
  import pickle as pkl
  if trajOutName is not None:
    if "csv" not in trajOutName:
      trajOutName+=".pkl"
    file = open(trajOutName, 'wb') 
    pkl.dump(ar,file)        
    file.close()
  
  return ts,xs, ys 

     
# pull more of this into notebook 
# PKH REMOVE THIS FUNCTION? 
def PlotStuff(
    
  ):     
  raise RuntimeError("depprecated") 
  # should move these
  #data = np.loadtxt("/Users/huskeypm/Downloads/trajectories.csv",delimiter=",")
  data = np.loadtxt("trajectories.csv",delimiter=",")

  # show trajectory 
  display = False   
  #display = True    
  if display: 
    plt.plot(xPos[:,0], xPos[:,1])
    plt.show()
  
  
  # display rmsd
  display = False 
  display = True    
  if display:
    plt.plot(ts/min_per_hour,msds,label="Sim")
    plt.xlabel("t [hr]") 
    plt.ylabel("MSD [um**2]") 

    adjTime = 2/50. # not sure why 50 entries, assuming 2 hrs 
    texp = data[:,0]*adjTime
    msdexp=data[:,1]
    texp_=texp.reshape((-1, 1))
    model = LinearRegression().fit(texp_,msdexp)
    msdfit= model.intercept_ + texp*model.coef_
    plt.plot(texp,msdexp, label="exp")
    plt.plot(texp,msdfit, label="fit")
    plt.legend(loc=0)
  
    # 1K ts --> 2hrs/7200s?  
    # 200 [um**2 ] 
    #plt.show()
    plt.gca().savefig("compare.png") 


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
  remap = "none"

  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  #fileIn= sys.argv[1]
  #if(len(sys.argv)==3):
  #  1
  #  #print "arg"

  friction = 1/picosecond
  trajOutName = None 
  display=False 
  addForce=False 
  # Loops over each argument in the command line 
  for i,arg in enumerate(sys.argv):
    # calls 'doit' with the next argument following the argument '-validation'
    if(arg=="-display"):
      display=True
    if(arg=="-addForce"):
      addForce=True
    if(arg=="-trajOutName"):
      trajOutName = sys.argv[i+1]

    if(arg=="-validation"):
      #arg1=sys.argv[i+1] 
      runBD(friction=friction,display=display,addForce=addForce,trajOutName=trajOutName)
      quit()
    if(arg=="-test"):
      runBD(friction=friction,display=display,addForce=addForce,trajOutName=trajOutName, nParticles = 10)
      quit()
  





  raise RuntimeError("Arguments not understood")




