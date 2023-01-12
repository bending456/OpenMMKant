import matplotlib.pylab as plt
import numpy as np 
from sklearn.linear_model import LinearRegression

def PlotStuff(
  msds,
  ts,
  outName=None
  ):     
  print("PLOTSTUFF has been renamed; use PlotMSD instead\n") 
  PlotMSD(ts,msds,outName=outName)

def PlotMSD(     
  ts,
  msds,
  outName=None
  ):     
  """
  ts - len(ts) = num frames from microscope (assuming 1/min) 
  msds - [um]  
  """
  print(np.shape(ts))
  print(ts[-1])

  # display rmsd
  display = False
  display = True
  if display:
    # exp data later 
    adjTime = 1.
    texp = ts*adjTime
    texp_=texp.reshape((-1, 1))

    model = LinearRegression().fit(texp_,msds)
    #model = LinearRegression().fit(texp,msds)
    msdfit= model.intercept_ + texp*model.coef_
    D = model.coef_ / 4.   # since msd = 4Dt for 2D diffusion 

    plt.plot(texp,msds  , label="exp")
    plt.plot(texp,msdfit, label="fit")
    plt.ylabel("MSD [um^2]")
    plt.xlabel("time [min]")
    plt.title("D= %f [um^2/min]"%D) 
    plt.legend(loc=0)
 
    # 1K ts --> 2hrs/7200s?  
    # 200 [um**2 ] 
    #plt.show()
    #plt.gca().savefig("compare.png")
    if outName is not None:
      plt.savefig(outName)       

    return D 

    


import pickle as pkl

def LoadPKLData(trajOutName="test.pkl"):
    file = open(trajOutName,"rb")
    data = pkl.load(file)
    file.close()

    [ts,xs,ys] = data
    nUpdates = np.shape(ts)[0]
    nParticles = np.shape(xs)[0]

    return ts,xs,ys,nUpdates,nParticles
