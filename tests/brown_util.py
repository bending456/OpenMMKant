import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression

def PlotStuff(
  msds,
  ts
  ):     

  min_per_hour = 1
  # display rmsd
  display = False
  display = True
  if display:
    # sim data 
    #plt.plot(ts/min_per_hour,msds,label="Sim")
    #plt.xlabel("t [hr]")
    #plt.ylabel("MSD [um**2]")

    # exp data later 
    adjTime = 2/50. # not sure why 50 entries, assuming 2 hrs 
    #texp = data[:,0]*adjTime
    #msdexp=data[:,1]
    texp = ts*adjTime
    texp_=texp.reshape((-1, 1))

    model = LinearRegression().fit(texp_,msds)
    #model = LinearRegression().fit(texp,msds)
    msdfit= model.intercept_ + texp*model.coef_
    plt.plot(texp,msds  , label="exp")
    plt.plot(texp,msdfit, label="fit")
    plt.legend(loc=0)
 
    # 1K ts --> 2hrs/7200s?  
    # 200 [um**2 ] 
    #plt.show()
    #plt.gca().savefig("compare.png")
    plt.savefig("compare.png")

