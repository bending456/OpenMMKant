import numpy as np

dim = 20 # [um]     
nLattice = 100
nRow     =  10 
latticeSpace = dim/nRow

crowderRad = 10   

def GenerateLattice( 
  crowderPos = np.array([0,0,0]), # where crowder is located 
  nParticles = 50 
  ) : 
  """ 
  Generates a distribution of cells and a crowder on a regular lattice
  """

  
  latticePts = np.zeros([nLattice,3]) 
  for i in range(nLattice):
      xi = int( np.floor(i/nRow) )
      yi = int( i-xi*nRow )
      #print(xi,yi)
      latticePts[i,0:2] = [xi*latticeSpace,yi*latticeSpace]
  
  # shift s.t. 
  latticePts[:,0] -= dim/2.
  latticePts[:,1] -= dim/2.

  #print(latticePts) 
  #quit()
  
  # place crowder 
  minDist = (latticePts[:,0] - crowderPos[0])**2
  minDist+= (latticePts[:,1] - crowderPos[1])**2
  crowderIdx = np.argmin(minDist)
  #print(crowderIdx)
  
  # 'block' neighbors 
  #crowderRad = 299
  cellIdx = np.argwhere(minDist > crowderRad**2) 
  nCellIdx = np.shape(cellIdx)[0]
  if nCellIdx < nParticles:
      raise RuntimeError("Not enough spaces to place cells") 
  
  # get random set of lattice points 
  #print(np.shape(cellIdx)) 
  randomized = np.random.choice( 
          np.ndarray.flatten( cellIdx ) , 
          size=nParticles,
          replace=False) # dont reuse lattice points 
  
  cellCoords = latticePts[randomized,:]
  #print(cellCoords) 
  #quit()
  return cellCoords

  
  #for i in cellIdx:
  #    print(latticePts[i,0:2], minDist[i])
  #print( latticePts[cellIdx,:] ) 
  #print( minDist[cellIdx])
  #print( np.shape(cellIdx) ) 

#GenerateLattice()
