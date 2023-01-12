import numpy as np



def GenerateLattice(
        nLattice,
        nRow,
        dim
        ):

  latticeSpace = dim/nRow
  
  latticePts = np.zeros([nLattice,3]) 
  for i in range(nLattice):
      xi = int( np.floor(i/nRow) )
      yi = int( i-xi*nRow )
      #print(xi,yi)
      latticePts[i,0:2] = [xi*latticeSpace,yi*latticeSpace]
  

  # shift s.t. 
  latticePts[:,0] -= dim/2.
  latticePts[:,1] -= dim/2.

  return latticePts

def GenerateCrowderLattice(
  nParticles,
  dim=20):

  nRow = int( np.ceil( np.sqrt( nParticles ) )  )
  nLattice=nRow**2

  latticePts = GenerateLattice(nLattice,nRow,dim)



def GenerateRandomLattice( 
  # PKH turn into nCrowder x 3 array 
  crowderPos = np.array([0,0,0]), # where crowder is located 
  crowderRad = 10.,
  nParticles = 50,
  dim = 200 # [um]     This should be passed in somewhere  
  ) : 
  """ 
  Generates a distribution of cells and a crowder on a regular lattice
  """
  nLattice = 100
  nRow     =  10 
  latticePts = GenerateLattice(nLattice,nRow,dim)
  latticeIdxs= np.arange( np.shape(latticePts)[0])

  #print(latticePts) 
  #quit()
  
  # place crowder 
  # PKH iterate over each crowder position 
  minDistSqd = (latticePts[:,0] - crowderPos[0])**2
  minDistSqd+= (latticePts[:,1] - crowderPos[1])**2
  #crowderIdx = np.argmin(minDist)
  #print(crowderIdx)
  
  # 'block' neighbors 
  # PKH find all cells that VIOLATE crowderRad
  #cellIdx = np.argwhere(minDist > crowderRad**2) 
  clashIdx = np.argwhere(minDistSqd <= crowderRad**2) 
  cellIdx = np.delete(latticeIdxs,clashIdx)
  #print(len(cellIdx))
  #print(cellIdx)
  #quit()

  # store only those indices that are not in the violaters group 

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
