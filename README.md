#
#  Note: this code has been superceded by git@github.com:huskeypm/cellmigration.git
#


###
###  Xode is no longer used 
###
Code e for running langevin particle simulations


# Installation
## Python packages via anaconda
See http://docs.openmm.org/latest/userguide/application/01_getting_started.html
```
conda install -c conda-forge openmm
```

## from CLI 
- Check out the code 
```
git clone https://github.com/bending456/OpenMMKant
```

- Revise the file config.bash to include the following environmental variables

```
export GOPATH=${HOME}/go
export PATH=/usr/local/go/bin:${PATH}:${GOPATH}:/usr/local/go:${HOME}/anaconda3/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/bchun/OpenMM/lib:/home/bchun/OpenMMpy2/lib:/home/bchun/OpenMM:/home/bchun/OpenMMpy2
export JUPYTER_PATH=/home/bchun/anaconda3/bin
export PATH=/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/bin:/home/bchun/.local/bin:/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/bchun/go:/usr/local/go:/home/bchun/anaconda3/bin:/home/bchun/.openmpi/bin:/home/bchun/.openmpi/bin
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:/usr/local/cuda-10.1/lib64::/home/bchun/OpenMM/lib:/home/bchun/OpenMMpy2/lib:/home/bchun/OpenMM:/home/bchun/OpenMMpy2:/home/bchun/.openmpi/lib/:/home/bchun/.openmpi/lib/
```

- add config.bash to your environment
```
source config.bash
```

- Test the installation 
```
python3 -c "import simtk"
```

## Execution 
- It is recommended to run brown.py from the command line via 
```
python3 tests/brown.py -validation 
```


- The program is customized using parameters that are loaded in 'yaml' format. The syntax for calling the code with FILE.yaml is
```
python3 tests/brown.py -yamlFile FILE.yaml -run
```

- An example yaml file is provided [here](https://github.com/bending456/OpenMMKant/blob/main/tests/paramSet1.yaml). In this example, the trajectory file is written to x.pkl


- Note: some of our installations are old, so you make have to import simtk.openmm instead of just openmm. If so, edit tests/brown.py accordingly


## Analysis
- Trajectory files like test.pkl can be opened and analyzed using the notebook bd_sims.ipynb in ./tests. Note that an example for computing mean square displacements (MSD) is provided therein. 
- code will also generate pdb/dcd files. These can be opened using vmd
-- VMD: load test.pdb. Right click entry in GUI. Select Load Data Into Molecule. Select dcd


## TODO
- RESOLVED There is some kind of problem with the z-constraint. Compare brown.py z coordinates relative to openmuller.py in the tests directory 
- PBCs in Y direction should be added 
- piecewise continuous functions? (this doesn't appear to be supported 


## Fitting procedure
I adjusted the nUpdates parameter to equal the number of frames taken by the microscope
The framerate parameter is set to #/min 
The distance units in the code are assumed to be [um]
The fraction parameter was adjusted s.t. the MSD at the last frame was close to the expt value

###############################################
## Other examples (Deprecated for now) 

WARNING: operator.py is now in Archive. I was getting a strange error about tuples that I nartrowed down to this script. 

Test case: 
```
$ python3 operator.py
```
or
```
$ /usr/bin/python3 operator.py
```

For the lengthy simulation, 

```
$ nohup python3 operator.py
```

Cases for final data (see notes below for additional instructions) 
```
# mutiple channel witdths 
$ python3 operatorFinal.py -pathwidths 

# open box sims 
```
## Configuration Guide for Various Cases
### Case 1: Open vs. Channel Model 
In this script, there are two ways to simulate the migration. 
1. To assess the motility (random motion) of cells, so called "Open" model is mimicking the cells placed in the large pool. To activate "open" model, the variable called "simType" needs to be set as "box". In ```operator.py```, this variable can be reassigned by the line with "Type" variable. As a default, it is "box" to simulate "open" model. If "Type" is given in "operator.py", there are only two options: "'box'" or "'non-box'". Any command other than "box" will activate the simulation with "slab", which is equivalent to "channel" structure (described in "runner.py".) 
2. Size Control: "UnitLength" represents the size of "box" or "channel" model. For instance, if you set "UnitLength = 1" then in the box model, it will create a square box with length of 10. "channel" model is a bit complicated. Due to the placement of channel between two reserviors or large pools, it takes another variable called "Indentation", which will control the width of "channel". Therefore, for the simplistic approach, it can be handy to adjust "UnitLength" but the downside is that it will result in differentiating the number of particles (cells) in the system. The best approach is to fix "UnitLength" and adjust "Indentation" with given allowance. For instance, "UnitLength = 50" then the reservior (square shape) length is "10". The channel width is determined by "Indentation", which is indicating how low from the height of reservior box and how high from the bottom of reservior box. If "Indentation = 1" then the width of channel is 8. For this reason, it is important "UnitLength" is substantially large enough to accomodate "Indentation" in the case where the simulation varies the width of channel. 
### Case 2: Different Densities
Densities are directly controlled by the variables called "Density1" and "Density2". 
1. Density1: density of cells in the reservior 1 (channel and box). This represent the number of "resting" or "unstimulated" cells. 
2. Density2: density of cells in the channel (not applicable for Box model). It is either all "resting" or "activated" 
3. numOfDeadCell: determins a number of crowder particles in the channel domain. It is also not applicable for the box model. The particle is larger than regular cells and they are placed with fixed distance from their neighboring crowder particle. 


## Analysis Guide 
### Generating CSV file for the RMSD/MSD analysis
The analysis is done by VMD script 
To run vmd without display or GUI, 
execute the following command

```
$ vmd -dispdev 
```

within the vmd environment, 
```
$ source ./analysis-path.tcl
$ rmsd start name 
```
for the series of simulations, 
```
$ rmsdseries start startNo endNo
```

- "start": to ignore the analysis on the part of "calibration", which is done be energy minimization process. 
- Check out the line 230 (minimization duration) and line 630 (dump size) in runner.py. Current value is 100 (dumpsize = 1 and step = 100). 
- "name": output name. There should be two csv files generated by this script. One is rmsd and another is msd
- "startNo": When there are series of simulations, it should be 1
- "endNo": When there are series of simulations, it should be n+1 where n is the last number of the simulation series. 

### Using pytraj 
Please, check out [pytraj example in Colab](https://colab.research.google.com/drive/139l9ci_iIkFixgfZMWVRi85ALpE9HCx6?usp=sharing)

Note: doesn't seem to like something about config.bash (permissions)



### Visulizing the data 
Please, check out Analysis.ipynb. 
As long as this script is being executed in Kant, it should properly load "experimental" data since paths are included. 
According to Xuan Fang, 

- A1, A2, B1, B2, C1, C2 are no ATP ones
- A3-6, B3-6, C3-6 are ATP ones
- A, B, C are different cell densities: 5k, 10k, 20k




