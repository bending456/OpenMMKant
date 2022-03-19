# OpenMMKant 
## Initialization  
Add the following path to your .bashrc

```
export GOPATH=${HOME}/go
export PATH=/usr/local/go/bin:${PATH}:${GOPATH}:/usr/local/go:${HOME}/anaconda3/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/bchun/OpenMM/lib:/home/bchun/OpenMMpy2/lib:/home/bchun/OpenMM:/home/bchun/OpenMMpy2
export JUPYTER_PATH=/home/bchun/anaconda3/bin
export PATH=/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/bin:/home/bchun/.local/bin:/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/bchun/go:/usr/local/go:/home/bchun/anaconda3/bin:/home/bchun/.openmpi/bin:/home/bchun/.openmpi/bin
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:/usr/local/cuda-10.1/lib64::/home/bchun/OpenMM/lib:/home/bchun/OpenMMpy2/lib:/home/bchun/OpenMM:/home/bchun/OpenMMpy2:/home/bchun/.openmpi/lib/:/home/bchun/.openmpi/lib/
```

or source config.bash

## Execution 
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




