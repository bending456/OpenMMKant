import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

def dataextractorx(nameOfFile):
    rawdata = open('xcoord_'+nameOfFile,'r')
    data = {}
    for line in rawdata:
        if line.strip():
            line = line.strip("\n' '")
            line = line.split(" ")
            #data_time1.append(int(line[0])-100)
            count = 0
            for i in line:
                if count == 0:
                    data[str(int(i)-100)] = []
                    time = str(int(i)-100)
                    count += 1
                else:
                    data[time].append(i)
    rawdata.close()
    return data

def dataextractory(nameOfFile):
    rawdata = open('ycoord_'+nameOfFile,'r')
    data = {}
    for line in rawdata:
        if line.strip():
            line = line.strip("\n' '")
            line = line.split(" ")
            #data_time1.append(int(line[0])-100)
            count = 0
            for i in line:
                if count == 0:
                    data[str(int(i)-100)] = []
                    time = str(int(i)-100)
                    count += 1
                else:
                    data[time].append(i)
    rawdata.close()
    return data

def dataextractor2(nameOfFile):
    rawdata = open('rmsd_'+nameOfFile,'r')
    data = {}
    for line in rawdata:
        if line.strip():
            line = line.strip("\n' '")
            line = line.split(",")
            #data_time1.append(int(line[0])-100)
            count = 0
            for i in line:
                if count == 0:
                    data[str(int(i)-100)] = []
                    time = str(int(i)-100)
                    count += 1
                else:
                    data[time].append(i)
    rawdata.close()
    return data

def dataextractor3(nameOfFile):
    rawdata = open('msd_'+nameOfFile,'r')
    data = {}
    for line in rawdata:
        if line.strip():
            line = line.strip("\n' '")
            line = line.split(",")
            #data_time1.append(int(line[0])-100)
            count = 0
            for i in line:
                if count == 0:
                    data[str(int(i)-100)] = []
                    time = str(int(i)-100)
                    count += 1
                else:
                    data[time].append(i)
    rawdata.close()
    return data

def dataprocessor(data):
    time = []

    for t in data.keys():
        time.append(str(t))

    numAtoms = len(data['1'])

    fluxData = []
    fluxSum = []
    halfTime = []
    fs = 0
    for i,t in enumerate(time):
        flux = 0
        if i < len(time)-1:
            for n in np.arange(numAtoms):
                x0 = float(data[str(t)][n])
                x1 = float(data[str(int(t)+1)][n])
                if x0 < 110 and x1 >= 110:
                    flux += 1
                    fs += 1
                if x0 >= 110 and x1 < 110:
                    fs -= 1
                if fs == round(numAtoms/2):
                    halfTime.append(t)

        fluxData.append(flux)
        fluxSum.append(fs)
    if len(halfTime) == 0:
        halfTime = [1000]
    return fluxData, fluxSum, halfTime[-1]

def dataprocessor2(data,mid):
    time = []

    for t in data.keys():
        time.append(str(t))

    numAtoms = len(data['1'])

    fluxData = []
    fluxSum = []
    halfTime = []
    fs = 0
    for i,t in enumerate(time):
        flux = 0
        if i < len(time)-1:
            for n in np.arange(numAtoms):
                x0 = float(data[str(t)][n])
                x1 = float(data[str(int(t)+1)][n])
                if x0 < mid and x1 >= mid:
                    flux += 1
                    fs += 1
                if x0 >= mid and x1 < mid:
                    fs -= 1
                if fs == round(numAtoms/2):
                    halfTime.append(t)

        fluxData.append(flux)
        fluxSum.append(fs)
    if len(halfTime) == 0:
        halfTime = [1000]
    return fluxData, fluxSum, halfTime[-1]

def averager(series,repeat):
    datacollection = {}

    for n in series:
        datacollection[str(n)] = {}
        flux1 = []
        flux2 = []
        flux3 = []
        flux4 = []
        flux5 = []
        half = []

        for m in np.arange(repeat):
            nameOfFile = 'test'+str(n)+'-'+str(m)+'.csv'
            try:
                data = dataextractor(nameOfFile)
            except:
                pass


            [fluxData,fluxSum,halfTime] = dataprocessor(data)
            flux1.append(np.sum(fluxData[200:300])/100)
            flux2.append(np.sum(fluxData[300:400])/100)
            flux3.append(np.sum(fluxData[400:500])/100)
            flux4.append(np.sum(fluxData[500:600])/100)
            flux5.append(np.sum(fluxData[600:700])/100)
            half.append(float(halfTime))
            if m == 0:
                AccPart = np.array(fluxSum)
            else:
                AccPart = AccPart + np.asarray(fluxSum)

        flx1avg = np.average(flux1)
        flx2avg = np.average(flux2)
        flx3avg = np.average(flux3)
        flx4avg = np.average(flux4)
        flx5avg = np.average(flux5)

        flx1sem = sem(flux1)
        flx2sem = sem(flux2)
        flx3sem = sem(flux3)
        flx4sem = sem(flux4)
        flx5sem = sem(flux5)
        flxavg = [flx1avg,flx2avg,flx3avg,flx4avg,flx5avg]
        flxsem = [flx1sem,flx2sem,flx3sem,flx4sem,flx5sem]
        AccPartAvg = AccPart/repeat
        halfAvg = np.average(half)

        datacollection[str(n)]['fluxAvg'] = flxavg
        datacollection[str(n)]['fluxSem'] = flxsem
        datacollection[str(n)]['Accum'] = AccPartAvg
        datacollection[str(n)]['half'] = halfAvg

    return datacollection, len(fluxData)

def profile(fileName):

    data = dataextractor(fileName+'.csv')
    time = list(data.keys())
    numerical = {}
    for t in time:
        numerical[t]=[]
        for element in data[t]:
            numerical[t].append(float(element))

    DensityBins={}
    CountBins = {}
    box_end = 108
    path_end = 292
    interval1 = np.linspace(0,box_end,3) # 140
    Area1 = (interval1[1]-interval1[0])*140
    interval2 = np.linspace(box_end+20,path_end,10) # 40
    Area2 = (interval2[1]-interval2[0])*40
    box2 = 375 # 140
    Area3 = (box2-interval2[-1])*140
    interval = []
    for i in interval1:
        interval.append(i)
    for j in interval2:
        interval.append(j)

    interval.append(box2)

    for t in time:
        #first in the box
        count = np.zeros(len(interval))
        for x in numerical[t]:
            for i in np.arange(len(interval)):
                if i < len(interval)-1 and x >= interval[i] and x < interval[i+1]:
                    count[i] += 1

        density = []
        for i,j in enumerate(count):
            if i <= 1:
                density.append(j/Area1)
            elif i == 2:
                density.append(j/((interval[3]-interval[2])*40))
            elif i >= 3 and i < 12:
                density.append(j/Area2)
            else:
                density.append(j/Area3)

        CountBins[t] = count
        DensityBins[t] = density


    return CountBins, DensityBins, interval

def avgprofile(data):
    binNum = len(data['1'])
    avgall = []
    for num in np.arange(binNum):
        dummy = []
        for t in list(data.keys())[-500:-1]:
            dummy.append(data[t][num])

        avgall.append(np.average(dummy))

    return avgall

def avgrmsd(data):
    keys = list(data.keys())
    RMSDavg = []
    for key in keys:
        SumOfRMSD = 0
        count = 0
        for i,j in enumerate(data[key]):
            rmsd = float(j)
            if rmsd < 1000:
                SumOfRMSD += float(j)
                count += 1

        RMSDavg.append(SumOfRMSD/count)

    return RMSDavg

def transposer(data):
    time = list(data.keys())
    sortedByCell = np.ones([len(data[time[0]]),len(time)])
    for tn,t in enumerate(time):
        for dn,d in enumerate(data[t]):
            sortedByCell[dn,tn] = d

    return sortedByCell

def velocity(data):
    data_t = transposer(data)
    time = list(data.keys())
    entry=[]
    Xi = []
    Ti = []
    Vc = []

    for nt, t in enumerate(time):
        dummy = data_t[:,nt]
        for nx, x in enumerate(dummy):
            if x > 110 and nx not in entry:
                entry.append(nx)
                Xi.append(x)
                Ti.append(nt)

    for en, e in enumerate(entry):
        xf = data_t[e,len(time)-1]
        if xf > 110:
            xi = Xi[en]
            dist = xf - xi
            vel = dist/(int(time[-1])-Ti[en])
            Vc.append(vel)

    avgVel = np.average(Vc)
    semVel = sem(Vc)

    return Vc, avgVel, semVel

def velocity2(datax,datay):
    datax_t = transposer(datax)
    datay_t = transposer(datay)
    time = list(datax.keys())
    entry=[]
    Xi = []
    Ti = []
    Vc = []

    for nt, t in enumerate(time):
        dummyx = datax_t[:,nt]
        dummyy = datay_t[:,nt]
        for nx, x in enumerate(dummyx):
            if x > 110 and nx not in entry and dummyy[nx] > 30 and dummyy[nx] < 70:
                entry.append(nx)
                Xi.append(x)
                Ti.append(nt)

    for en, e in enumerate(entry):
        xf = datax_t[e,len(time)-1]
        yf = datay_t[e,len(time)-1]
        if xf > 110 and yf > 30 and yf < 70:
            xi = Xi[en]
            dist = xf - xi
            vel = dist/(int(time[-1])-Ti[en])
            Vc.append(vel)

    avgVel = np.average(Vc)
    semVel = sem(Vc)

    return Vc, avgVel, semVel

def msd(datax,datay,xlim,ylim1,ylim2):
    datax_t = transposer(datax)
    datay_t = transposer(datay)
    time = list(datax.keys())
    entry = []
    entrytime = []
    xi = []
    for nt, t in enumerate(time):
        foo_x = datax_t[:,nt]
        foo_y = datay_t[:,nt]
        for nx, x in enumerate(foo_x):
            if x > xlim and nx not in entry and foo_y[nx] > ylim1 and foo_y[nx] < ylim2:
                entry.append(nx)
                entrytime.append(nt)
                xi.append(x)

    end = len(time)

    msd = {}

    for nx, n1 in enumerate(entry):
        msd[str(n1)] = []
        t = entrytime[nx]
        while t < end-1:
            dxi = datax_t[n1,t]
            dx_new = dxi
            t += 1
            d = (dx_new - xi[nx])**2 ## <- I need to do with X0 not
            msd[str(n1)].append(d)



    return msd, xi, entrytime


