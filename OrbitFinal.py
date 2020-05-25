#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import math as m
import os
from time import time as tm
from numba import cuda, njit, vectorize
from findhubs import *
from a_star import *
from astro import *
from ConCreate import *


# In[284]:


def LoadTime(Log):
    with open(Log,'r') as log:
        contents = log.read().split('\n')
        firstline =  contents[0].split()
        lastcontents = contents[-2] if contents[-1] == '' else contents[-1]
        del contents
        data = lastcontents.split()
        StartTime = datetime.datetime.strptime(firstline[0]+' '+firstline[1],'%Y-%m-%d %H:%M:%S')
        time = datetime.datetime.strptime(data[0]+' '+data[1],'%Y-%m-%d %H:%M:%S') 
        t = int((time - StartTime).total_seconds())
    return StartTime,time,t

def AlToTH(alpha,R):
    return np.degrees(np.pi/2 -  np.arccos(np.sin(alpha)*R/6378137) - alpha )

def WriteTime(Log,time):
    with open(Log,'a') as log:
        log.write(str(time) + '\n')
        
def GetSSP(cartSat,t):
    azim = np.arctan2(cartSat[:,1],cartSat[:,0])
    elev = np.arctan2(cartSat[:,2],np.sqrt((cartSat[:,:2]**2).sum(1)))
    R0 = 6378137
    av = 7.2921158553e-5
    z0ecf = R0 * np.sin(elev)
    x0ecf = R0 * np.cos(elev) * np.cos(azim)
    y0ecf = R0 * np.cos(elev) * np.sin(azim)
    ang = av*t
    R0eci = (astro.R3(ang).dot(np.array([x0ecf, y0ecf, z0ecf]))).T
    Lons = np.degrees(np.arctan2(R0eci[:,1],R0eci[:,0]))
    Lats = np.degrees(np.arctan2(R0eci[:,2],np.sqrt((R0eci[:,:2]**2).sum(1))))
    return np.vstack((Lats,Lons)).T


@cuda.jit
def GetTRF(ssp, theta,PopCount, TR):
    pos = cuda.grid(1)
    if pos < ssp.shape[0]:
        N = 1.2 if ssp[pos,0] <85.0 else 1
        rows,columns = PopCount.shape
        step = 180/rows
        cl = ssp[pos,0]+theta[pos]/8
        dl = m.degrees(m.acos((m.cos(m.radians(theta[pos]))-m.sin(m.radians(cl))**2)/m.cos(m.radians(cl))**2))
        dl = 180 if dl!=dl else dl
        
        LonStart,LonStop = ssp[pos,1]- dl*N, ssp[pos,1]+ dl*N
        LatStart,LatStop = ssp[pos,0]+ theta[pos]*N, ssp[pos,0]- theta[pos]*N
        
        RowStart,ColumnStart = LatLonToindex(LatStart,LonStart,step)
        upEdge = 0.038148148148148146*rows   
        RowStart = int(upEdge if RowStart < upEdge else RowStart)

        RowStop, ColumnStop = LatLonToindex(LatStop,LonStop,step)
        downEdge = 0.8110185185185185*rows
        RowStop = int(downEdge if RowStop  > downEdge else RowStop)
        

        summ = 0
        cx =  m.radians(ssp[pos,0])
        cy = m.radians(ssp[pos,1])
        for row in range(RowStart,RowStop):
            for col in range(ColumnStart,ColumnStop):
                value = PopCount[row,col]
                lat,lon = indexToLatLon(row, col,step)
                wLat, wLon = m.radians(lat), m.radians(lon)
                ang = m.acos(m.sin(cx)*m.sin(wLat)+ m.cos(cx)*m.cos(wLat)*m.cos(m.fabs(cy-wLon)))
                if m.degrees(ang) <= theta[pos]:
                    summ = summ + value
        TR[pos] = summ
        
@njit
def indexToLatLon(row, col,step):
    lat = 90 - row*step - step/2
    lon = -180 + col*step + step/2
    return lat, lon

@njit
def LatLonToindex(lat, lon,step):
    lat = 90 - lat
    lon = 180 + lon
    row = lat//step
    col = lon//step
    return int(row),int(col)

def GetTraffic(ssp,Thetta,newMap,Kerns):
    TR = np.zeros(ssp.shape[0],dtype=np.float64)
    threadsperblock = Kerns
    blockspergrid = (ssp.shape[0] + (threadsperblock - 1)) // threadsperblock
    GetTRF[blockspergrid, threadsperblock](ssp, Thetta,newMap, TR)
    return TR


# In[3]:

print('Extracting parameters')
ConstParam = pd.read_excel(os.path.join('inputs', 'Constellation.xlsx'))
CP = ConstellationParameters(ConstParam['Satellites'].to_numpy(),ConstParam['Planes'].to_numpy(),
                             np.radians(ConstParam['Inclination'].to_numpy()),ConstParam['Altitude'].to_numpy(),
                             ConstParam['Raan shift'].to_numpy(),ConstParam['Shift'].to_numpy())
gs = pd.read_excel(os.path.join('inputs', 'Stations.xlsx'))
GS = gs[['LAT','LON','MaxConnections']].to_numpy()

alpha = np.radians(56)
HubsNumber = 20                                        # Number of Hub satellites to find  closest path to
timestep = datetime.timedelta(seconds=1)               # Simulation time step
SimulationTime = 1000*6

StartTime,time,t = LoadTime('TimeLogL.txt')
hour = time.hour
Map = np.load(os.path.join('maps',f'TrafGrid{time.hour}.npz'))['UTC']
Mask = np.load(os.path.join('masks',f'MaskG{time.hour}.npy'))
Map[Mask[0],Mask[1]] = 0
print('Creating Constellation')
Constellation = CreateConstellation(CP)
Params = IniConstellation(CP,Constellation,[])
cartSat = UpdateConstellation(Params,Constellation,0)
eshs = np.asarray([sat.eshelon for sat in Constellation])
Thetta = np.asarray([AlToTH(alpha,sat.r) for sat in Constellation])


SimulationTime += t - 1
while t <= SimulationTime:
    st = tm()
    print(f'Current time: {t} out of {SimulationTime}')
    if time.hour != hour:
        del Map
        Map = np.load(os.path.join('maps',f'TrafGrid{time.hour}.npz'))['UTC']
        Mask = np.load(os.path.join('masks',f'MaskG{time.hour}.npy'))
        Map[Mask[0],Mask[1]] = 0
    
    

    jd = jday (time.year, time.month, time.day, time.hour, time.minute, time.second)
    
    print('Updating constellation and connections')
    cartSat = UpdateConstellation(Params,Constellation,t)
    Connections = CreateCM(Constellation,True)
    
    hubIndexes = findHubs(GS, jd, cartSat, 25)

    print('Getting the traffic')
    SSP = GetSSP(cartSat,t)
    TRF = GetTraffic(SSP,Thetta,Map,320)

    print('Getting parents')
    Parents = GetParents(Connections,320)
    print('Finding the paths')
    Paths,lengthM,lengthNodes = GetPaths(Parents,HubsNumber,cartSat,hubIndexes,eshs)
    print(np.mean(lengthNodes),hubIndexes.size,np.where(Parents==-1)[0].size)

    WritePathes(Paths,lengthM,lengthNodes,TRF,os.path.join('outputs','PathsL', f'Paths_{t}.csv'))
    
    print(f'Parhs for time {t} has been saved, it took %.1f s' %(tm()- st))
    time += timestep
    hour = time.hour
    t = int((time - StartTime).total_seconds())
    
    WriteTime('TimeLogL.txt',time)
