#!/usr/bin/env python
# coding: utf-8

from numba import cuda, njit, vectorize
import numpy as np
import pandas as pd




def CreateEshelon(CP,e,esh):
    SpP,NoP,h = CP.sats[esh],CP.planes[esh],CP.alts[esh]
    INC,gshift,raanshift = CP.incs[esh],CP.shifts[esh],CP.raanshifts[esh]
    EarthRadius = 6378137
    shift = 2*np.pi/(NoP*SpP)*gshift
    RAANS = np.linspace(0,2*np.pi,NoP,endpoint=False) + np.radians(raanshift)
    AOPS = np.linspace(0,2*np.pi,SpP,endpoint=False)
    Eshelon = []
    for RaanNumber, RAAN in enumerate(RAANS):
        for AopNumber, AOP in enumerate(AOPS):
            globalid = int(np.sum((CP.sats*CP.planes)[:esh]) + SpP*RaanNumber + AopNumber)
            Eshelon.append(Satellite(h,0,AOP+shift*RaanNumber,INC,RAAN,e,RaanNumber,AopNumber,esh,globalid,CP))
    return Eshelon

@njit(parallel = True)  # 7 µs
def OPtoCC(Params):
    #[0,sat.aop,sat.inc,sat.raan,sat.e,sat.a]
    r = Params[:,5]*(1-Params[:,4]**2)/(1+Params[:,4]*np.cos(Params[:,0]))
    x = r*np.cos(Params[:,0])
    y = r*np.sin(Params[:,0])
    sraan,craan = np.sin(Params[:,3]),np.cos(Params[:,3])
    saop,caop = np.sin(Params[:,1]),np.cos(Params[:,1])
    cinc,sinc = np.cos(Params[:,2]),np.sin(Params[:,2])
    X = craan*(x*caop + y*saop) + sraan*cinc*(y*caop - x*saop)
    Y = craan*cinc*(y*caop - x*saop) - sraan*(x*caop + y*saop)
    Z = sinc*(y*caop - x*saop)
    return X,Y,Z

def UpdateConstellation(Params,Constellation,t):
    #[0,sat.aop,sat.inc,sat.raan,sat.e,sat.a]
    tempParams = Params.copy()
    mu = 3.9860044188e14
    J2 = 1.082626e-3
    Re = 6378137
    meanMotion = np.sqrt(mu/(Params[:,5]**3))
    ta = (meanMotion)*t
    aV = -3/2*meanMotion*(Re/Params[:,5])**2*J2*np.cos(Params[:,2]) 
    tempParams[:,0] = ta
    tempParams[:,3] += aV*t  
    x,y,z = OPtoCC(tempParams)
    cartSat = np.asarray([x,y,z]).T
    for sat in Constellation:
        sat.x = x[sat.globalid]
        sat.y = y[sat.globalid]
        sat.z = z[sat.globalid]
# #             sat.ta = ta[sat.globalid]
    return cartSat


def CreateConstellation(CP):
    Constellation = [CreateEshelon(CP,0,n) for n in range(len(CP.alts))]
    Const = np.asarray([sat for Eshelon in Constellation for sat in Eshelon])
    return Const

def GetEshelon(n,Constellation,CP):
    if n == 0:
        Sats = slice(0,np.sum(CP.planes[n]*CP.sats[n]))
    else:
        Sats = slice(np.sum(CP.planes[:n]*CP.sats[:n]),np.sum(CP.planes[:n+1]*CP.sats[:n+1]))
    return Constellation [Sats]


@njit
def GetClosestHubs(satC,satEsh,HubC,eshs,hubIndexes,N,section):
    x,y,z = satC[0],satC[1],satC[2]
    eshInd = eshs[hubIndexes]
    inEshC = HubC[eshInd==satEsh]
    step = np.radians(5)
    section = np.radians(section)
    Azumut = np.arctan2(HubC[:,1],HubC[:,0])
    Zenith = np.arctan2(np.sqrt((HubC**2).sum(1)),HubC[:,2])

    SatZenith = np.arctan2(np.sqrt(x**2+y**2),z)
    SatAzumut = np.arctan2(y,x)
    closest = np.argsort(np.sqrt(((HubC - satC)**2).sum(1)))[:N-1]
    closestInEsh = np.argsort(np.sqrt((inEshC - satC)**2).sum(1))[0]
    finals = np.array([closestInEsh])
    while len(finals)<N:
        arr1 = np.where(np.abs(Zenith - SatZenith)<section)[0]
        arr2 = np.where(np.abs(Azumut - SatAzumut)<section)[0]
        finals = np.append(finals,np.array( list(set(arr1)&set(arr2)&set(closest))))
        section += step
    
    ClosestHubs = hubIndexes[finals]
    return ClosestHubs


@cuda.jit
def findParents(graph,vertices,visited,dist,parents):
#     pos = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    pos = cuda.grid(1)
    if pos < vertices:  # Check array boundaries
        dist[pos,pos] = 0
        for n in range(vertices):
            
            minDist = np.inf
            for vertex in range(vertices): 
                if dist[pos,vertex] < minDist and visited[pos,vertex] == False: 
                    minDist = dist[pos,vertex]
                    origin = vertex
                    
            visited[pos,origin] = True
            
            for target in range(vertices): 
                if graph[origin,target] > 0 and visited[pos,target] == False:
                    if dist[pos,origin] + graph[origin,target] < dist[pos,target]: 
                        dist[pos,target] = dist[pos,origin] + graph[origin,target]
                        parents[pos,target] = origin
# @njit                       
def findPath(pars,origin,target):
    if origin == target:
        return np.array([origin])
    else:
        pt = [target]
        while pars[origin,target]!=pars[origin,origin]:
            target = pars[origin,target]
            pt.append(target)
        if len(pt)==1:
            return np.array([-101])
        else:
            return np.asarray(pt)

# @njit
def findPathToClosest(pars,origin,hubs):
    truehubs = []
    for hub in hubs:
        path = findPath(pars,origin,hub)
        if path[0]!=-101:
            shortestPath = path
            truehubs.append(hub)
    if truehubs:
        for hub in truehubs:
            Npath = findPath(pars,origin,hub)
            if Npath.size<path.size:
                shortestPath = Npath
        return shortestPath         
    else:
        return np.array([-101])


def IniConstellation(CP,Constellation,NoInterPlaneLinks):
    Params = []
    esh = Constellation[0].eshelon
    Eshelon = GetEshelon(esh,Constellation,CP)
    ids = np.asarray([sat.id for sat in Eshelon])
    for sat in Constellation:
        CurrentEshelon = sat.eshelon
        if esh != CurrentEshelon:
            Eshelon = GetEshelon(CurrentEshelon,Constellation,CP)
            ids = np.asarray([sat.id for sat in Eshelon])
        InterPlane = False if CurrentEshelon in NoInterPlaneLinks else True
        sat.links.CreateLinks(Eshelon,ids,InterPlane)
        Params.append([0,sat.aop,sat.inc,sat.raan,sat.e,sat.a])
        esh = sat.eshelon
    Params = np.asarray(Params)
    return Params


def CreateCM(Constellation,tp):
    Consize = Constellation.size
    if tp == False:
        Connections = np.empty((Consize,Consize), dtype = np.uint8)
        for sat in Constellation:
            master = sat.globalid
            for slave in sat.links.list:
                Connections[master,slave.globalid] = 1
                Connections[slave.globalid,master] = 1
    if tp == True:
        Connections = np.empty((Consize,Consize), dtype = np.uint32)
        for sat in Constellation:
            master = sat.globalid
            for slave in sat.links.list:
                Connections[master,slave.globalid] = sat.DistanceTo(slave)
                Connections[slave.globalid,master] = sat.DistanceTo(slave)
    return Connections

def FindLinks(Constellation, cartSat, From, To,t,view,pt):
    for sat in Constellation[From]:
        sat.UpdateLinks(t,Constellation[To],cartSat[To],view,pt)

        
def GetParents(Connections,N):
    Conn = np.ascontiguousarray(Connections)

    vertices  = Conn.shape[0]
    parents = np.ones((vertices,vertices),dtype=np.int16)*(-1)
    dist = np.ones((vertices,vertices))*np.inf
    visited = np.zeros((vertices,vertices),dtype=bool)

    threadsperblock = N
    blockspergrid = (Conn.shape[0] + (threadsperblock - 1)) // threadsperblock
#     print(blockspergrid, threadsperblock)
    findParents[blockspergrid, threadsperblock](Conn,vertices,visited,dist,parents)
    
    return parents

@njit
def pathLength(cartSat,path):
    sats = cartSat[path]
    From = sats[1:]
    To = sats[:-1]
    dist = np.sum(np.sqrt(((From - To)**2).sum(1)))
    return dist, len(path)

# @njit
def GetPaths(parents,N,cartSat,hubIndexes,eshs):
    pathes = []
    lengthNodes = []
    lengthM = []
    HubC = cartSat[hubIndexes]
    for sat in range(parents.shape[0]):
        satC = cartSat[sat]
        satEsh = eshs[sat]
        ClosestHubs = GetClosestHubs(satC,satEsh, HubC,eshs,hubIndexes,N,20)
        path = findPathToClosest(parents,sat,ClosestHubs)
        Nodes, pathlen = pathLength(cartSat,path)
        
        pathes.append(path)
        lengthNodes.append(Nodes)
        lengthM.append(pathlen)
    return pathes,lengthNodes,lengthM

def WritePathes(pathes,lengthM,lengthNodes,TRF,path):
    result = pd.DataFrame(columns={'Origin','Hub','Path','Length [nodes]','Length [m]','Traffic'})
    origins = [n[-1] for n in pathes]
    hubs = [n[0] for n in pathes]

    result['Origin'] = origins
    result['Hub'] = hubs
    result['Path'] = pathes
    result['Length [nodes]'] = lengthNodes
    result['Length [m]'] = lengthM
    result['Traffic'] = TRF

    result.to_csv(path,index=False)

    
    
def WriteState(file,Constellation):
    with open(file,'w') as log:
        for sat in Constellation:
            pt = int(sat.links.up.pointing)
            state = int(sat.links.up.state)
            ptstart = int(sat.links.up.pointingstart)
            linkUp = sat.links.up.to.globalid if sat.links.up.to else -1
            line = str(pt)+ ' ' +str(state)+ ' ' +str(ptstart)+ ' ' +str(linkUp)
            log.write(line + '\n')
            
def LoadState(file,Constellation):
    loaded = np.loadtxt(file)
    pt = loaded[:,0].astype('bool')
    state = loaded[:,1].astype('bool')
    ptstart = loaded[:,2].astype('int16')
    linkUp = loaded[:,3].astype('int16')
    for n,sat in enumerate(Constellation):
        sat.links.up.pointing = pt[n]
        sat.links.up.state = state[n]
        sat.links.up.pointingstart = ptstart[n]
        sat.links.up.to = None if linkUp[n]==-1 else Constellation[linkUp[n]]
    return Constellation

class ConstellationParameters():
    def __init__(self,sats,planes,incs,alts,raanshifts,shifts):
        self.sats = sats
        self.planes = planes
        self.incs = incs
        self.alts = alts
        self.shifts = shifts
        self.raanshifts = raanshifts
        

class ForwardLink():
    def __init__(self,sat):
        self.id = self.forward(sat)
        self.to = None 
        self.data = 0
    def forward(self,sat):
        desnumber = sat.number + 1 if sat.number < sat.cp.sats[sat.eshelon]-1 else 0
        return str(sat.eshelon)+'.'+str(sat.orbit)+'.'+str(desnumber)

class BackwardLink():
    def __init__(self,sat):
        self.id = self.backward(sat)
        self.to = None 
        self.data = 0
    def backward(self,sat):
        desnumber = sat.number - 1 if sat.number > 0 else sat.cp.sats[sat.eshelon]-1
        return str(sat.eshelon)+'.'+str(sat.orbit)+'.'+str(desnumber)

    
class LeftLink():
    def __init__(self,sat):
        self.id = self.left(sat)
        self.to = None 
        self.data = 0
    def left(self,sat):
        if sat.orbit == 0:
            satnumber = sat.number - sat.cp.shifts[sat.eshelon]
            if satnumber<0:
                satnumber = sat.cp.sats[sat.eshelon] + satnumber
        else:
            satnumber = sat.number
        desnumber = sat.orbit - 1 if sat.orbit > 0 else sat.cp.planes[sat.eshelon]-1
        return str(sat.eshelon)+'.'+str(desnumber)+'.'+str(satnumber)

    
class RightLink():
    def __init__(self,sat):
        self.id = self.right(sat)
        self.to = None 
        self.data = 0
    def right(self,sat):
        if sat.orbit == sat.cp.planes[sat.eshelon]-1:
            satnumber = sat.number + sat.cp.shifts[sat.eshelon]
            if satnumber > sat.cp.sats[sat.eshelon]-1:
                satnumber = satnumber - sat.cp.sats[sat.eshelon] 
        else:
            satnumber = sat.number
        desnumber = sat.orbit + 1 if sat.orbit < sat.cp.planes[sat.eshelon]-1 else 0
        return str(sat.eshelon)+'.'+str(desnumber)+'.'+str(satnumber)
    
    
class UpLink():
    def __init__(self):
        self.to = None
        self.state = False
        self.pointing = False
        self.pointingstart = 0
        self.linkstart = []
        self.linkend = []
        self.angle = None

        
class Links():
    def __init__(self,sat):
        self.dists = []
        self.uploaded = 0
        self.forward = ForwardLink(sat)
        self.backward = BackwardLink(sat)
        self.left = LeftLink(sat)
        self.right = RightLink(sat)
        self.up = UpLink()
        self.busy = False #specially for upper eshelon satellites
        
    def CreateLinks(self,Eshelon,ids,interplane = True):
        self.forward.to = Eshelon[np.where(ids==self.forward.id)[0][0]]
        self.backward.to = Eshelon[np.where(ids==self.backward.id)[0][0]]
        if interplane == True:
#             self.interplane = True
            self.left.to = Eshelon[np.where(ids==self.left.id)[0][0]]
            self.right.to = Eshelon[np.where(ids==self.right.id)[0][0]]
        self.ListLinks()
        
    def ListLinks(self):
        self.list = [self.forward.to,self.backward.to,self.left.to,self.right.to]
        if self.up.to != None and self.up.state == True:
            self.list.append(self.up.to)
        self.list = [i for i in self.list if i]
        
    
        
               
class Satellite():
    def __init__(self, h, ta, AOP, INC, RAAN, e, orbit,number,eshelon,globalid,CP):
        self.cp = CP
        self.mu = 3.9860044188e14
        self.r = h + 6378137
        self.alt = h
        self.aop = AOP
        self.inc = INC        
        self.raan = RAAN        
        self.e = e
        self.a = self.r/(1-self.e)
        self.ta = ta
        self.eshelon = eshelon
        self.orbit = orbit
        self.number = number
        self.id = str(self.eshelon)+'.'+str(self.orbit)+'.'+str(self.number)
        self.globalid = globalid
        self.mam = np.sqrt(self.mu/self.a**3)
        self.links = Links(self)
        
    def UpdatePosition(self,t):
        TrueAnomaly = ToTrue(self.mam*t, self.e)
        self.x,self.y,self.z = OPtoCCone(TrueAnomaly,self.aop,self.inc,self.raan,self.e,self.a)
        self.ta = TrueAnomaly
        
    def CheckAngle(self,s2):
        tv = np.array([s2.x - self.x,s2.y - self.y,s2.z - self.z])
        angle = np.arccos((self.x*tv[0]+self.y*tv[1]+self.z*tv[2])/
                          (np.sqrt(self.x**2+self.y**2+self.z**2)*np.sqrt(tv[0]**2+tv[1]**2+tv[2]**2)))
        
        return angle
    
    def FindLinkUp(self,t,Constellation,cartSat,view):
        thisOne = np.array([self.x,self.y,self.z])
        tv = cartSat - thisOne
        angle = np.arccos(((thisOne*tv).sum(1))/
                          (np.sqrt((thisOne**2).sum())*np.sqrt((tv**2).sum(1))))
        angMask = np.where(angle < view)[0]
        candidats = Constellation[angMask]
        rd = np.asarray([sat.links.busy for sat in candidats]) #14.3 µs
        rdMask = np.where(rd == False)[0]
        if rdMask.size !=0:
            distances = np.sqrt(((cartSat[angMask] - thisOne)**2).sum(1))[rdMask]  #77.8  \\11.8 µs
            candidats = candidats[rdMask]
            target =  candidats[np.argmin(distances)]
            self.links.up.to = target         
            self.links.up.to.links.busy = True
            self.links.up.pointing = True
            self.links.up.pointingstart = t
            
            
    
    def DistanceTo(self,sat):
        return np.sqrt((sat.x-self.x)**2 +(sat.y-self.y)**2 + (sat.z-self.z)**2)
        
        
    def UpdateLinks(self,t,Constellation,cartSat,view,pt):
        if self.links.up.state == False and self.links.up.pointing == False:
            self.FindLinkUp(t,Constellation,cartSat,view)
        
        if self.links.up.state == True and self.CheckAngle(self.links.up.to) > view:
            self.links.up.state = False
            self.links.up.to.links.busy = False
            self.links.up.to = None
            self.links.up.linkend.append(t)
            
        if self.links.up.pointing == True:
            if self.CheckAngle(self.links.up.to) > view:
                self.links.up.to.links.busy = False
                self.links.up.pointing = False
                self.links.up.to = None
                self.FindLinkUp(t,Constellation,cartSat,view)
            elif (t - self.links.up.pointingstart)>= pt:
                self.links.up.state = True
                self.links.up.pointing = False
                self.links.up.linkstart.append(t)
        
        self.links.ListLinks()
        