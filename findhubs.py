# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 13:32:36 2020

@author: Alexander.Kharlan
"""

import numpy as np
import numpy.matlib
import astro
deg = np.pi/180
R0 = 6378137

latlonGS = np.array ([ [64.83, -147.75, 20], # Fairbanks, Alaska
    [21.321605, -157.861396, 15], # Honolulu, Hawaii
    [33.933216, -118.281481, 19], # Los Angeles, California
    [43.017832, -89.385132, 25],  # Madison, Wisconsin
    [25.812291, -80.355975, 35],  # Miami, Florida
    [18.399474, -66.049074, 14],  # San Juan, Puerto Rico
    [9.859500, 8.887399, 18],     # Jos, Nigeria
    [48.388457, -4.510587, 25],   # Brest, France
    [41.634157, -4.738647, 25],   # Valladolid, Spain
    [46.034898, 14.538621, 25],   # Ljubljana, Slovenija
    [45.498342, 13.747929, 25],   # Pomjan, Slovenija
    [41.124640, 14.774235, 25],   # Benevento, Italy
    [55.749287, 37.624671, 31],   # Moscow, Russia
    [53.210687, 50.231718, 32],   # Samara, Russia
    [58.005860, 56.259677, 15],   # Perm, Russia
    [56.017824, 92.961860, 15],   # Krasnoyarsk, Russia
    [52.301306, 104.285353, 38],  # Irkutsk, Russia
    [59.557912, 150.815451, 21],  # Magadan, Russia
    [30.780302, 120.776480, 15],  # Jiaxing, China
    [0. , 0., 10]  # test equatorial station
    ])

def findHubs (geodeticGS, jd, cartSat, elevCrit):
    
    Ngs = geodeticGS.shape[0] # number of ground stations
    Nsat = cartSat.shape[0] # number of satellites
    
    hubs = np.array([],dtype='uint16')
    
    for i in range(Ngs): # for each GS
        # find this GS's ECF coordinates
        z0ecf = R0 * np.sin(geodeticGS[i][0]*deg)
        x0ecf = R0 * np.cos(geodeticGS[i][0]*deg) * np.cos(geodeticGS[i][1]*deg)
        y0ecf = R0 * np.cos(geodeticGS[i][0]*deg) * np.sin(geodeticGS[i][1]*deg)
        
        #convert to ECI, then repmat
        R0eci = astro.R3(-astro.gstime(jd)).dot(np.array([x0ecf, y0ecf, z0ecf]))
        R0eci_mat = np.matlib.repmat(R0eci,Nsat,1)
        
        # find auxiliary vector array (from GS to satellites)
        R1 = cartSat - R0eci_mat # from this GS to all satellites
        R1abs  = np.sqrt((R1*R1).sum(1)) # R1 magnitude
        Satabs = np.sqrt((cartSat*cartSat).sum(1)) # satellite positions magnitude
        
        beta  = np.arccos((R0eci_mat*cartSat).sum(1) / Satabs / R0) # angle at the Earth centre
        alpha = np.arccos((R1*cartSat).sum(1) / Satabs / R1abs ) # nadir angle
        elevs = - (alpha + beta) + np.pi/2 # alpha + beta + elevation = 90 deg
        
        s =  np.argsort(R1abs) #fide ascending order of distances from GS to satellites

        elevFits = elevs > elevCrit*deg # which satellites have access with given elevation constraint
        satNumbers = np.arange(Nsat,dtype='uint16')[s] #sort sat indexes with order of increasing of their distances 
        Fits = elevFits[s] # same for elev angle mask to not mess up everithing
        hubs = np.append (hubs, satNumbers[Fits][:int(geodeticGS[i,2])])
    #print(hubs.sort())
    
    hubs.sort()
    hubs = np.unique(hubs)
    
    return hubs


def findHubs2 (geodeticGS, jd, cartSat,eshs, elevCrit):
    Ngs = geodeticGS.shape[0] # number of ground stations
    satNumbers = np.arange(cartSat.shape[0])
    hubs = np.array([],dtype='uint16')
    for i in range(Ngs): # for each GS
        # find this GS's ECF coordinates
        z0ecf = R0 * np.sin(geodeticGS[i][0]*deg)
        x0ecf = R0 * np.cos(geodeticGS[i][0]*deg) * np.cos(geodeticGS[i][1]*deg)
        y0ecf = R0 * np.cos(geodeticGS[i][0]*deg) * np.sin(geodeticGS[i][1]*deg)
        
        #convert to ECI, then repmat
        R0eci = astro.R3(-astro.gstime(jd)).dot(np.array([x0ecf, y0ecf, z0ecf]))
        
        esh = 0
        ForThiGS = np.array([],dtype='uint16')
        while ForThiGS.size<8:
            sats = cartSat[eshs==esh]
            Nsat = sats.shape[0]
            satN = satNumbers[eshs==esh]
            # find auxiliary vector array (from GS to satellites)
            R0eci_mat = np.matlib.repmat(R0eci,Nsat,1)
            R1 = sats - R0eci_mat # from this GS to all satellites
            R1abs  = np.sqrt((R1*R1).sum(1)) # R1 magnitude
            Satabs = np.sqrt((sats*sats).sum(1)) # satellite positions magnitude

            beta  = np.arccos((R0eci_mat*sats).sum(1) / Satabs / R0) # angle at the Earth centre
            alpha = np.arccos((R1*sats).sum(1) / Satabs / R1abs ) # nadir angle
            elevs = - (alpha + beta) + np.pi/2 # alpha + beta + elevation = 90 deg

            s =  np.argsort(R1abs) #fide ascending order of distances from GS to satellites
            elevFits = elevs > elevCrit*deg # which satellites have access with given elevation constraint
            Fits = elevFits[s] # same for elev angle mask to not mess up everithing
            satNasc = satN[s]
            new= satNasc[Fits]
            if new.size!=0:
                ForThiGS = np.append (ForThiGS, new[0])
            esh = esh+1 if esh < eshs.max() else 0
        hubs = np.append (hubs, ForThiGS)
    
    hubs.sort()
    hubs = np.unique(hubs)
    return hubs