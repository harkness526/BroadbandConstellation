# -*- coding: utf-8 -*-

import numpy as np
deg = np.pi/180
R0 = 6378.137

def R3(x): # simple rotation arount the 3rd axis    
    A = np.array ([ [np.cos(x), np.sin(x), 0],
    [-np.sin(x), np.cos(x), 0],
    [0, 0, 1]])
    return A   

def gstime(jdut1): # greenwich sidereal time of the given JD (based on Vallado)

        tut1 = ( jdut1 - 2451545.0 ) / 36525.0;

        temp = -6.2e-6 * (tut1**3) + 0.093104 * (tut1**2) + (876600.0 * 3600.0 + 8640184.812866) * tut1 + 67310.54841;

        #temp = temp % (2*np.pi)
        temp = np.mod(temp*deg/240, 2*np.pi)

        return temp
    
    
def jday (yr, mon, day, hr, mi, sec): # Julian day from Gregorian UTC date/time
        jd = 367.0 * yr  - np.floor( (7 * (yr + np.floor( (mon + 9) / 12.0) ) ) * 0.25 ) + np.floor( 275 * mon / 9.0 ) + day + 1721013.5 + ( (sec/60.0 + mi ) / 60.0 + hr ) / 24.0
        return jd
