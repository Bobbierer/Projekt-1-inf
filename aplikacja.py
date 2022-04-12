# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 19:56:22 2022

@author: IMikulak
"""

from Transformacje import Transformacje
import numpy as np

geo = Transformacje(model = input("Podaj model: "))
plik = "wsp.txt"
dane = np.genfromtxt(plik, delimiter=',', skip_header = 4)

# Obliczenie f, l, h
flh = []
for x in range(dane.shape[0]):
    flh.append(geo.xyz2flh(dane[x][0], dane[x][1], dane[x][2]))

dane = np.c_[dane, np.array(flh)]

# Obliczenie N, E, U
neu = []
for x in range(dane.shape[0]):
    neu.append(geo.xyz2neu(dane[x][3], dane[x][4], dane[x][5], dane[x][0], dane[x][1], dane[x][2]))
    
dane = np.c_[dane, np.array(neu)]

# Obliczenie xgk, ygk
gk = []
for x in range(dane.shape[0]):
   gk.append(geo.gauss_kruger(dane[x][3], dane[x][4]))
dane = np.c_[dane, np.array(gk)]
# Obliczenie u2000
xy2000 = []
for x in range(dane.shape[0]):
   xy2000.append(geo.u2000(dane[x][3], dane[x][4]))
dane = np.c_[dane, np.array(xy2000)]

# Obliczenie u1992
xy1992 = []
for x in range(dane.shape[0]):
    xy1992.append(geo.u2000(dane[x][3], dane[x][4]))
dane = np.c_[dane, np.array(xy1992)]

# Obliczenie kąta azymutu i kąta elewacji
az_i_el = []
for x in range(dane.shape[0]):
    az_i_el.append(geo.azymut_elewacja(dane[x][3], dane[x][4], dane[x][5], dane[x][0], dane[x][1], dane[x][2]))
dane = np.c_[dane, np.array(az_i_el)]

# Odległości 2d oraz 3d
odl_2d_3d = []
for x in range(dane.shape[0]):
    try:
        odl_2d_3d.append([geo.odl2D(dane[x], dane[x+1]), geo.odl3D(dane[x], dane[x+1])])
    except IndexError:
        odl_2d_3d.append([0,0])

dane = np.c_[dane, np.array(odl_2d_3d)]
