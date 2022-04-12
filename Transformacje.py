# -*- coding: utf-8 -*-
"""
Spyder Editor,

This is a temporary script file..
"""

from math import sin, cos, sqrt, tan, atan, degrees, pi, atan2, asin
import numpy as np

class Transformacje:

    def __init__(self, model: str = "wgs84"):
        if model == "wgs84":
            self.a = 6378137
            self.b = 6356752.31424518
        elif model == "grs80":
            self.a = 6378137
            self.b = 6356752.31414036
        else:
            raise NotImplementedError(f"{model} model not implemented")
        self.flattening = (self.a - self.b) / self.a
        self.ecc2 = 2 * self.flattening - self.flattening ** 2
    def Np(self, phi, a, e2):
        N = a/(1 - self.ecc2*(sin(phi))**2)**(0.5)
        return N
    
        
        
    def xyz2flh(self, X, Y, Z):
        r   = sqrt(X**2 + Y**2)           # promień
        phi = atan(Z / (r * (1 - self.ecc2)))    # pierwsze przyblizenie
        j=0
        phi_next = phi
        while True:
            j+=1
            phi_prev = phi_next
            N = N = self.a/(1 - self.ecc2*(sin(phi_next))**2)**(0.5)
            hel  = (r/cos(phi_prev))- N
            phi_next = atan(Z/(r *(1 - (self.ecc2 * (N/(N + hel))))))
            phi_next = phi_next
            if abs(phi_next - phi_prev) < (0.0000001/206265):  # rho'' =206265
                break
            phi = phi_prev
            lam   = atan(Y/X)
            N   = self.a/(1 - self.ecc2*(sin(phi))**2)**(0.5)
            hel = (r/cos(phi))- N
            return degrees(phi), degrees(lam), hel
        
    def flh2xyz(self, phi, lam, hel):
        phi = phi * pi/180
        lam = lam * pi/180
        N = self.a/(1 - self.ecc2*(sin(phi))**2)**(0.5)
        Xkon = (N+hel)*cos(phi)*cos(lam)
        Ykon = (N+hel)*cos(phi)*sin(lam)
        Zkon = (N*(1-self.ecc2)+hel)*sin(phi)
        return(Xkon, Ykon, Zkon)
    
    def xyz2neu(self, phi, lam, h, X, Y, Z):
        R = np.array([(-sin(phi)*cos(lam), -sin(lam), cos(phi) * cos(lam)), (-sin(phi)*sin(lam), cos(lam), cos(phi)*sin(lam)), (cos(phi), 0, sin(phi))])
        N = self.Np(phi,self.a,self.ecc2)
        Xp = (N + h) * cos(phi) * cos(lam)
        Yp = (N + h) * cos(phi) * sin(lam)
        Zp = (N * (1 - self.ecc2) + h) * sin(phi)
        XYZp = np.array([Xp, Yp, Zp])
        XYZs = np.array([X, Y, Z])
        XYZ = XYZs - XYZp
        XYZ = np.array(XYZ)
        n, e, u = R.T @ XYZ
        return(n, e, u)
    
    def sigma(self, phi, a, ecc2):
        A0 = 1 - (self.ecc2/4) - ((3*(self.ecc2**2))/64) - ((5*(self.ecc2**3))/256);
        A2 = (3/8) * (self.ecc2+(self.ecc2**2/4)+((15*(self.ecc2**3))/128));
        A4 = (15/256) * ((self.ecc2**2)+((3*(self.ecc2**3))/4));
        A6 = (35*(self.ecc2**3))/3072;
        sig = self.a*(A0*phi - A2*sin(2*phi) + A4*sin(4*phi) - A6*sin(6*phi))
        return(sig)
    
    def gauss_kruger(self, phi, lam):
        phi = phi * pi/180
        lam = lam * pi/180
        b2 = (self.a**2)*(1-self.ecc2);
        ep2 = ((self.a**2)-(b2))/(b2);
        t = tan(phi);
        n2 = ep2 * ((cos(phi))**2);
        N1 = self.a/(1 - self.ecc2*(sin(phi))**2)**(0.5)
        A0 = 1 - (self.ecc2/4) - ((3*(self.ecc2**2))/64) - ((5*(self.ecc2**3))/256);
        A2 = (3/8) * (self.ecc2+(self.ecc2**2/4)+((15*(self.ecc2**3))/128));
        A4 = (15/256) * ((self.ecc2**2)+((3*(self.ecc2**3))/4));
        A6 = (35*(self.ecc2**3))/3072;
        sig = self.a*(A0*phi - A2*sin(2*phi) + A4*sin(4*phi) - A6*sin(6*phi))
        if lam < 17.5*pi/180:
            L0 = 15
            nrS = 5
        elif lam > 17.5*pi/180 and lam < 20.5*pi/180:
            L0 = 18
            nrS = 6
        elif lam > 20.5*pi/180 and lam < 22.5*pi/180:
            L0 = 21
            nrS = 7
        elif lam > 22.5*pi/180:
            L0 = 24
            nrS = 8
        dL = lam - L0*pi/180;
        xgk = sig + ((dL**2)/2)*N1*sin(phi)*cos(phi)*(1+((dL**2)/12)*((cos(phi))**2)*(5-(t**2)+(9*n2)+(4*(n2**2)))+((dL**4)/360)*((cos(phi))**4)*(61-(58*(t**2))+(t**4)+(270*n2)-(330*n2*(t**2))));
        ygk = dL * N1 * cos(phi) * (1+((dL**2)/6)*((cos(phi))**2)*(1-t**2+n2)+((dL**4)/120)*((cos(phi))**4)*(5-18*(t**2)+t**4 + 14*n2 -58*n2*(t**2)));
        
        return(xgk, ygk, nrS)
    

    def u2000(self, phi, lam):
        xgk, ygk, nrS = self.gauss_kruger(phi, lam)
        x2000 = xgk * 0.999923
        y2000 = ygk * 0.999923 + ((nrS * 1000000)+500000)
        
        return(x2000, y2000)
    
    def u1992(self, phi, lam):
        xgk, ygk, nrS = self.gauss_kruger(phi, lam)
        x92 = (xgk * 0.9993) - 5300000
        y92 = (ygk * 0.9993) + 500000
        
        return(x92, y92)
    
    def azymut_elewacja(self, phi, lam, h, x, y, z):

        n, e, u = self.xyz2neu(phi, lam, h, x, y, z)
        azymut = atan2(e, n)
        azymut = degrees(azymut)
        azymut = azymut + 360 if azymut < 0 else azymut
        elewacja = asin(u/(sqrt(e**2+n**2+u**2)))
        elewacja = degrees(elewacja)
        return azymut,elewacja
    
    def odl3D(self, A, B):
        od = sqrt( (A[0] - B[0])**2 + (A[1] - B[1])**2 + (A[2] - B[2])**2 )
        return(od)
    
    def odl2D(self, A, B):
        odl2d = sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)
        return(odl2d)
    

if __name__ == "__main__":
    # utworzenie obiektu
    geo = Transformacje(model = "wgs84")
    # dane XYZ geocentryczne
    X = 3664940.500
    Y = 1409153.590
    Z = 5009571.170
    phi, lam, hel = geo.xyz2flh(X, Y, Z)
    print(phi, lam, hel)
    n, e, u = geo.xyz2neu(phi, lam)
    print(n, e, u)
    xk, yk, zk = geo.flh2xyz(phi, lam, hel)
    print(xk,yk,zk)
    xgk, ygk, nrS = geo.gauss_kruger(phi, lam)
    print(xgk, ygk, nrS)
    x2000, y2000 = geo.u2000(xgk, ygk)
    print(x2000, y2000)
    x92, y92 = geo.u1992(xgk, ygk)
    print(x92, y92)
        