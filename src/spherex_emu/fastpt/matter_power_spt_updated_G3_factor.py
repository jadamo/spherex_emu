'''
* Module to calculate the matter power specturm in
Standard Pertrubation Theory at one-loop.
* This module uses the routine J_k to calculate each Legendre
component of the matter power spectrum kernals as given in the appendix of
XXX.

Author: J. E. McEwen, 2015 & Xiao Fang
email : jmcewen314@gmail.com

'''

import numpy as np
from numpy import log, sqrt, exp, pi
from scipy.signal import fftconvolve
from .J_k import J_k
import sys


def P_22(k,P,P_window,C_window,n_pad):

	# P_22 Legendre components
	# We calculate a regularized version of P_22
	# by omitting the J_{2,-2,0} term so that the
	# integral converges.  In the final power spectrum
	# we add the asymptotic portions of P_22 and P_13 so
	# that we get a convergent integral.  See section of XXX.

	param_matrix=np.array([[0,0,0,0],[0,0,2,0],[0,0,4,0],[2,-2,2,0],\
							[1,-1,1,0],[1,-1,3,0],[2,-2,0,1] ])


	Power, mat=J_k(k,P,param_matrix,P_window=P_window,C_window=C_window,n_pad=n_pad)
	A=1219/1470.*mat[0,:]
	B=671/1029.*mat[1,:]
	C=32/1715.*mat[2,:]
	D=1/3.*mat[3,:]
	E=62/35.*mat[4,:]
	F=8/35.*mat[5,:]
	reg=1/3.*mat[6,:]

	return Power, 2*(A+B+C+D+E+F)+ reg

#This is the original code to calculate regularized P_13 integral for the F3 kernel
def P_13_regF(k,P):

	# calculates the regularized version of P_13 by
	# a discrete convolution integral

	N=k.size
	n= np.arange(-N+1,N )
	dL=log(k[1])-log(k[0])
	s=n*dL

	cut=7
	high_s=s[s > cut]
	low_s=s[s < -cut]
	mid_high_s=s[ (s <= cut) &  (s > 0)]
	mid_low_s=s[ (s >= -cut) &  (s < 0)]

    #Z_reg found in eq. 2.30 in FAST-PT paper
	Z=lambda r : (12./r**2 +10. + 100.*r**2-42.*r**4 \
	+ 3./r**3*(r**2-1.)**3*(7*r**2+2.)*log((r+1.)/np.absolute(r-1.)) ) *r
    #Equations for Z_reg that include the expansions around small and large r
    #Z_low=Zhigh from the notebook and Z_high = Zlow
    #Not sure why but Z_low=Zlow and Z_high=Zhigh gives a result of the order 10^50
	Z_low=lambda r : (352./5.+96./5./r**2 -160./21./r**4 - 526./105./r**6 +236./35./r**8) *r
	Z_high=lambda r: (928./5.*r**2 - 4512./35.*r**4 +416./21.*r**6 +356./105.*r**8) *r

	f_mid_low=Z(exp(-mid_low_s))
	f_mid_high=Z(exp(-mid_high_s))
	f_high = Z_high(exp(-high_s))
	f_low = Z_low(exp(-low_s))

	f=np.hstack((f_low,f_mid_low,80,f_mid_high,f_high))

	g= fftconvolve(P, f) * dL
	g_k=g[N-1:2*N-1]
	P_bar= 1./252.* k**3/(2*pi)**2*P*g_k

	return P_bar

#Calculate P13 without using regularization
def P_13_F(k,P):

	# calculates the regularized version of P_13 by
	# a discrete convolution integral

	N=k.size
	n= np.arange(-N+1,N )
	dL=log(k[1])-log(k[0])
	s=n*dL

	cut=7
	high_s=s[s > cut]
	low_s=s[s < -cut]
	mid_high_s=s[ (s <= cut) &  (s > 0)]
	mid_low_s=s[ (s >= -cut) &  (s < 0)]

    #Z_reg found in eq. 2.30 in FAST-PT paper
	Z=lambda r : (12./r**2 -158. + 100.*r**2-42.*r**4 \
	+ 3./r**3*(r**2-1.)**3*(7*r**2+2.)*log((r+1.)/np.absolute(r-1.)) ) *r
    #Equations for Z_reg that include the expansions around small and large r
    #Z_low=Zhigh from the notebook and Z_high = Zlow
    #Not sure why but Z_low=Zlow and Z_high=Zhigh gives a result of the order 10^50
	Z_low=lambda r : (-97.6+96./5./r**2 -160./21./r**4 - 526./105./r**6 +236./35./r**8) *r
	Z_high=lambda r: (-168.+928./5.*r**2 - 4512./35.*r**4 +416./21.*r**6 +356./105.*r**8) *r

	f_mid_low=Z(exp(-mid_low_s))
	f_mid_high=Z(exp(-mid_high_s))
	f_high = Z_high(exp(-high_s))
	f_low = Z_low(exp(-low_s))

	f=np.hstack((f_low,f_mid_low,-88,f_mid_high,f_high))

	g= fftconvolve(P, f) * dL
	g_k=g[N-1:2*N-1]
	P_bar= 1./252.* k**3/(2*pi)**2*P*g_k

	return P_bar

#My results for Z_reg calculated with G3 kernel
#The code here is the same as above except for Z, Z_low, and Z_high
def P_13_regG(k,P):

	# calculates the regularized version of P_13 by
	# a discrete convolution integral

	N=k.size
	n= np.arange(-N+1,N )
	dL=log(k[1])-log(k[0])
	s=n*dL

	cut=7
	high_s=s[s > cut]
	low_s=s[s < -cut]
	mid_high_s=s[ (s <= cut) &  (s > 0)]
	mid_low_s=s[ (s >= -cut) &  (s < 0)]

    #My additions
	Z=lambda r : (12./r**2 -26. + 4.*r**2-6.*r**4 \
	+ 3./r**3*(r**2-1.)**3*(r**2+2.)*log((r+1.)/np.absolute(r-1.)) ) *r
	Z_low=lambda r : (-44.8+1248./35./r**2 -608./105./r**4 - 160./231./r**6) *r
	Z_high=lambda r: (-6.4*r**2 - 13.7143*r**4 +352./105.*r**6 + 544./1155*r**8 ) *r
    

	f_mid_low=Z(exp(-mid_low_s))
	f_mid_high=Z(exp(-mid_high_s))
	f_high = Z_high(exp(-high_s))
	f_low = Z_low(exp(-low_s))

	f=np.hstack((f_low,f_mid_low,-16,f_mid_high,f_high))

	g= fftconvolve(P, f) * dL
	g_k=g[N-1:2*N-1]
    #changed from 1/252.
	P_bar= 1./84.* k**3/(2*pi)**2*P*g_k

	return P_bar

#Does not subtract diverging term in P13G
def P_13_G(k,P):

	# calculates the regularized version of P_13 by
	# a discrete convolution integral

	N=k.size
	n= np.arange(-N+1,N )
	dL=log(k[1])-log(k[0])
	s=n*dL

	cut=7
	high_s=s[s > cut]
	low_s=s[s < -cut]
	mid_high_s=s[ (s <= cut) &  (s > 0)]
	mid_low_s=s[ (s >= -cut) &  (s < 0)]

    #My additions
	Z=lambda r : (12./r**2 -82. + 4.*r**2-2.*r**4 \
	+ 3./r**3*(r**2-1.)**3*(r**2+2.)*log((r+1.)/np.absolute(r-1.)) ) *r
	Z_low=lambda r : (-504./5.+1248./35./r**2 -608./105./r**4) *r
	Z_high=lambda r: (-56-32./5.*r**2 - 68./7.*r**4 +352./105.*r**6) *r
    

	f_mid_low=Z(exp(-mid_low_s))
	f_mid_high=Z(exp(-mid_high_s))
	f_high = Z_high(exp(-high_s))
	f_low = Z_low(exp(-low_s))

	f=np.hstack((f_low,f_mid_low,-68,f_mid_high,f_high))

	g= fftconvolve(P, f) * dL
	g_k=g[N-1:2*N-1]
	P_bar= 1./84.* k**3/(2*pi)**2*P*g_k

	return P_bar

#My results for Z_reg calculated with F3+G3 kernel
def P_13_regFG(k,P):

	# calculates the regularized version of P_13 by
	# a discrete convolution integral

	N=k.size
	n= np.arange(-N+1,N )
	dL=log(k[1])-log(k[0])
	s=n*dL

	cut=7
	high_s=s[s > cut]
	low_s=s[s < -cut]
	mid_high_s=s[ (s <= cut) &  (s > 0)]
	mid_low_s=s[ (s >= -cut) &  (s < 0)]

    #my additions
	Z=lambda r : (24./r**2 -16. + 104.*r**2-48.*r**4 \
	+ 3./r**3*(r**2-1.)**3*(8*r**2+4.)*log((r+1.)/np.absolute(r-1.)) ) *r
	Z_low=lambda r : (25.6+384./7./r**2 -1408./105./r**4 - 2176./1155./r**6) *r
	Z_high=lambda r: (179.2*r**2 - 142.629*r**4 +2432./105.*r**6 + 640./231.*r**8) *r
  

	f_mid_low=Z(exp(-mid_low_s))
	f_mid_high=Z(exp(-mid_high_s))
	f_high = Z_high(exp(-high_s))
	f_low = Z_low(exp(-low_s))

	f=np.hstack((f_low,f_mid_low,64,f_mid_high,f_high))

	g= fftconvolve(P, f) * dL
	g_k=g[N-1:2*N-1]
	P_bar= 1./252.* k**3/(2*pi)**2*P*g_k

	return P_bar

#Does not subtract diverging term in P13_F3+G3
def P_13_FG(k,P):

	# calculates the regularized version of P_13 by
	# a discrete convolution integral

	N=k.size
	n= np.arange(-N+1,N )
	dL=log(k[1])-log(k[0])
	s=n*dL

	cut=7
	high_s=s[s > cut]
	low_s=s[s < -cut]
	mid_high_s=s[ (s <= cut) &  (s > 0)]
	mid_low_s=s[ (s >= -cut) &  (s < 0)]

    #my additions
	Z=lambda r : (24./r**2 -240. + 104.*r**2-44.*r**4 \
	+ 3./r**3*(r**2-1.)**3*(8*r**2+4.)*log((r+1.)/np.absolute(r-1.)) ) *r
	Z_low=lambda r : (-198.4+384./7./r**2 -1408./105./r**4) *r
	Z_high=lambda r: (-224.+179.2*r**2 - 138.629*r**4 +2432./105.*r**6) *r
    

	f_mid_low=Z(exp(-mid_low_s))
	f_mid_high=Z(exp(-mid_high_s))
	f_high = Z_high(exp(-high_s))
	f_low = Z_low(exp(-low_s))

	f=np.hstack((f_low,f_mid_low,80,f_mid_high,f_high))

	g= fftconvolve(P, f) * dL
	g_k=g[N-1:2*N-1]
	P_bar= 1./252.* k**3/(2*pi)**2*P*g_k

	return P_bar

def Y1_reg_NL(k,P):

	# calculates the regularized version of P_13 by
	# a discrete convolution integral

	N=k.size
	n= np.arange(-N+1,N )
	dL=log(k[1])-log(k[0])
	s=n*dL

	cut=7
	high_s=s[s > cut]
	low_s=s[s < -cut]
	mid_high_s=s[ (s <= cut) &  (s > 0)]
	mid_low_s=s[ (s >= -cut) &  (s < 0)]

	Z=lambda r : (1./126.)*(-6./r**2 +22. + 22.*r**2-6.*r**4 \
	+ 3./r**3*(r**2-1.)**4*log((r+1.)/np.absolute(r-1.)) )*r
	Z_low=lambda r : (1./126.)*(256./5.-768./35./r**2 +256./105./r**4 + 256./1155./r**6 +256./5005./r**8)*r
	Z_high=lambda r: (1./126.)*(256./5.*r**2 - 768./35.*r**4 +256./105.*r**6 +256./1155.*r**8)*r

	f_mid_low=Z(exp(-mid_low_s))
	f_mid_high=Z(exp(-mid_high_s))
	f_high = Z_high(exp(-high_s))
	f_low = Z_low(exp(-low_s))

	f=np.hstack((f_low,f_mid_low,32./126.,f_mid_high,f_high))

	g= fftconvolve(P, f) * dL
	g_k=g[N-1:2*N-1]
	P_bar= k**3/(2*pi)**2*P*g_k

	return P_bar

def sig3nl_Test(k,P):

	# calculates sigma_3^2 by
	# a discrete convolution integral

	N=k.size
	n= np.arange(-N+1,N )
	dL=log(k[1])-log(k[0])
	s=n*dL

	cut=7
	high_s=s[s > cut]
	low_s=s[s < -cut]
	mid_high_s=s[ (s <= cut) &  (s > 0)]
	mid_low_s=s[ (s >= -cut) &  (s < 0)]

	Z=lambda r : (149./441.*r**2 -4./21. + 1./7./r**2 \
	- 1./42./r**3*(r**2-1.)**2*(3.+r**2)*log((r+1.)/np.absolute(r-1.)) )*r
	Z_low=lambda r : (-16./63.+16/45./r**2 -176./2205./r**4 - 16./1323./r**6 +304./72765./r**8)*r
	Z_high=lambda r: (752./2205.*r**2 - 16./441.*r**4 -16./2205.*r**6 +208./72765.*r**8)*r

	f_mid_low=Z(exp(-mid_low_s))
	f_mid_high=Z(exp(-mid_high_s))
	f_high = Z_high(exp(-high_s))
	f_low = Z_low(exp(-low_s))

	f=np.hstack((f_low,f_mid_low,.2902,f_mid_high,f_high))

	g= fftconvolve(P, f) * dL
	g_k=g[N-1:2*N-1]
	P_bar= (105.*k**3)/(64.*(pi)**2)*P*g_k

	return P_bar

def Y2_reg_NL(k,P):

	# calculates the regularized version of P_13 by
	# a discrete convolution integral

	N=k.size
	n= np.arange(-N+1,N )
	dL=log(k[1])-log(k[0])
	s=n*dL

	cut=7
	high_s=s[s > cut]
	low_s=s[s < -cut]
	mid_high_s=s[ (s <= cut) &  (s > 0)]
	mid_low_s=s[ (s >= -cut) &  (s < 0)]

	Z=lambda r : (1./126.)*(-6./r**2 +22. + 22.*r**2-6.*r**4 \
	+ 3./r**3*(r**2-1.)**4*log((r+1.)/np.absolute(r-1.)) )*r
	Z_low=lambda r : (1./126.)*(256./5.-768./35./r**2 +256./105./r**4 + 256./1155./r**6 +256./5005./r**8)*r
	Z_high=lambda r: (1./126.)*(256./5.*r**2 - 768./35.*r**4 +256./105.*r**6 +256./1155.*r**8)*r

	f_mid_low=Z(exp(-mid_low_s))
	f_mid_high=Z(exp(-mid_high_s))
	f_high = Z_high(exp(-high_s))
	f_low = Z_low(exp(-low_s))

	f=np.hstack((f_low,f_mid_low,32./126.,f_mid_high,f_high))

	g= fftconvolve(P, f) * dL
	g_k=g[N-1:2*N-1]
	P_bar= k**3/(2*pi)**2*P*g_k

	return P_bar


def one_loop(k,P,P_window=None,C_window=None,n_pad=None):

	P1,P22_reg=P_22(k,P,P_window,C_window,n_pad)
	P13_reg=P_13_regF(k,P1)

	return P1,P22_reg,P13_reg


