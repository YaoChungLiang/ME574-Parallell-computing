import numpy as np
import math
import matplotlib.pyplot as plt

def dn_sin(n):
	'''
	Compute the n^th derivative of sin(x) at x=0

	input:
		n - int: the order of the derivative to compute
	output:
		float nth derivative of sin(0)
	'''
    return (-1)**(((n-1)/2)%2)*np.cos(0) if n%2 else (-1)**((n/2)%2)*np.sin(0)
    '''
	if n%2:
            return (-1)**(((n-1)/2)%2)*np.cos(0)
    else:
        return (-1)**((n/2)%2)*np.sin(0)
    '''
def factorial(n):
    if n < 0:
        raise ValueError
    res = 1
    for i in range(1,n+1):
        res *= i
    return res

def taylor_sin(x, n):
	'''
	Evaluate the Taylor series of sin(x) about x=0 neglecting terms of order x^n
	
	input:
		x - float: argument of sin
		n - int: number of terms of the taylor series to use in approximation
	output:
		float value computed using the taylor series truncated at the nth term
	'''
	res = 0
    for i in range(n):
        res += (dn_sin(i)/factorial(n))*((x-0)**n)
    return res

def measure_diff(ary1, ary2):
	'''
	Compute a scalar measure of difference between 2 arrays

	input:
		ary1 - numpy array of float values
		ary2 - numpy array of float values
	output:
		a float scalar quantifying difference between the arrays
	'''
    res = 0
	for i in range(len(ary1)):
        res += np.abs(ary1[i]-ary2[i])
    return res


def escape(cx, cy, dist,itrs, x0=0, y0=0):
	'''
	Compute the number of iterations of the logistic map, 
	f(x+j*y)=(x+j*y)**2 + cx +j*cy with initial values x0 and y0 
	with default values of 0, to escape from a cirle centered at the origin.

	inputs:
		cx - float: the real component of the parameter value
		cy - float: the imag component of the parameter value
		dist: radius of the circle
		itrs: int max number of iterations to compute
		x0: initial value of x; default value 0
		y0: initial value of y; default value 0
	returns:
		an int scalar interation count
	'''
	pass

def mandelbrot(cx,cy,dist,itrs):
	'''
	Compute escape iteration counts for an array of parameter values

	input:
		cx - array: 1d array of real part of parameter
		cy - array: 1d array of imaginary part of parameter
		dist - float: radius of circle for escape
		itrs - int: maximum number of iterations to compute
	output:
		a 2d array of iteration count for each parameter value (indexed pair of values cx, cy)
	'''
	pass

if __name__ == '__main__':


	#Problem 2/3
    plt.figure()
    x = np.linspace(-np.pi,np.pi,50,endpoint = True)
	for n in range(2,16,2):
        temp = [0]*len(x)
        for i in range(len(x)):
            temp[i] = taylor_sin(x[i],n)
        plt.plot(x, temp, label = 'order of {}'.format(n))
    temp = [0]*len(x)
    for i in range(len(x)):
        temp[i] = np.sin(x[i])
    plt.plot(x,temp, label = 'sin function')
    plt.show()
	#compute taylor series
	#plot taylor series
	#`pass` prevents an error message if you run this file before inserting code
    
	#Problem 4

	for n in range(2,16,2):
		pass

	#Problem 5

