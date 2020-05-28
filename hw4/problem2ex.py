import numpy as np
from numba import cuda, float32, int32, jit

# Richarson extrapolation
def RicharsonRecur(f,x,h,n):
    if n <= 2:
        return float(f(x+h)-f(x-h))/(2*h)
    else:
        return float(2**(n-2)*Dc(f,x,h,n-2)-Dc(f,x,2*h,n-2))/(2**(n-2)-1)

def f(x):
    return 3*x**2+2*np.sin(x)+np.exp(x)

def df(x):
    return 6*x + 2*np.cos(x) + np.exp(x)


def Si(x):
    return np.sin(x)/float(x)

def testRicharson():
    n = 2
    x0 = 2.0
    h = 0.1
    a = RicharsonRecur(f, x0, h, n)
    print(df(x0) , a)

def IntegralSi():
    a = 0
    b = 50
    n = 6001
    if a == 0:
        a = (b-a)/float(n) + a 
    print(Simpson(Si, a ,b, n))


def testSimpson():
    a = 1
    b = 2
    n = 10
    print(Simpson(f, a,b ,n))

def Simpson(f,a,b,n):
    if a == 0:
        a = (b-a)/float(n) + a
        h = (b-a)/float(n-1)
    else:
        h = (b-a)/n
    k = 0.0
    x = a + h
    for i in range(1,int(n/2)+1):
        k += 4*f(x)
        x += 2*h
    x = a + 2*h
    for i in range(1,int(n/2)):
        k += 2*f(x)
        x += 2*h
    return (h/3)*(f(a) + f(b) + k)

if __name__ == "__main__":
    #testRicharson()
    #testSimpson()
    IntegralSi()
