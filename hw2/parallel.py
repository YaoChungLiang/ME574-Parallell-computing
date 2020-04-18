import math
import numpy as np
from numba import jit, cuda, float32

PI = np.pi
TPB = 32

@cuda.jit(device = True)
def s(x0):
    return (1.-2.*math.sin(PI*x0)**2)
     
@cuda.jit #Lazy compilation
#@cuda.jit('void(float32[:], float32[:])') #Eager compilation
def sKernel(d_f, d_x):
    i = cuda.grid(1)
    n = d_x.shape[0]    
    if i < n:
        d_f[i] = s(d_x[i])

def sArray(x):
    n = x.shape[0]
    d_x = cuda.to_device(x)
    d_f = cuda.device_array(n, dtype = np.float32) #need dtype spec for eager compilation
    blockdims = TPB
    gridDims = (n+TPB-1)//TPB
    sKernel[gridDims, blockDims](d_f, d_x)
    return d_f.copy_to_host()
