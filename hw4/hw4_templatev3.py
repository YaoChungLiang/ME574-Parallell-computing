from math import sin, sinh

import matplotlib.pyplot as plt
import numpy as np
from numba import cuda, float32, int32, jit
import time

tR = 1.0
freq = 1000
t_steps = int(freq*tR + 1)
timeRange = np.linspace(0.0, tR, t_steps, endpoint=True)
dt = timeRange[1] - timeRange[0]

NY, NX = 151, 151
PI = np.pi
TPB = 8
RAD = 2
#SH_N = NX+2*RAD
SH_N = 25
X = np.linspace(0.0, 1.0, NX, endpoint=True)
Y = np.linspace(0.0, 1.0, NY, endpoint=True)
h = 1.0/float(NX)


@cuda.reduce
def max_kernel(a, b):
    return max(a, b)


@cuda.reduce
def sum_kernel(a, b):
    return a + b


@cuda.jit
def heat_step(d_u, d_out, stencil, t):
    # h=1.0/151.0
    edge = stencil[0]
    corner = stencil[1]
    i, j = cuda.grid(2)
    dims = d_u.shape
    if i >= dims[0] or j >= dims[1]:
        return
    NX, NY = cuda.blockDim.x, cuda.blockDim.y
    t_i, t_j = cuda.threadIdx.x, cuda.threadIdx.y
    sh_i, sh_j = t_i + RAD, t_j + RAD

    sh_u = cuda.shared.array(shape=(SH_N, SH_N), dtype=float32)

    sh_u[sh_i, sh_j] = d_u[i, j]

    # Halo edge values assignment
    if t_i < RAD:
        sh_u[sh_i - RAD, sh_j] = d_u[i - RAD, j]
        sh_u[sh_i + NX, sh_j] = d_u[i + NX, j]

    if t_j < RAD:
        sh_u[sh_i, sh_j - RAD] = d_u[i, j - RAD]
        sh_u[sh_i, sh_j + NY] = d_u[i, j + NY]

    # Halo corner values assignment
    if t_i < RAD and t_j < RAD:
        # upper left
        sh_u[sh_i - RAD, sh_j - RAD] = d_u[i - RAD, j - RAD]
    # upper right
        sh_u[sh_i + NX, sh_j - RAD] = d_u[i + NX, j - RAD]
    # lower left
        sh_u[sh_i - RAD, sh_j + NY] = d_u[i - RAD, j + NY]
    # lower right
        sh_u[sh_i + NX, sh_j + NY] = d_u[i + NX, j + NY]

    cuda.syncthreads()

    if i > 1 and j > 1 and i <= dims[0] and j <= dims[1]:
        d_out[i, j] = \
            (sh_u[sh_i, sh_j-1]*edge +
             sh_u[sh_i, sh_j+1]*edge +
             sh_u[sh_i-1, sh_j]*edge +
             sh_u[sh_i+1, sh_j]*edge -
             sh_u[sh_i, sh_j] * -1.0)*h*h*dt +\
            sh_u[sh_i, sh_j]


def p1():
    # initialization
    global X, Y
    u = np.zeros(shape=[NX, NY], dtype=np.float32)
    # initial condition
    stencil = np.array([0.25, 0])
    for i in range(len(X)):
        for j in range(len(Y)):
            u[i][j] = sin(2*PI*X[i])*sin(PI*Y[j])
    # boundary conditions
    for j in range(NY):
        u[0][j] = 0
        u[1][j] = 0
    for i in range(NX):
        u[i][0] = 0
        u[i][1] = 0
    d_u = cuda.to_device(u)
    d_out = cuda.to_device(u)
    dims = u.shape
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB]
    blockSize = [TPB, TPB]
    u_max = np.max(u)
    print(u_max)
    for i in range(len(timeRange)):
        heat_step[gridSize, blockSize](d_u, d_out, stencil, timeRange[i])
        heat_step[gridSize, blockSize](d_out, d_u, stencil, timeRange[i])
        u = d_u.copy_to_host()
        print(np.max(u))
        print(i)
        if np.max(u) <= u_max*np.exp(-2):
            print(timeRange[i])
            break
    xvals = np.linspace(0., 1.0, NX)
    yvals = np.linspace(0., 1.0, NY)
    X, Y = np.meshgrid(xvals, yvals)
    levels = [0.025, 0.1, 0.25, 0.5, 0.75]
    #plt.contourf(X, Y,exact.T, levels = levels)
    plt.contour(X, Y, u.T, levels=levels, colors="r", linewidths=4)
    plt.axis([0, 1, 0, 1])
    plt.savefig('./tRange%d_freq%d.jpg' % (tR, freq))
    plt.show()


@cuda.jit
def integrate_kernel(d_y, d_out, quad):
    '''
    y: input device array
    out: output device array
    quad: quadrature stencil coefficients
    '''
    i = cuda.grid(1)
    if i < d_out.size:
        d_out[i] = d_y[2*i] * quad[0] + d_y[2*i+1] * \
            quad[1] + d_y[2*i+2] * quad[2]


def integrate(y, quad):
    '''
    y: input array
    quad: quadrature stencil coefficients
    '''
    n = int((len(y)-1)/2)
    d_y = cuda.to_device(y)
    d_out = cuda.device_array(n, dtype=np.float32)
    TPBx = 32
    gridDim = (n+TPBx-1)//TPBx
    blockDim = TPBx
    integrate_kernel[gridDim, blockDim](d_y, d_out,  quad)
    # here it is
    return d_out.copy_to_host()


@cuda.jit
def monte_carlo_kernel_sphere_intertia(rng_states, iters, out):
    '''
    rng_states: rng state array generated from xoroshiro random number generator
    iters: number of monte carlo sample points each thread will test
    out: output array
    '''
    pass


@cuda.jit
def monte_carlo_kernel_sphere_vol(rng_states, iters, out):
    pass


@cuda.jit
def monte_carlo_kernel_shell_intertia(rng_states, iters, out):
    pass


@cuda.jit
def monte_carlo_kernel_shell_vol(rng_states, iters, out):
    pass


def monte_carlo(threads, blocks, iters, kernel, seed=1):
    '''
    threads: number of threads to use for the kernel
    blocks: number of blocks to use for the kernel
    iters: number of monte carlo sample points each thread will test 
    kernel: monte_carlo kernel to use
    seed: seed used when generating the random numbers (if the seed is left at one the number generated will be the same each time)
    '''
    pass


@cuda.jit(device=True)
def chi(f, levelset):
    '''
    f: function value
    levelset: surface levelset
    '''
    return f <= levelset


@cuda.jit
def grid_integrate_sphere_intertia(y, out, stencil):
    '''
    y: input device array
    out: output device array
    stencil: derivative stencil coefficients
    '''
    pass


@cuda.jit
def grid_integrate_sphere_vol(y, out, stencil):
    pass


@cuda.jit
def grid_integrate_shell_intertia(y, out, stencil):
    pass


@cuda.jit
def grid_integrate_shell_vol(y, out, stencil):
    pass


def grid_integrate(kernel):
    '''
    kernel: grid integration kernel to use
    '''
    pass


def RicharsonRecur(f, x, h, n):
    if n <= 2:
        return float(f(x+h)-f(x-h))/(2*h)
    else:
        return float(2**(n-2)*RicharsonRecur(f, x, h, n-2)-RicharsonRecur(f, x, 2*h, n-2))/(2**(n-2)-1)


def Si(x):
    if x == 0:
        return 1
    return np.sin(x)/float(x)


def p2():
    n = 6001
    a = 0
    b = 50
    h = (b-a)/float(n-1)
    arr = np.linspace(a, b, n, endpoint=True)
    for i in range(len(arr)):
        arr[i] = Si(arr[i])
    quar = np.array([1, 4, 1])
    resArr = integrate(arr, quar)
    res = 0
    for i in resArr:
        res += i
        # print(i)
    print("Without Richarson")
    print(res*h/3.0)


def p2withRichar():
    n = 6001
    a = 0
    b = 50
    h = (b-a)/float(n-1)
    arr = np.linspace(a, b, n, endpoint=True)
    arrRichar = np.zeros(2*n-1)
    for i in range(len(arr)-1):
        arrRichar[2*i] = Si(arr[i])
        arrRichar[2*i+1] = arrRichar[2*i] + h/2.0*RicharsonRecur(Si, arr[i], h, 2)
    arrRichar[-1] = Si(arr[-1])
    quar = np.array([1, 4, 1])

    resArr = integrate(arrRichar, quar)
    res = 0
    for i in resArr:
        res += i
    print("With Richarsin")
    print(res*h/2.0/3.0)


if __name__ == '__main__':
    pass
    # p1()
    #p2()
    #p2withRichar()
