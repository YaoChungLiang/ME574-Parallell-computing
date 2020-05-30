from math import sin, sinh,sqrt, isnan

import matplotlib.pyplot as plt
import numpy as np
from numba import cuda, float32, int32, jit
from mpl_toolkits.mplot3d import Axes3D

import time
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from scipy.optimize import curve_fit

#tR = 100000.0
#freq = 0.1
#t_steps = int(freq*tR + 1)
#timeRange = np.linspace(0.0, tR, t_steps, endpoint=True)
#timeRange = np.linspace(0.0,3000.0,301,endpoint=True)
#dt = timeRange[1] - timeRange[0]
# ------------------------ #

RAD = 1
TPB = 8
SH_N = 10


@cuda.reduce
def max_kernel(a, b):
    return max(a, b)


@cuda.reduce
def sum_kernel(a, b):
    return a + b


@cuda.jit
def heat_step(d_u, d_out, stencil, dt):
    # h=1.0/151.0
    h = 1.0/(len(d_u[0])-1)
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

    if i > 0 and j > 0 and i < dims[0]-1 and j < dims[1]-1:
        d_out[i, j] = \
            (sh_u[sh_i, sh_j-1] +
             sh_u[sh_i, sh_j+1] +
             sh_u[sh_i-1, sh_j] +
             sh_u[sh_i+1, sh_j] -
             4*sh_u[sh_i, sh_j] )/h/h*dt +\
            sh_u[sh_i, sh_j]


# def jacobi_update_parallel()

def p1():
    # initialization
    #global X, Y,dt
    dt = 0.00001
    n_iter = 10000
    NY, NX = 151, 151
    PI = np.pi
    TPB = 8
    RAD = 1
    SH_N = 10 # TPB +2*RAD
    X = np.linspace(0.0, 1.0, NX, endpoint=True)
    Y = np.linspace(0.0, 1.0, NY, endpoint=True)
    h = 1.0/float(NX-1)
    
    
    
    
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
        
    xvals = np.linspace(0., 1.0, NX)
    yvals = np.linspace(0., 1.0, NY)
    XX, YY = np.meshgrid(xvals, yvals)
    levels = [-1.0,-0.75,-0.5,-0.25,-0.1,-0.025,0.0,0.025, 0.1, 0.25, 0.5, 0.75,1.0]
    plt.figure('Initial figure')
    plt.contourf(XX, YY, u.T)
    #plt.contour(X, Y, u.T)
    plt.axis([0, 1, 0, 1])
    plt.colorbar()
    #plt.show()    
        
    d_u = cuda.to_device(u)
    d_out = cuda.to_device(u)
    dims = u.shape
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB]
    blockSize = [TPB, TPB]
    u_max = np.max(u)
    print(u_max)
    i = 0
    # for i in range(n_iter):
    while(True):
        heat_step[gridSize, blockSize](d_u, d_out, stencil, dt)
        # u = d_out.copy_to_host()
        # d_u = cuda.to_device(u)
        heat_step[gridSize, blockSize](d_out, d_u, stencil, dt)
        u = d_u.copy_to_host()
        print("now = {}, goal ratio = {}".format(np.max(u)/u_max, np.exp(-2)))
        print(i)
        if np.max(u)/u_max <= np.exp(-2):
            print(f'It takes {i*dt*2} seconds to decrease by a factor of e^-2')
            break
        i += 1

    plt.figure('Reslt figure')
    plt.contourf(XX, YY, u.T)
    plt.axis([0, 1, 0, 1])
    plt.colorbar()
    plt.show()

    # axp = plt.axes(projection='3d')
    # axp.contour3D(X, Y, u.T, 100, cmap='viridis')
    # plt.savefig('./p1a.jpg')
    # plt.draw()
    # plt.show()
    
    #---------- p1-c : dt > dt_max = 0.00001 --------------------#
    dt = 0.00002
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
        
    xvals = np.linspace(0., 1.0, NX)
    yvals = np.linspace(0., 1.0, NY)
    X, Y = np.meshgrid(xvals, yvals)
    levels = [-1.0,-0.75,-0.5,-0.25,-0.1,-0.025,0.0,0.025, 0.1, 0.25, 0.5, 0.75,1.0]

        
    d_u = cuda.to_device(u)
    d_out = cuda.to_device(u)
    dims = u.shape
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB]
    blockSize = [TPB, TPB]
    u_max = np.max(u)
    print(u_max)
    i = 0
    # for i in range(n_iter):
    while(True):
        heat_step[gridSize, blockSize](d_u, d_out, stencil, dt)
        heat_step[gridSize, blockSize](d_out, d_u, stencil, dt)
        u = d_u.copy_to_host()
        if isnan(np.max(u)/u_max):
            break
        print("now = {}, goal ratio = {}".format(np.max(u)/u_max, np.exp(-2)))
        print(i)
        if np.max(u)/u_max <= np.exp(-2):
            print(f'It takes {i*dt*2} seconds to decrease by a factor of e^-2')
            break
        i += 1

    plt.figure('Result figure')
    plt.contourf(XX, YY, u.T)
    plt.axis([0, 1, 0, 1])
    plt.colorbar()
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
def monte_carlo_kernel_sphere_intertia(rng_states, iters, d_out):
    '''
    rng_states: rng state array generated from xoroshiro random number generator
    iters: number of monte carlo sample points each thread will test
    out: output array
    '''
    i = cuda.grid(1)
    counter = 0
    #if i > d_out.shape:
    #    return
    d = 0.0
    for _ in range(iters):
        x = xoroshiro128p_uniform_float32(rng_states,i)
        y = xoroshiro128p_uniform_float32(rng_states,i)
        z = xoroshiro128p_uniform_float32(rng_states,i)
        if  x**2 + y**2 + z**2 <= 1 :
            counter += 1
    # method 2, no reference 
    #d_out[i] = 8.0*counter/iters*0.314
            
    # method 1, ref:https://aapt.scitation.org/doi/abs/10.1119/1.1987089     
    # 4.188790 : the original cube has the volume of 8 and the ball inside of it is 4.188790     
            d += (y**2 + z**2)
    d_out[i] = 4.188790*(d)/float(counter)


@cuda.jit
def monte_carlo_kernel_sphere_vol(rng_states, iters, d_out):
    i = cuda.grid(1)
    counter = 0
    #if i > d_out.shape:
    #    return
    for _ in range(iters):
        x = xoroshiro128p_uniform_float32(rng_states,i)
        y = xoroshiro128p_uniform_float32(rng_states,i)
        z = xoroshiro128p_uniform_float32(rng_states,i)
        if x**2 + y**2 + z**2 <= 1:
            counter += 1
    d_out[i] = 8.0 * counter / iters


@cuda.jit
def monte_carlo_kernel_shell_intertia(rng_states, iters, d_out):
    i = cuda.grid(1)
    counter = 0
    #if i > d_out.shape:
    #    return
    tol = 1e-3
    d = 0.0
    for _ in range(iters):
        x = xoroshiro128p_uniform_float32(rng_states,i)
        y = xoroshiro128p_uniform_float32(rng_states,i)
        z = xoroshiro128p_uniform_float32(rng_states,i)
        if  1-tol <= x**2 + y**2 + z**2  <= 1+tol:
            counter += 1   
            d += (y**2 + z**2)
    d_out[i] = 4.188790*(d)/float(counter) / tol / iters


@cuda.jit
def monte_carlo_kernel_shell_vol(rng_states, iters, d_out):
    tol = 1e-3
    i = cuda.grid(1)
    counter = 0
    #if i > d_out.shape:
    #    return
    for _ in range(iters):
        x = xoroshiro128p_uniform_float32(rng_states,i)
        y = xoroshiro128p_uniform_float32(rng_states,i)
        z = xoroshiro128p_uniform_float32(rng_states,i)
        if 1-tol <= x**2 + y**2 + z**2 <= 1+tol:
            counter += 1
    d_out[i] = 8.0 * counter / iters / (tol)


def monte_carlo(threads, blocks, iters, kernel, seed=1):
    '''
    threads: number of threads to use for the kernel
    blocks: number of blocks to use for the kernel
    iters: number of monte carlo sample points each thread will test 
    kernel: monte_carlo kernel to use
    seed: seed used when generating the random numbers (if the seed is left at one the number generated will be the same each time)
    '''
    d_out_len = threads*blocks
    d_out = cuda.device_array(d_out_len, dtype = np.float32)
    rng_states = create_xoroshiro128p_states(threads*blocks,seed)
    kernel[blocks*threads, threads](rng_states, iters,d_out)
    return np.mean(d_out.copy_to_host())


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
def grid_integrate_sphere_vol(d_y, d_out, stencil):
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


def p3a():
    
    iters = 1000
    n = iters
    TPB = 32
    gridDim = (n+TPB-1)//TPB
    blockDim = TPB
    rng_states = create_xoroshiro128p_states(TPB*gridDim,seed = 1)
    #out = np.zeros(iters)
    #d_out = np.zeros(iters, dtype = np.float32)
    d_out = cuda.device_array(iters, dtype = np.float32)
    monte_carlo_kernel_sphere_vol[gridDim, blockDim](rng_states,iters,d_out)
    
    out = d_out.copy_to_host()
    MC_parallel_sphere_vol = np.mean(out)
    print(f'sphere vol = {MC_parallel_sphere_vol}')
    
    monte_carlo_kernel_sphere_intertia[gridDim,blockDim](rng_states, iters, d_out)
    out = d_out.copy_to_host()
    MC_parallel_sphere_MOI = np.mean(out)
    print(f'sphere MOI = {MC_parallel_sphere_MOI}')    
    
def p3a_with_monte():
    iters = 1000
    n = iters
    TPB = 32
    gridDim = (n+TPB-1)//TPB
    blockDim = TPB
    kernel = monte_carlo_kernel_sphere_vol
    res = monte_carlo(TPB,gridDim,iters,kernel)
    print(f'sphrical vol = {res}')
    
def p3a_serial():
    # spherical volume
    r = 1
    real_vol = 4/3.0*np.pi*r*r
    errs = []
    iters_arr = []
    n = 1000
    TPB = 32
    gridDim = (n+TPB-1)//TPB
    blockDim = TPB
    kernel = monte_carlo_kernel_sphere_vol
    for  i in range(1,6):
        iters = 10**i
        res = monte_carlo(TPB,gridDim,iters,kernel)
        print(f'sphrical vol = {res}')
        errs.append(np.abs(real_vol-res))
        iters_arr.append(10**i)
    
    plt.figure()
    plt.title('Error of spherical vol corresponds to number of points')
    plt.plot(iters_arr,errs,'k*')
    plt.xlabel('Number of points')
    plt.ylabel('Error')
    plt.xscale('log')
    plt.savefig('./tmp.jpg')
    plt.show()
    
    # spherical moment of inertia
    real_moment_of_inertia = 1.675516 # 2.0/5.0*4.188790*1.0**2
    errs = []
    iters_arr = []
    kernel = monte_carlo_kernel_sphere_intertia
    for  i in range(1,6):
        iters = 10**i
        res = monte_carlo(TPB,gridDim,iters,kernel)
        print(f'sphrical moment of inertia = {res}')
        errs.append(np.abs(real_moment_of_inertia-res))
        iters_arr.append(10**i)
    
    plt.figure()
    plt.title('Error of spherical moment of inertia corresponds to number of points')
    plt.plot(iters_arr,errs,'k*')
    plt.xlabel('Number of points')
    plt.ylabel('Error')
    plt.xscale('log')
    plt.savefig('./tmp_MOI.jpg')
    plt.show()
    
def p3b():
    # spherical shell volume
    r = 1
    real_vol = 4*np.pi*r*r
    errs = []
    iters_arr = []
    n = 1000
    TPB = 32
    gridDim = (n+TPB-1)//TPB
    blockDim = TPB
    kernel = monte_carlo_kernel_shell_vol
    for  i in range(1,6):
        iters = 10**i
        res = monte_carlo(TPB,gridDim,iters,kernel)
        print(f'sphrical shell vol = {res}')
        errs.append(np.abs(real_vol-res))
        iters_arr.append(10**i)
    
    plt.figure()
    plt.title('Error of spherical shell vol corresponds to number of points')
    plt.plot(iters_arr,errs,'k*')
    plt.xlabel('Number of points')
    plt.ylabel('Error')
    plt.xscale('log')
    plt.savefig('./tmp.jpg')
    plt.show()
    
    # spherical shell moment of inertia
    real_moment_of_inertia = 2.0/3.0*4.188790 # 2/3*M*r**2
    errs = []
    iters_arr = []
    kernel = monte_carlo_kernel_shell_intertia
    for  i in range(1,6):
        iters = 10**i
        res = monte_carlo(TPB,gridDim,iters,kernel)
        print(f'sphrical shell moment of inertia = {res}')
        errs.append(np.abs(real_moment_of_inertia-res))
        iters_arr.append(10**i)
    
    plt.figure()
    plt.title('Error of spherical shell moment of inertia corresponds to number of points')
    plt.plot(iters_arr,errs,'k*')
    plt.xlabel('Number of points')
    plt.ylabel('Error')
    plt.xscale('log')
    plt.savefig('./tmp_MOI.jpg')
    plt.show()
    
    

if __name__ == '__main__':
    
    #p1() : Done
    # p1 (a) : t2 = 0.04052 seconds for 151*151 grids
    # p1 (b) : already show plot
    # p1 (c) : delta T max = 0.00001 s
    
    
    #p2()
    #p2withRichar()
    
    
    #p3a()
    #p3b()
    #p3a_with_monte()
    # p3a_serial()  # Done without curve fitting
    p3b()
    