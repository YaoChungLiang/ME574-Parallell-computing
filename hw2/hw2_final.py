import numba
from numba import cuda
import numpy as np
import math
import matplotlib.pyplot as plt
from time import time
devices = cuda.list_devices()

def gpu_total_memory():
    gpus = cuda.gpus.lst
    for gpu in gpus:
        with gpu:
            meminfo = cuda.current_context().get_memory_info()
            return int(meminfo[1])
            #print("%s, free: %s bytes, total, %s bytes" % (gpu, meminfo[0], meminfo[1]))
def gpu_compute_capability():
    return devices[0].compute_capability
def gpu_name():
    return devices[0].name
    
def max_float64s():
    # 64 bits = 8 bytes
    # total number of floats can be stored is : GPU total bytes//8
    return gpu_total_memory()//(64/8)

def map_64():
    N = 100000000
    from map_parallel import sArray
    x = np.linspace(0,1,N,dtype = np.float64)
    f = sArray(x)
    '''
    plt.plot(x,f,'o',label='1-2*sin(2*PI*x))**2')
    plt.legend()
    plt.show()
    '''
def sf(x,r):
    return r*x*(1-x)

@cuda.jit(device = True)
def f(x,r):
    return r*x*(1-x)

@cuda.jit
def logistic_map_kernel(ss,r,x,transient,steady):
    i = cuda.grid(1)
    if i < r.size:
        x_old = x
        for j in range(transient):
            x_new = f(x_old,r[i])
            x_old = x_new
        for j in range(steady):
            ss[j,i] = x_old
            x_new = f(x_old,r[i])
            x_old = x_new


def parallel_logistic_map(r,x,transient,steady):
    n = r.size
    
    d_r = cuda.to_device(r)
    d_2Dres = cuda.device_array((n,steady), dtype=np.float64)
    
    TPBX = 32
    gridDims = (n + TPBX - 1)//TPBX
    blockDims = TPBX

    logistic_map_kernel[gridDims, blockDims](d_2Dres, d_r ,x ,transient,steady)
    
    return d_2Dres.copy_to_host()
    
def problem3d():
    n_ss= 8
    n_transient = 100
    x0 = 0.5
    rmin = 2.5
    rmax = 4
    m = 1000
    itrs = 1000
    r = np.linspace(rmin,rmax,m)
    para_st = time()
    for _ in range(itrs):
        res = parallel_logistic_map(r,x0,n_transient,n_ss)
    para_time = time()-para_st
    print('parallel time = {}'.format(para_time))

    seri_st = time()
    for _ in range(itrs):
        a = serial_logistic(r,n_ss,n_transient)
    seri_time = time() - seri_st
    print('serial time = {}'.format(seri_time))
    print('accelaration factor is {}'.format(seri_time/para_time))
    
    '''
    fig = plt.figure()
    plt.plot(r,res)
    plt.xlabel('r value' )
    plt.ylabel('x value')
    plt.savefig('logistic_parallel.png')
    '''
    return res

def serial_f(x,r):
    return r*x*(1-x)

def logisticSteadyArray(x0,r,n_transient, n_ss):
    x = np.zeros(n_ss, dtype=np.float64) 
    x_old = x0
    for i in range(n_transient):
        x_new = serial_f(x_old, r)
        x_old = x_new
    for i in range(n_ss): 
        x[i] = x_old
        x_new = serial_f(x_old, r)
        x_old = x_new 
    return x
def serial_logistic(r,n_ss,n_transient):
    x = np.zeros([r.size,n_ss])
    x0 = 0.5
    for j in range(r.shape[0]):
        tmp = logisticSteadyArray(x0, r[j], n_transient, n_ss) 
        for i in range(n_ss): 
            x[j,i] = tmp[i] 
    return x


@cuda.jit(device = True)
def iteration_count(cx,cy,dist,itrs):
    zReal = 0.0
    zImg = 0.0
    i, j = cuda.grid(2)
    for idx in range(itrs):
        tmp_z_Real = zReal**2 - zImg**2 + cx
        zImg = zReal*zImg*2 + cy
        zReal = tmp_z_Real
        if zReal**2 + zImg**2 > dist:
            return idx
    return itrs


@cuda.jit
def mandelbrot_kernel(d_out,d_cx,d_cy,dist,itrs):
    i , j = cuda.grid(2)
    nx , ny = d_out.shape
    if i < nx and j < ny:
        d_out[i,j] = iteration_count(d_cx[i], d_cy[j],dist,itrs)


def parallel_mandelbrot(cx,cy,dist,itrs):
    nx = cx.size
    ny = cy.size
    d_cx = cuda.to_device(cx)
    d_cy = cuda.to_device(cy)
    d_ManRes = cuda.device_array((nx,ny), dtype = np.float64)
    n = 32
    TPBX = n
    TPBY = n
    gridDims = ((nx + TPBX - 1)//TPBX ,(ny + TPBY - 1)// TPBY )
    blockDims = (TPBX , TPBY )
    mandelbrot_kernel[ gridDims , blockDims ](d_ManRes, d_cx, d_cy, dist,itrs)
    return d_ManRes.copy_to_host()

def escape(cx,cy,dist,itrs,x0=0,y0=0):
    c = np.complex64(cx+cy*1j)
    z = np.complex64(x0+y0*1j)
    for i in range(itrs):
        z = z**2+c
        if np.abs(z)>dist:
            return i
    return itrs

def serial_mandelbrot(cx,cy,dist,itrs):
    w = len(cx)
    h = len(cy)
    graph_iter = 256*np.ones((h,w), dtype=np.float64)
    for x in range(w):
        for y in range(h):
            res = escape(cx[x],cy[y],dist, itrs)
            graph_iter[y,x] = res
    return graph_iter.transpose()

def problem4(i=1):
    w = 2**i
    h = 2**i
    if i == 1:
        w = 512
        h = 512
    cx = np.linspace(-2,2,w,dtype=np.float64)
    cy = np.linspace(-2,2,h,dtype=np.float64)
    dist = 2.5
    itrs = 256
    res_parallel = parallel_mandelbrot(cx,cy,dist,itrs)
    print('resolution of {} in 4 grids works '.format(w))
    #res_serial = serial_mandelbrot(cx,cy,dist,itrs)
    '''
    fig = plt.figure()
    plt.imshow(res_parallel)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('mandel_parallel.png')
    plt.show()
    
    fig = plt.figure()
    plt.imshow(res_serial)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('mandel_serial.png')
    plt.show()
    '''
if __name__ == "__main__":
    
    # Problem 1 - a 
    print("GPU memory in GB: ", gpu_total_memory()/1024**3)
    print("Compute capability (Major, Minor): ",gpu_compute_capability())
    print("GPU Model Name: ", gpu_name())
    ''' 
    # 1 - a
    result:
        GPU memory in GB: 3.9476318359375
        Compute capability (Major,Minor): (6,1)
        GPU Model Name: b'GeForce GTX 1050'
    '''
    # 1 - b
    print('Maximum number of 64-bit floats is {}'.format(max_float64s()))
    # Maximum number of 64-bit floats is 529842176.0

    # Problem 2
    map_64()
    # 2 - a : 10^(8) 64-bit floats can be ran successfully
    # 2 - b : Error message is below when array size is to large
    # numba.cuda.cudadrv.driver.CudaAPIError: [2] Call to cuMemAlloc results in CUDA_ERROR_OUT_OF_MEMORY

    # Problem 3
    '''
    # 3 - a : r and x is been looping through 
    logistic map code is shown below:
    for j in range(r.shape[0]):
        tmp = logisticSteadyArray(x0, r[j], n_transient, n_ss)
        for i in range(n_ss):
            x[j,i] = tmp[i]
    
    # 3 - b: in the for j in range(r.shape[0]) loop, the every calaulation is independent of each other
            because the r is only treated as a independent paramater in each iteration
    '''
    problem3d()
    '''
    # 3 - c : parallized code as shown above
    # 3 - d : acceleration factor = 109.8856/1.61636 = 68
    '''

    # Problem 4
    # 
    problem4()
    # 4 - a : in serial code, first for loop iterates the Cx while the second for loop iterates the Cy
    #         these Cx and Cy are independant
    #         but in the escape iteration loop, every z is  dependent on previous z, 
    #         which makes it impossible to be parallelized 
    
    # 4 - b : result is the same in 'mandel_parallel.png' and 'mandel_serial.png'
    for i in range(1,14):
        try:
            problem4(i)
        except Exception as e:
            print(e)

    # 4 - c:
    # the finest grid I can run on is 1/4096 = 2^(-12).
    # after this situation, Error messages pop up
    #     Call to cuMemAlloc, results in CUDA_ERROR_OUT_OF_MEMORY
    # when stepsize gets even smaller, 
    #     Unable to allocate 38.1 Gib for an array with shape......

    # 4 - d
    # the largest square blocks I can get is 32*32
    # if I ask for more, error message is shown below
    # numba.cuda.cudadrv.driver.CudaAPIError: [1] Call to cuLaunchKernel results in CUDA_ERROR_INVALID_VALUE
