from math import sin, sinh

import matplotlib.pyplot as plt
import numpy as np
from numba import cuda, float32, int32, jit

t_steps = 300
timeRange = np.linspace(0, 3.0, t_steps, endpoint=True)
dt = timeRange[1] - timeRange[0]

NY, NX = 21, 21
PI = np.pi
TPB = 8
RAD = 1
SH_N = 10


@cuda.reduce
def max_kernel(a, b):
    return max(a, b)

@cuda.reduce
def sum_kernel(a, b):
    return a + b



@cuda.jit
def heat_step(d_u, d_out, stencil, t):
	h=0.01
	edge,corner = 0.25, 0.0
	i, j = cuda.grid(2)
	dims = d_u.shape
	if i >= dims[0] or j >= dims[1]:
		return
	NX,NY = cuda.blockDim.x, cuda.blockDim.y
	t_i, t_j = cuda.threadIdx.x , cuda.threadIdx.y
	sh_i, sh_j = t_i + RAD, t_j + RAD

	sh_u = cuda.shared.array(shape=(SH_N,SH_N) , dtype = float32)

	sh_u[sh_i, sh_j] = d_u[i,j]
    
    # Halo edge values assignment
	if t_i < RAD:
		sh_u[sh_i - RAD, sh_j] = d_u[i - RAD , j]
		sh_u[sh_i + NX , sh_j] = d_u[i + NX  , j]

	if t_j < RAD:
		sh_u[sh_i , sh_j - RAD] = d_u[i, j - RAD]
		sh_u[sh_i , sh_j + NY ] = d_u[i, j + NY ]

    # Halo corner values assignment
	if t_i < RAD and t_j < RAD:
        # upper left
		sh_u[sh_i - RAD, sh_j - RAD] = d_u[i - RAD, j - RAD]
        # upper right
		sh_u[sh_i + NX, sh_j - RAD] = d_u[i + NX, j - RAD]
        # lower left
		sh_u[sh_i - RAD, sh_j + NY] = d_u[i -RAD, j + NY]
        # lower right
		sh_u[sh_i + NX, sh_j + NY] = d_u[i + NX, j + NY]
    
	cuda.syncthreads()

	if i > 0 and j > 0 and i < dims[0] -1 and j < dims[1] -1 :
		d_out[i, j] = \
                    (sh_u[sh_i,sh_j-1]*edge +\
                    sh_u[sh_i,sh_j+1]*edge +\
                    sh_u[sh_i-1,sh_j]*edge +\
                    sh_u[sh_i+1,sh_j]*edge -\
                    sh_u[sh_i,sh_j] * -1.0)*h*h*dt +\
					sh_u[sh_i,sh_j]


def p1():
	# initialization
	u = np.zeros(shape = [NX,NY], dtype = np.float32)
    # initial condition
	stencil = 0.25
	for i in range(NX):
		for j in range(NY):
			u[i][j] = sin(2*PI*i)*sin(PI*j)
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
	gridSize= [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB]
	blockSize = [TPB,TPB]	

	for i in range(len(timeRange)):
		heat_step[gridSize, blockSize](d_u, d_out, stencil, timeRange[i])
		heat_step[gridSize, blockSize](d_out, d_u, stencil, timeRange[i])
	
	xvals = np.linspace(0.,1.0,NX)
	yvals = np.linspace(0.,1.0,NY)
	X,Y = np.meshgrid(xvals, yvals)
	levels = [0.025,0.1,0.25,0.5,0.75]
    #plt.contourf(X, Y,exact.T, levels = levels)
	plt.contour(X, Y, u.T, levels= levels, colors = "r", linewidths = 4)
	plt.axis([0,1,0,1])
	plt.show()
        


@cuda.jit
def integrate_kernel(y, out, quad):
    '''
    y: input device array
    out: output device array
    quad: quadrature stencil coefficients
    '''
    pass


def integrate(y, quad):
    '''
    y: input array
    quad: quadrature stencil coefficients
    '''
    pass

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

# @cuda.jit


def monte_carlo_kernel_shell_intertia(rng_states, iters, out):
    pass

# @cuda.jit


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

@cuda.jit(device = True)
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

if __name__ == '__main__':
	p1()
