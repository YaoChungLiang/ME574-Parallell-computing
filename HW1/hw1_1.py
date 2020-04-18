import numpy as np
import math
import matplotlib.pyplot as plt
from time import time

def dn_sin(n):
    '''
    if n%2:
        if(n/2)%2:
            return -np.sin(0)
        else:
            return np.sin(0)
    else:
        if ((n+1)/2)%2:
            return np.cos(0)
        else:
            return -np.cos(0)

    '''
    if n%2:
        return (-1)**((((n-1)/2)%2))*np.cos(0)
    else:
        return (-1)**((n/2)%2)*np.sin(0)
    

def factorial(n):
    if n < 0:
        raise ValueError
    if n == 1 or n == 0:
        return 1
    res = 1
    for i in range(1,n+1):
        res *= i
    return res


def helper(x,n):
    if n <= 0 :
        return x
    if x == 0:
        return 0
    return helper(x, n-1)+(-1)**n*x**(2*n+1)/np.math.factorial(2*n+1)

def taylor_sin(x, n):
    if n < 2:
        raise ValueError
    n = int(n/2)-1
    return helper(x,n)


def measure_diff(ary1,ary2):
    #res = abs(max(ary1)-max(ary2))
    res = np.linalg.norm(np.array(ary1)-np.array(ary2))
    return res


def create2Dmapping(w,h,Real_L, Real_H, Img_L, Img_H, max_iters, upper_bound):
    Real_val = np.linspace(Real_L, Real_H, w)
    Img_val = np.linspace(Img_L, Img_H, h)
    graph = np.ones((h,w), dtype = np.float32)
    graph_iter = 256*np.ones((h,w), dtype = np.float32)
    for x in range(w):
        for y in range(h):
            c = np.complex64(Real_val[x] + Img_val[y]*1j)
            z = np.complex64(0)
            for i in range(max_iters):
                z = z**2 + c
                if np.abs(z) > upper_bound:
                    graph[y,x] = 0
                    graph_iter[y,x] = i
                    break
    return graph, graph_iter

def save_graph(graph):
    fig = plt.figure()
    plt.imshow(graph,extent=(-2,2,-2,2))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.savefig('binarymap.png',dpi=fig.dpi)

def mandelbrot(cx,cy,dist,itrs):
    w = len(cx)
    h = len(cy)
    graph_iter = 256*np.ones((h,w),dtype = np.float32)
    graph = np.ones((h,w), dtype= np.float32)
    for x in range(w):
        for y in range(h):
            res  = escape(cx[x],cy[y],dist,itrs)
            graph_iter[y,x] = res
            if res >= 256:
                graph[y,x] = 0
    save_graph(graph)
    return graph_iter
    
def escape(cx, cy, dist, itrs, x0=0, y0=0):
    c = np.complex64(cx + cy*1j) 
    z = np.complex64(x0 + y0*1j)
    for i in range(itrs):
        z = z**2+c
        if np.abs(z) > dist:
            return i
    return itrs



def problem5_new():
    w = 512
    h = 512
    cx = np.linspace(-2,2,w)
    cy = np.linspace(-2,2,h)
    dist = 2.5
    itrs = 256
    t = time()
    graph_iter = mandelbrot(cx,cy,dist,itrs)
    graph_time = time()-t
    print('total constructing graph time = {} seconds'.format(graph_time))
    fig = plt.figure()
    plt.imshow(graph_iter,extent=(-2,2,-2,2))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.savefig('itrsmap.png', dpi=fig.dpi)


def problem5():
    t = time()
    graph, graph_iter = create2Dmapping(512,512,-2,2,-2,2,256,2.5)
    graph_time= time()-t
    fig= plt.figure()
    plt.imshow(graph, extent = (-2,2,-2,2))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.savefig('binarymap.png', dpi=fig.dpi)
    

    fig = plt.figure()
    plt.imshow(graph_iter, extent=(-2,2,-2,2))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.savefig('iter.png', dpi = fig.dpi)
    
    

if __name__ == '__main__':
    # Problem 2/3
    plt.figure()
    start_order = 2
    end_order = 16
    order_num = (end_order - start_order)/2 +1
    x = np.linspace( 0, np.pi/4, 50, endpoint = True)
    for n in range(start_order,end_order,2):
        temp = [0]*len(x)
        for i in range(len(x)):
            temp[i] = taylor_sin(x[i],n)
        plt.plot(x,temp, label = 'order of {}'.format(n))
    
    temp = [0]*len(x)
    for i in range(len(x)):
        temp[i] = np.sin(x[i])
    plt.plot(x, temp, label= 'sin function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    # Problem 4 - a 
    plt.figure()
    sin_array = temp
    diff_array = []
    for n in range(start_order, end_order, 2):
        truncated = [0]*len(x)
        for i in range(len(x)):
            truncated[i]= taylor_sin(x[i],n)
        diff = measure_diff(truncated,sin_array)
        diff_array.append(diff)
    order = np.linspace(start_order,end_order,order_num-1 ,endpoint = False)
    plt.plot(order, diff_array)
    plt.xlabel('order')
    plt.ylabel('infinity norm error')
    plt.title('')
    plt.show()
    # Problem 4 - b
    
    for i in range(len(diff_array)):
        if diff_array[i] < 1e-2:
            print('order of {} is enough for truncated error under 1e-2'.format(order[i]))
            break
    # Problem 5
    problem5_new()
