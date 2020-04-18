import numpy as np
import math
import matplotlib.pyplot as plt
from time import time

def dn_sin(n):
  if n%2:
    return (-1)**(((n-1)/2%2))*np.cos(0)
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

def taylor_sin(x, n):
    res = x
    for i in range(n):
        if i%2 == 1:
            res += (dn_sin(i)/float(factorial(n)))*((x-0)**i)
    return res


def measure_diff(ary1,ary2):
    '''
    res = 0
    for i in range(len(ary1)):
        res += np.abs(ary1[i]- ary2[i])
    '''
    res = abs(max(ary1)-max(ary2))
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
    end_order = 20
    order_num = (end_order - start_order)/2 +1
    x = np.linspace(0, np.pi/4.0, 50, endpoint = True)
    for n in range(start_order,end_order,2):
        temp = [0]*len(x)
        for i in range(len(x)):
            temp[i] = taylor_sin(x[i],n)
        plt.plot(x,temp, label = 'order of {}'.format(n-1))
    
    temp = [0]*len(x)
    for i in range(len(x)):
        temp[i] = np.sin(x[i])
    plt.plot(x,temp, label= 'sin function')
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
    order = np.linspace(start_order,end_order,order_num-1 ,endpoint = True)
    plt.plot(order, diff_array)
    plt.xlabel('order')
    plt.ylabel('L1 norm error')
    plt.title('')
    plt.show()
    # Problem 4 - b
    # No matter how the truncation order be, the error cannot be under 1e-2
    
    # Problem 5
    problem5()
