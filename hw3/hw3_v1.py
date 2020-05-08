import numba
from numba import cuda, int32, float32
import time
import numpy as np
import matplotlib.pyplot as plt


pi_string = "3141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117450284102701938521105559644622948954930381964428810975665933446128475648233786783165271201909145648566923460348610454326648213393607260249141273724587006606315588174881520920962829254091715364367892590360011330530548820466521384146951941511609433057270365759591953092186117381932611793105118548074462379962749567351885752724891227938183011949129833673362440656643086021394946395224737190702179860943702770539217176293176752384674818467669405132000568127145263560827785771342757789609"
pi_digits = [int(char) for char in pi_string]
v = 0.1*np.array(pi_digits)[0:374]


@cuda.jit
def ewprod_kernel(d_res ,d_u,d_v):
    i = cuda.grid(1)
    if i < d_u.size:
        d_res[i] = d_u[i]*d_v[i]
    if cuda.threadIdx.x % 32 == 0:
        #print('(threadIdx,blockIdx) = ({},{})'.format(cuda.threadIdx.x,cuda.blockIdx.x))
        #print('threadIdx')
        #print(cuda.threadIdx.x)
        #print('blockIdx')
        #print(cuda.blockIdx.x)
        print(cuda.blockIdx.x, cuda.threadIdx.x)
        

def ewprod(u,v):
    n = u.size
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(v)
    d_res = cuda.device_array(n, dtype = np.float64)

    TPBx = 32
    gridDim = (n+TPBx-1)//TPBx
    blockDim = TPBx

    ewprod_kernel[gridDim, blockDim](d_res, d_u, d_v)
    
    return d_res.copy_to_host()

def ewprod_1c(u,v):
    n = u.size
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(v)
    d_res = cuda.device_array(n, dtype = np.float64)
    TPBx = 96
    gridDim = 4
    blockDim = TPBx
    
    ewprod_kernel[gridDim, blockDim](d_res, d_u, d_v)

    return d_res.copy_to_host()
def smooth_serial(v,rad):
    w = [0]*len(v)
    for i in range(len(v)):
        counter = 0
        for j in range(-rad,rad+1):
            if (i+j)> -1 and (i+j)<len(v):
                counter += 1
                w[i] += v[i+j]
        w[i] = w[i]//counter
    return w

@cuda.jit
def smooth_kernel(d_out, d_v, rad):
    i = cuda.grid(1)
    if i < d_v.size:
        counter = 0
        for j in range(-rad,rad):
            if i+j > -1 and (i+j)<len(v):
                d_out[i] += d_v[i+j]
                counter += 1
        d_out[i] = d_out[i]/counter

def smooth_parallel(v,rad):
    n = len(v)
    d_v = cuda.to_device(v)
    d_out = cuda.device_array(n, dtype = np.float64)
    TPBx = 32
    gridDim = (n+TPBx-1)//TPBx
    blockDim = TPBx
    
    st = cuda.event()
    end = cuda.event()
    st.record()
    smooth_kernel[gridDim, blockDim](d_out, d_v, rad)
    end.record()
    end.synchronize()
    elapsed = cuda.event_elapsed_time(st, end)
    print('elapsed time = {}'.format(elapsed))

    return d_out.copy_to_host(), elapsed

@cuda.jit
def smooth_sm_kernel(d_out,d_arr, rad):
    global NSHARED_2c
    n = d_arr.shape[0]
    i = cuda.grid(1)
    sh_arr = cuda.shared.array(NSHARED_2c,dtype = float32)
    
    t_Idx = cuda.threadIdx.x # thread index
    sh_Idx = t_Idx + rad

    if i>=n:
        return
    sh_arr[sh_Idx] = d_arr[t_Idx]
    
    if t_Idx < rad:
        if i >= rad:
            sh_arr[sh_Idx-rad] = d_arr[i-rad]
        if i+cuda.blockDim.x < n:
            sh_arr[sh_Idx + cuda.blockDim.x] = d_arr[i + cuda.blockDim.x]
    
    cuda.syncthreads()
    
    if i >= rad and i < n-rad:
        tmp = 0
        for d in range(-rad,rad+1):
            tmp += sh_arr[sh_Idx+d]
        d_out[i] = tmp/(2*rad+1)


def smooth_parallel_sm(v,rad):
    n = len(v)
    d_v = cuda.to_device(v)
    d_out = cuda.device_array(n+2*rad, dtype = np.float64)

    TPBx = 32
    gridDim =  (n+TPBx-1)//TPBx
    blockDim = TPBx

    st = cuda.event()
    end = cuda.event()
    st.record()
    smooth_sm_kernel[gridDim, blockDim](d_out, d_v, rad)
    end.record()
    end.synchronize()
    elapsed = cuda.event_elapsed_time(st, end) 
    print('elapsed time = {}'.format(elapsed))
    return d_out.copy_to_host(), elapsed

def rhs(y,t):
    return np.array([y[1],-y[0]])

def rk4_step(f,y,t0,h):
    k1 = h*f(y,t0)
    k2 = h*f(y+k1/2.0, t0+h/2.0)
    k3 = h*f(y+k2/2.0, t0+h/2.0)
    k4 = h*f(y+k3, t0+h)
    y_new = y + 1.0/6.0*(k1+2*k2+2*k3+k4)

    return y_new


def rk_solve(f, y0, t):
    n = t.size
    y = [y0]
    h = t[1]-t[0]
    for i in range(n-1):
        y_new = rk4_step(f,y[i],t[i],h)
        y.append(y_new)
    return np.array(y)

def omega_rhs(y,t,w):
    return np.array([y[1],-w*w*y[0]])

@cuda.jit(device = True)
def rk4_step_parallel(f,y,t0,h,w):
    k1 = h*f(y,t0, w)
    k2 = h*f(y+k1/2.0, t0+h/2.0,w)
    k3 = h*f(y+k2/2.0, t0+h/2.0,w)
    k4 = h*f(y+k3, t0+h,w)
    y_new = y + 1.0/6.0*(k1+2*k2+2*k3+k4)
    return y_new
    

@cuda.jit
def rk4_solve_parallel(f,d_y0,d_t,d_w):
    n = t.size
    h = d_t[1]-d_t[0]
    i = cuda.grid(1)
    c = 1
    if i < d_w.size:
        for j in range(len(d_t)):
            d_y0 = rk4_step_parallel(f, d_y0, d_t[j], h, d_w[i])
        res = d_y0[1]-c*d_y0[0]
        d_w[i] = res

def rk4_parallel(f,y0,t,w):
    n = y0.size
    d_y0 = cuda.to_device(y0)
    d_w = cuda.to_device(w) 
    d_t = cuda.to_device(t)
    
    TPBX = 32
    gridDims= (n+TPBX-1)//TPBX
    blockDims = TPBX
    

    rk4_solve_parallel[gridDims,blockDims](f,d_y0,d_t,d_w)
    return dw.copy_to_host()
# optional
def eigenvals_parallel(NC,NW,NT,h):
    pass

def jac_f(x,y):
    return x**4+y**4-1

def jacobi_update_serial(u,f,x,y):
    nx = len(x)
    ny = len(y)
    u_new = np.copy(u)
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            #u_new[i,j] = (u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1])/4.
            u_new[i,j] = f(x[i],y[j])
    return u_new
    
@cuda.jit
def jacobi_update_kernel(d_u,d_x,d_y):
    pass

if __name__ == "__main__":
    # problem 1
    # (a) paralleled version of ewprod finished
    # (b) 
    
    u = 1-v
    '''
    ewprod_para = np.array(ewprod(u,v)))
    ewprod_np = np.multiply(u.v)
    print(np.array_equal(ewprod_para, ewprod_np))
    '''
    # (c)
    '''
    ewprod_1c(u,v)
    '''
    # result:
    # test1 : thread 64 64 64  0  0 32  0 64 32 32  0 32
    # test1 : block   0  2  3  0  2  0  2  3  1  3  1  1
    # test2 : thread 64  0 32 64  0 32 64  0 32 64  0 32
    # test2 : block   0  2  2  0  0  1  3  3  3  1  1  2
    # test3 : thread  0 64 32 64  0  0 32 32 64  0 64 32
    # test3 : block   0  0  0  2  2  3  3  2  3  1  1  1
    
    # conclusion:
    # The order of block execution is not predictable.
    # The order of execution within a block is not predictable

    # problem 2
    # (a)
    
    w = np.outer(v,1-v).flatten()
    '''
    rad = 2
    st_r2 = time.time()
    res_r2 = smooth_serial(w,rad)
    t_r2 = time.time()-st_r2

    rad = 4
    st_r4 = time.time()
    res_r4 = smooth_serial(w,rad)
    t_r4 = time.time()-st_r4

    plt.figure()
    plt.plot(res_r2, label =  'time of rad 2 = {}'.format(t_r2))
    plt.plot(res_r4, label =  'time of rad 4 = {}'.format(t_r4))
    plt.legend()
    plt.show()
    '''
    
    # (b)
    '''
    rad = 2
    res_pr2, eTime_r2 =  smooth_parallel(w,rad)
    res_pr2, eTime_r2 =  smooth_parallel(w,rad) 
    rad = 4
    res_pr4, eTime_r4 =  smooth_parallel(w,rad)

    plt.figure()
    plt.plot(res_pr2, label =  'time of rad 2 = {:2.4f} ms'.format(eTime_r2))
    plt.plot(res_pr4, label =  'time of rad 4 = {:2.4f} ms'.format(eTime_r4))
    plt.legend()
    plt.show()
    '''
    
    # (c)
    '''
    rad_2c = 2
    TPB_2c = 32
    NSHARED_2c = TPB_2c + 2*rad_2c
    res_sm_r2, sh_r2_time = smooth_parallel_sm(v,rad_2c)
    res_sm_r2, sh_r2_time = smooth_parallel_sm(v,rad_2c)
    rad_2c = 4
    NSHARED_2c = TPB_2c + 2*rad_2c
    res_sm_r4, sh_r4_time = smooth_parallel_sm(v,rad_2c)
    
    plt.figure()
    plt.plot(res_sm_r2, label =  'time of rad 2 = {:2.4f} ms'.format(sh_r2_time))
    plt.plot(res_sm_r4, label =  'time of rad 4 = {:2.4f} ms'.format(sh_r4_time))
    plt.legend()
    plt.show()
    '''
    
    # problem 3
    # 3-a
    '''
    steps_10 = 10
    y0 = np.array([0,1])
    t_10 = np.linspace(0,np.pi,steps_10+1)
    y_3a_t10 = rk_solve(rhs, y0, t_10)
    
    steps_100 = 100
    t_100 = np.linspace(0,np.pi,steps_100+1)
    y_3a_t100 = rk_solve(rhs, y0, t_100)
    
    exact = 0
    err_t10  = abs(y_3a_t10[-1,0] - exact)
    err_t100 = abs(y_3a_t100[-1,0]- exact)
    
    plt.figure()
    plt.plot(t_10,y_3a_t10[:,0], label = 'steps = {}, x = pi error = {:.2E}'.format(steps_10,err_t10))
    plt.plot(t_100,y_3a_t100[:,0], label = 'steps = {},x =pi err = {:.2E} '.format(steps_100,err_t100))
    plt.legend()
    plt.show()
    '''
    # 3-b
    '''
    steps_10 = 10
    w = np.array([0,1,2,3,4,5,6,7,8,9,10])
    y = np.array([0,1])
    t_10 = np.linspace(0,np.pi,steps_10+1)
    para_10 = rk4_parallel(omega_rhs, y, t_10, w )
    plt.figure()
    plt.plot(para_10)
    plt.show()
    '''
    # 3-c
    
    # 4
    # 4-a 
    NX,NY = 128,128
    iters = NY*NX
    u = np.zeros(shape=[NX,NY], dtype=np.float32)
    for i in range(NX):
        for j in range(NY):
            if i > 0:
                u[i][j] = 0.5
            if i < 0:
                u[i][j] = -1
    u[0,0] = 1

    xvals = np.linspace(0., 1.0, NX)
    yvals = np.linspace(0., 1.0, NY)
    u = jacobi_update_serial(u,jac_f,xvals,yvals)
    
    X,Y = np.meshgrid(xvals, yvals)
    levels = [0.025, 0.1, 0.25, 0.50, 0.75]
    plt.contourf(X,Y,u.T, levels = levels)
    plt.contour(X,Y,u.T, levels = levels,colors = 'r', linewidths = 4)
    plt.axis([0,1,0,1])
    plt.show()
     

