import numpy as np

def _iniCon(x,y):
    u = np.zeros([len(x),len(y)], dtype = np.float64)
    for i in range(len(x)):
        for j in range(len(y)):
            u[i][j] = np.sin(2*np.pi*x[i])*np.sin(np.pi*y[j])
    return u
def p1_serial_update(u,timeRange):
    nx,ny = u.shape
    times = np.linspace(0,timeRange,num= 101,endpoint=True)
    dt = times[1]-times[0]
    h = x[1]-x[0]

    for _ in times:
        for i in range(1,len(x)-1):
            for j in range(1,len(y)-1):
                u[i][j] += dt*(1/4.0*h*h) *(-4*u[i][j]+ u[i-1][j]+ u[i+1][j]+ u[i][j-1]+u[i][j+1])
    return u


@jit
def p1_parallel_kernel():
    pass


def p1_parallel(u,t):
    d_u = cuda.copy_to_device(u)
    d_res = 


if __name__ == "__main__":
    # Problem 1
    x = np.linspace(0, np.pi ,num = 101, endpoint = True)
    y = np.linspace(0, np.pi ,num = 101, endpoint = True)
    init_u = _iniCon(x,y)
    print(init_u)

    serial_res = p1_serial_update(init_u, 10)
    print(serial_res)
