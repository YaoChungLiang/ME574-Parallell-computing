import sympy
import numpy as np

def simpson(f,a,b,n):
    h = (b-a)/n
    k = 0.0
    x = a + h
    for i in range(1,n/2+1):
        k += 4*f(x)
        x += 2*h
    x = a + 2*h
    for i in range(1,n/2):
        k += 2*f(x)
        x += 2*h
    return (h/3)*(f(a) + f(b) + k)


def MonteCarlo_double(f,g,x0,x1,y0,y1,n):

    np.random.seed(0)
    x = np.random.uniform(x0,x1,n)
    y = np.random.uniform(y0,y1,n)
    f_mean = 0
    num_inside = 0
    for i in range(len(x)):
        for j in range(len(y)):
            if g(x[i], y[j]) >= 0:
                num_inside += 1
                f_mean += f(x[i], y[j])
    f_mean = f_mean/num_inside
    area = num_inside/(n**2)*(x1-x0)*(y1-y0)
    return area*f_mean

def g(x,y):
    return 1 if (0 <= x <= 2 and 3 <= y <= 4.5) else -1

def f(x,y):
    return 1

def midpoint_triple1(g,a,b,c,d,e,f,nx,ny,nz):
    hx = (b-a)/nx
    hy = (d-c)/ny
    hz = (f-e)/nz
    I = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                xi = a + hx/2 + i*hx
                yj = c + hy/2 + j*hy
                zk = e + hz/2 + k*hz
                I += hx*hy*hz*g(xi,yj,zk)
    return I

def midpoint(f,a,b,n):
    h = (b-a)/n
    f_sum = 0
    for i in range(0,n,1):
        x = (a+h/2.0) + i*h
        f_sum += f(x)
    return h*f_sum

def midpoint_triple2(g,a,b,c,d,e,f,nx,ny,nz):
    def p(x,y):
        return midpoint(lambda z : g(x,y,z), e,f,nz)
    def q(x):
        return midpoint(lambda y:p(x,y), c, d, ny)
    return midpoint(q, a, b, nx)


def test_midpoint_triple():
    def g(x,y,z):
        return 2*x+y-4*z
    a = 0
    b = 2
    c = 2
    d = 3
    e = -1
    f = 2
    
    x,y,z= sympy.symbols('x y z')
    I_expect = sympy.integrate(
        g(x,y,z), (x,a,b), (y,c,d), (z,e,f)
    )
    for nx,ny,nz in (3,5,2),(4,4,4),(5,3,6):
        I_computed1 = midpoint_triple1(
            g,a,b,c,d,e,f,nx,ny,nz
        )
        I_computed2 = midpoint_triple2(
            g,a,b,c,d,e,f,nx,ny,nz
        )
        tol = 1e-14
        print(I_expect, I_computed1,I_computed2)
        assert abs(I_computed1-I_expect) < tol
        assert abs(I_computed2-I_expect) < tol


def MC():
    x0 = 0
    x1 = 3
    y0 = 2
    y1 = 5
    n = 2000
    simulate = MonteCarlo_double(f,g,x0,x1,y0,y1,n)
    print("MC simulation %f" % simulate)
if __name__ == "__main__":
    test_midpoint_triple()
    MC()

