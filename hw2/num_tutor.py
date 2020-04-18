from numba import jit
import random
from time import time

@jit(nopython=True)
def a(n):
    acc = 0
    for i in range(n):
        x = random.random()
        y = random.random()
        if (x**2+y**2)<1.0:
            acc += 1
    return 4.0*acc/n

if __name__ == '__main__':
    st = time()
    a(100)
    print(time()-st)
