import numpy as np

def p3a():
    realVol = 4.0/3.0*np.pi
    
    N = 10000
    counter = 0
    for _ in range(N):
        sample = np.random.uniform(low=-1.0, high=1.0, size=(3,))
        if sample[0]**2+sample[1]**2 + sample[2]**2 <=1:
            counter += 1
    
def monteSample():
    r = 1
    unitSquare = (2*r)**2
    realCircleArea = np.pi*r**2
    N = 10000
    counter = 0
    for _ in range(N):
        sample = np.random.uniform(low=-1.0, high=1.0, size=(2,))
        if sample[0]**2 + sample[1]**2 <= 1:
            counter += 1

    print("Square Area %f" % unitSquare)
    print("circle Area : %f" % (unitSquare*counter/float(N)) )
    print("Explicit unit cir Area : %f" % realCircleArea)
    
if __name__ == "__main__":
    #p3a()
    monteSample()