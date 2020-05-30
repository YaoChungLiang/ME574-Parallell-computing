import numpy as np

def p3a():
    realVol = 4.0/3.0*np.pi
    r = 1
    unitBlock = (2*r)**3
    N = 100000
    counter = 0
    for _ in range(N):
        sample = np.random.uniform(low=-1.0, high=1.0, size=(3,))
        if sample[0]**2+sample[1]**2 + sample[2]**2 <=1:
            counter += 1 
    print("Real Sphere vol : %f" %realVol)
    print("Monte Sphere vol: %f" % (unitBlock*counter/float(N)))
    
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

    print("Square Area : %f" % unitSquare)
    print("circle Area : %f" % (unitSquare*counter/float(N)) )
    print("Explicit unit cir Area : %f" % realCircleArea)
    
def p3a_Intertia():
    # sphere moment of inertia :2 / 5* M * r ^ 2,   ref :  https://www.youtube.com/watch?v=fbD5txXPWPw    
    # cube MOI :  1/6 * M * (2r)^2,   ref: https://www.youtube.com/watch?v=CqPjWjhURDw
    print(f'explicit sphere MOI = {2.0/5.0*4.188790}')
    #  1/12 * m * (l1^2 + l2^2)
    print(f'explicit cube MOI = {1/12.0 * (2**3) * (2**2+2**2)}' )
    print((2.0/5.0*4.188790)/(1/6.0*8*2**2)  )
if __name__ == "__main__":
    p3a()
    #monteSample()
    p3a_Intertia()