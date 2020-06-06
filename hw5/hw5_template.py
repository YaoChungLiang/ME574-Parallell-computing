import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from numpy.fft import fftfreq, fft, ifft
import cupyx
##Import other packages here##


def cupy_fft():
    """
    Modify the code below so it calls the cupy fft function
    """

    pts = 1000
    L = 100
    w0 = 2.0 * np.pi/L
    n1, n2, n3 = 10.0, 20.0, 30.0
    a1, a2, a3 = 1., 2., 3.

    # create signal data with 3 frequency components
    x = np.linspace(0, L, pts)
    y1 = a1*np.cos(n1*w0*x)
    y2 = a2*np.sin(n2*w0*x)
    y3 = a3*np.sin(n3*w0*x)
    y = y1 + y2 + y3

    # create signal including only 2 components
    y12 = y1 + y2

    # analytic derivative of signal
    dy = w0*(-n1*a1*np.sin(n1*w0*x)
             + n2*a2*np.cos(n2*w0*x)
             + n3*a3*np.cos(n3*w0*x))

    # use fft.fftfreq to get frequency array corresponding to number of sample points
    freqs = cp.fft.fftfreq(pts)
    freqs = cp.asnumpy(freqs)
    # compute number of cycles and radians in sample window for each frequency
    nwaves = freqs*pts
    nwaves_2pi = w0*nwaves

    # compute the fft of the full signal
    fft_vals = cp.fft.fft(cp.asarray(y))
    fft_vals = cp.asnumpy(fft_vals)
    # mask the negative frequencies
    mask = freqs > 0
    # double count at positive frequencies
    fft_theo = 2.0 * np.abs(fft_vals/pts)
    # plot fft of signal
    plt.xlim((0, 50))
    plt.title('cupy_fft original wave in frequency domain')
    plt.xlabel('frequency')
    plt.ylabel('original amplitude')
    plt.plot(nwaves[mask], fft_theo[mask])
    plt.show()

    # create a copy of the original fft to be used for filtering
    fft_new = np.copy(fft_vals)
    # filter out y3 by setting corr. frequency component(s) to zero
    fft_new[np.abs(nwaves) == n3] = 0.
    # plot fft of filtered signal
    plt.xlim((0, 50))
    plt.title('cupy_fft original wave in frequency domain, zero out f3')
    plt.xlabel('cycles in window')
    plt.ylabel('filtered amplitude')
    plt.plot(nwaves[mask], 2.0*np.abs(fft_new[mask]/pts))
    plt.show()

    # invert the filtered fft with numpy.fft.ifft
    filt_data = np.real(cp.asnumpy(cp.fft.ifft(cp.asarray(fft_new))))
    #filt_data = cp.asnumpy(filt_data)
    # plot filtered data and compare with y12
    plt.title('cupy_fft original and filtered waves in time domain')
    plt.plot(x, y12, label='original signal')
    plt.plot(x, filt_data, label='filtered signal')
    plt.xlim((0, 50))
    plt.legend()
    plt.show()

    # multiply fft by 2*pi*sqrt(-1)*frequency to get fft of derivative
    dy_fft = 1.0j*nwaves_2pi*fft_vals
    # invert to reconstruct sampled values of derivative
    dy_recon = np.real(cp.asnumpy(cp.fft.ifft(cp.asarray(dy_fft))))
    # plot reconstructed derivative and compare with analuytical version
    plt.plot(x, dy, label='exact derivative')
    plt.plot(x, dy_recon, label='fft derivative')
    plt.title('cupy_fft derivative of original and filtered waves in time domain')
    plt.xlim((0, 50))
    plt.legend()
    plt.show()


def cupy_filter():
    """
    Implement code below to:
    Create noise consisting of an array of pts random values chosen from a uniform distribution over the interval [−3,3]
    Create a noisy signal by adding noise to the original signal: y_n = y + noise
    Compute and plot the frequency content of the noisy signal.
    Create and apply an appropriate filter to suppress noise in the frequency domain.
    Invert the filtered fft to obtain a "denoised signal".
    Plot and compare the original, noisy, and denoised signals.
    """
    pts = 1000
    L = 100
    w0 = 2.0 * np.pi/L
    n1, n2, n3 = 10.0, 20.0, 30.0
    a1, a2, a3 = 1., 2., 3.

    # create signal data with 3 frequency components
    x = np.linspace(0, L, pts)
    y1 = a1*np.cos(n1*w0*x)
    y2 = a2*np.sin(n2*w0*x)
    y3 = a3*np.sin(n3*w0*x)
    y = y1 + y2 + y3

    # add noise data
    noise = np.random.uniform(-3, 3, pts)
    y_n = y + noise

    # use fft.fftfreq to get frequency array corresponding to number of sample points
    freqs = cp.fft.fftfreq(pts)
    freqs = cp.asnumpy(freqs)

    # compute number of cycles and radians in sample window for each frequency
    nwaves = freqs*pts
    nwaves_2pi = w0*nwaves

    # compute the fft of the full signal
    fft_vals = cp.fft.fft(cp.asarray(y_n))
    fft_vals = cp.asnumpy(fft_vals)

    # filter out noise
    thres = 8 * np.mean(np.abs(fft_vals))
    fft_vals[np.abs(fft_vals) < thres] = 0

    # print(fft_vals)
    # plt.figure()
    # plt.plot(fft_vals)
    # plt.show()
    # raise ValueError

    # mask the negative frequencies
    mask = freqs > 0
    # double count at positive frequencies
    fft_theo = 2.0 * np.abs(fft_vals/pts)
    # plot fft of signal
    plt.xlim((0, 50))
    plt.xlabel('frequencies')
    plt.ylabel('original amplitude')
    plt.title('cupy_filter original wave in frequency domain')
    plt.plot(nwaves[mask], fft_theo[mask])
    plt.show()

    # invert the filtered fft with numpy.fft.ifft
    filt_data = np.real(cp.asnumpy(cp.fft.ifft(cp.asarray(fft_vals))))
    #filt_data = cp.asnumpy(filt_data)
    # plot filtered data and compare with y12
    plt.plot(x, y_n, label='original signal')
    plt.plot(x, filt_data, label='denoised signal')
    plt.title('cupy_filter original and filtered waves in time domain')
    plt.xlim((0, 50))
    plt.legend()
    plt.show()


def cupy_eig(mat):
    """
    Returns the eigen values of square matrix mat
    """
    w, v = cp.linalg.eigh(mat)
    return w


def cupy_J(n):
    """
    Constructs the nxn matrix J (as described in q2)
    Returns J, leading eigenvalue of J, and the associated eigenvector
    """
    jmatArr = cupyx.scipy.sparse.diags(
        [0.5, 1, 0.5], offsets=[1, 0, -1], shape=(n, n))
    jmat = cupyx.scipy.sparse.dia_matrix.toarray(jmatArr)
    w, v = cp.linalg.eigh(jmat)
    # print(w[-1])
    # print(v[:,-1])
    return jmat, w[-1], v[:, -1]

def rand_mat_gauss_symmetric(B):
    B_T = cp.transpose(B)
    A = 1.0/np.sqrt(2) * cp.add(B, B_T)  # cupy.ndarray
    # print(cp.asnumpy(A))
    return A    

def rand_mat_gauss(n):
    """
    Returns nxn size array of random numbers sampled from a normal distribution with a mean of 0 and standard deviation of 1
    (Uses cupy to generate the random matrix)
    """
    B = cp.random.normal(0, 1, size=(n, n))
    return B


def rand_mat_plusminus(n):
    """
    Returns nxn size array of random numbers uniformly distributed on the 2-value set {-1,1}
    OPTIONAL
    """
    m = cp.random.uniform(0.0, 1.0 , size=(n,n))
    m[m < 0.5] = -1
    m[m >= 0.5] = 1
    return m


def rand_mat_uniform(n):
    """
    Returns nxn size array of random numbers uniformly distrubted on the range [-1,1]
    OPTIONAL
    """
    B = cp.random.uniform(0.0, 1.0 , size=(n,n))
    B_T = cp.transpose(B)
    A = cp.add(B, B_T)  # cupy.ndarray
    A[A < 1] = -1
    A[A >= 1] = 1    
    
    return A

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def problem2():
    eigArr = []
    nArr = []
    normVec = []
    for i in range(1, 20):
        n = i*5
        _, maxEigval, maxEigvec = cupy_J(n)
        eigArr.append(maxEigval)
        nArr.append(n)
        normVec.append(np.linalg.norm(maxEigvec))
    plt.figure()
    plt.plot(nArr, eigArr)
    plt.title('problem2 : eigenvalue vs matrix size(n)')
    plt.ylabel('Eigenvalue')
    plt.xlabel('Size of matrix')
    plt.show()

    # plt.figure()
    # plt.plot(nArr,normVec)
    # plt.show()
    # print(jmat)


def problem3b():
    # verify the matrix to be normal distribution like
    n = 10
    mu, sigma = 0.0, 1.0
    s = np.random.normal(mu, sigma, 1000)
    # Create the bins and histogram
    #count, bins, ignored = plt.hist(s, 20, normed=True)
    mat = rand_mat_gauss_symmetric(rand_mat_gauss(n))
    npMat = cp.asnumpy(mat)
    print(f'The random normal distribution matrix is symmetrix : {check_symmetric(npMat)}')
    arr = cp.asnumpy(cp.squeeze(mat))
    count, bins, ignored = plt.hist(arr,bins = 10, density=True)
    # Plot the distribution curve
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu)**2 / (2 * sigma**2)),linewidth=3, color='y')
    plt.title('problem3b: normal distribution of values in a matrix ')
    plt.show()
    
def problem3c():
    n = 1000
    m = rand_mat_gauss_symmetric(rand_mat_gauss(n))
    eigvals = cupy_eig(m)
    plt.hist(cp.asnumpy(eigvals))
    plt.title('problem3c: normal distribution of eigenvalues of a matrix, n= 1000 ')
    plt.show()

def problem3d():
    n = 2000
    m_2000 = rand_mat_gauss_symmetric(rand_mat_gauss(n))
    eigvals_2000 = cupy_eig(m_2000)
    plt.hist(cp.asnumpy(eigvals_2000))
    plt.title('problem3d: normal distribution of eigenvalues of a matrix, n= 2000 ')
    plt.show()
    
    n = 4000
    m_4000 = rand_mat_gauss_symmetric(rand_mat_gauss(n))
    eigvals_4000 = cupy_eig(m_4000)
    plt.hist(cp.asnumpy(eigvals_4000))
    plt.title('problem3d: normal distribution of eigenvalues of a matrix, n= 4000 ')
    plt.show()
    
def problem4a():
    arr = [10,100,1000,2000]
    for n in arr:
        m = rand_mat_plusminus(n)
        npm = cp.asnumpy(m)
        eigvals = cupy_eig(m)
        plt.hist(cp.asnumpy(eigvals)[:-1])
        plt.title('problem4a: normal distribution of eigenvalues of a matrix, n= {} '.format(n))
        plt.show()
        
    
def problem4b():
    arr = [10,100,1000,2000]
    for n in arr:
        m = rand_mat_uniform(n)
        npm = cp.asnumpy(m)
        eigvals = cupy_eig(m)
        plt.hist(cp.asnumpy(eigvals)[:-1])
        plt.title('problem4b: normal distribution of eigenvalues of a matrix, n= {} '.format(n))
        plt.show()      
    
if __name__ == "__main__":

    # Fill in code here to call functions

    # # problem 1 Done
    cupy_fft()
    cupy_filter()  # need to execute this one
    # # 1-(a) : cupy installed, using pip install cupy-cuda80, checked with ``import cupy as cp``
    # # 1-(b) : cupy_fft() function modified successfully
    # # 1-(c): cupy_filter() works well

    problem2() #done
    # # problem2 discussion:
    # # When n becomes large, the max eigenvalue asymptotically approch to 2
    # # the corresponding eigenvector all come with the same sign (all positive or negative)
    # # also, the eigenvector is symmetric

    # # problem3
    # # 3-(a) : rand_mat_gauss finished, return an symmetric normal distribution matrix
    problem3b() #done
    problem3c() #done
    # # 3-(c) : the distribution of eigenvalues is also normal distribution
    problem3d() # done
    # # 3-(d) : I found that the number of eigenvalues within different region are increasing as n goes up, O(n)
    
    # problem 4
    problem4a()
    # 4-(a): rand_mat_plusminus() finished, the number of eigenvalues in a distribution region has growing rate to be O(n)
    problem4b()
    # 4-(b): rand_mat_uniform() finished, the number of eigenvalues in a distribution region has growing rate to be O(n)

    
    
    