import numpy as np
from statsmodels import api as sm
import itertools
import numba
from statistics import mean
from mpl_toolkits.mplot3d import Axes3D   
import matplotlib.pyplot as plt


#TODO
# Optinon parse for segment switching

def get_data(seq):
    return [ float(v) for v in seq.split(',') ]


#FIXME
def get_power(dim, order):
    p = (order+1)**dim
    base = [ int(np.base_repr(i,3)) for i in range(p) ]
    base = [ "{:02d}".format(int(v)) for v in base ]
    for i, v in enumerate(base):
        ii, iii = list(v)
        base[i] = (int(ii), int(iii))
    return base[1:]


def coefOfDet(y,f):
    f_bar = mean(f)
    SSR = sum([ (fi - f_bar)**2 for fi in f ])
    SST = sum([ (yi - y_bar)**2 for yi in y ])
    SSE = sum([ (yi - fi)**2 for yi, fi in zip(f,y) ])
    r2 = SSR/SST
    print((SSR+SSE)/SST)
    return r2


if __name__ == '__main__':
    print('Reading input data')
    with open('aaa.csv', 'r') as f:
        Y = []
        #TODO Non-dimentionalize
        A = get_data(f.readline())
        B = get_data(f.readline())
        for l in f:
            Y.append(get_data(l))
    Y = np.array(Y).T.flatten()

    print('Constructing Simultaneous Linear Equations')
    # dim := number of parameters
    dim = 2
    # order := order of polynominal equation
    order = 2
    # p := nb of terms; n := nb of observed points
    p, n = (order + 1) ** dim - 1 , 6 * 6
    # S := variance-covariance matrix
    S = np.zeros([p,p])
    # a := coefficient vector
    a = M = np.zeros([p])
    # x := variable matrix
    x = np.zeros([n,p])
    # y := observed value vector
    y = np.zeros([n])
    
    #TODO non-dimentionalize
    params = [ (a,b) for a, b in itertools.product(A, B) ]
    params = np.array(params)
    b = get_power(dim, order)
    for i in range(n):
        for j in range(p):
            ii, iii = b[j]
            x[i][j] = (params[i][0]**ii) * (params[i][1]**iii)
    y = np.array(Y)

    x_bar = [ mean(x[:,j]) for j in range(p) ]
    y_bar = mean(y)

    print('Computating Variance-Covariance Matrix Sjk')
    for j, k in itertools.product(range(p),repeat=2):
        S[j][k] =  sum([ (x[i][j] - x_bar[j]) * (x[i][k] - x_bar[k]) for i in range(n) ])

    print('Computating Sum of Deviation Vector M')
    for j in range(p):
        M[j] = sum([(x[i][j] - x_bar[j]) * (y[i] - y_bar) for i in range(n) ])

    print('Computating Coefficient Vector a')
    a = np.dot(np.linalg.inv(S),M)

    print('Generating Polynominal Function F')
    X = [(14500.**b[i][0]) * (28.**b[i][1]) for i in range(p)]
    X = np.array(X)
    a0 = y_bar - np.dot(a,x_bar)
    print('a0: {:> 10e}'.format(a0))
    for i, ai in enumerate(a):
        print('a{:}: {:> 10e}'.format(i+1,ai))

    def F(x):
        X = [(x[0]**b[i][0])*(x[1]**b[i][1]) for i in range(p)]
        X = np.array(X)
        return np.dot(X,a) + a0

    print('Computating Coefficient of Determination')
    x1 = np.linspace(14500., 15500.)
    x2 = np.linspace(28.,33.)
    z = [[ F([x1i,x2i]) for x1i in x1] for x2i in x2]
    z = np.array(z)
    print('R2: {}'.format(coefOfDet(y.flatten(),z.flatten())))

    print('Plotting Response Surface')
    fig = plt.figure() 
    ax = fig.gca(projection='3d')
    x1, x2 = np.meshgrid(x1, x2)
    ax.plot_surface(x1,x2, z, rstride=1, cstride=1, cmap='viridis', linewidth=0.3)
    x1 = np.linspace(14500,15500,6)
    x2 = np.linspace(28,33,6)
    x1,x2 = np.meshgrid(x1,x2)
    ax.scatter(x1,x2,y,c='r')
    plt.show()
