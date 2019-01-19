import lagrangeMultiplier as lm
from matplotlib import pyplot as plt
from statsmodels import api as sm
import scipy
import numpy as np
from sklearn import metrics
import glob
import sys


def get_power(dim, order):
    p = (order+1)**dim
    base = [ int(np.base_repr(i,order+1)) for i in range(p) ]
    #FIXME Format should varies on the number of variables
    base = [ "{:02d}".format(int(v)) for v in base ]
    for i, v in enumerate(base):
        base[i] = [ int(ei) for ei in list(v) ]
    return base


def get_params(file_name):
    with open(file_name, 'r') as f:
        Y = []
        A = lm.get_data(f.readline())
        B = lm.get_data(f.readline())
        for l in f:
            Y.append(lm.get_data(l))
    A, B = np.array(A), np.array(B)
    a1, a2 = A, B
    A, B = np.meshgrid(A,B)
    A, B = A.flatten(), B.flatten()
    Y = np.array(Y)
    Y = Y.T
    Y = Y.flatten()
    return (A, B,a1,a2, Y)


if __name__ == '__main__':
    # Source
    print('Reading input file')
    print('Please input file name:')
    print('[NOTE] The prorgram check only the same location as '\
            'this source code locates.')
    file_name = input()
    
    print(file_name)
    
    print('Initializing inputs')
    A, B, a1,a2, Y = get_params(file_name)
    # Plot nb of param
    print()
    
    print('Please input the order for polynominal regression:')
    order = input()
    dim = order = 2
    n = len(Y)
    p = (order+1)**dim
    X = np.repeat(1, n) # Coefficient term
    e = get_power(dim, order)
    for j in range(1,p):
        ii, iii = e[j]
        X = np.column_stack((X,(A**ii)*(B**iii)))
    model = sm.OLS(Y,X)
    results = model.fit()
    params = results.params
    print("\n========")
    print("Parameter: ")
    for i, b in enumerate(params):
        print("Beta{} = {}".format(i,b))
    #print(results.summary())

    def F(x):
        X = [(x[0]**e[i][0])*(x[1]**e[i][1]) for i in range(p)]
        X = np.array(X)
        return np.dot(X,params)

    # 3D projection

    ax = plt.figure().gca(projection='3d')
    ax.scatter(A,B,Y)
    x1 = np.linspace(A[0],A[-1])
    x2 = np.linspace(B[0],B[-1])
    y = np.array([[ F([x1i,x2i]) for x1i in x1] for x2i in x2])
    x1, x2 = np.meshgrid(x1,x2)
    ax.plot_surface(x1, x2, y, cmap='viridis')
    Y_p = np.array([[ F([x1i,x2i]) for x1i in a1] for x2i in a2])
    Y_p = Y_p.flatten()
    print("R squared score (No configured): ", metrics.r2_score(Y,Y_p))

    # Compute optima point
    x_init = np.array([A[1],B[1]])
    bounds = np.array([[A[0],A[-1]], [B[0],B[-1]]])
    rslt = scipy.optimize.minimize(F, x_init, method = 'L-BFGS-B', bounds=bounds)
    x, y = rslt.x
    print("Optima point:")
    print("1st variable = ", x, "; 2nd variable =", y)
    ax.scatter(x,y,F([x,y]))

    plt.show()
