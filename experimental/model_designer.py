# pylint: disable=C0111
import matplotlib.pyplot as plt
import numpy as np
import pylab
from scipy.optimize import curve_fit

def _sigmoid(_x):
    return 2 / (1 + np.exp(np.multiply(_x, -1))) - 1

X = list(range(0, 73))
Y = _sigmoid(X)

plt.plot(X, Y)
plt.show()


def _sigmoid2(_x, _x0, _k):
    return 1.0 / (1.0 + np.exp(-_k*(_x-_x0)))

DATA = np.transpose([[0.0, 0.0],
                     [0.25, 0.1],
                     [0.5, 0.5],
                     [0.75, 0.9],
                     [1.0, 1.0]])

X = DATA[0]
Y = DATA[1]

OPT, COV = curve_fit(_sigmoid2, X, Y)
print(OPT)

XX = np.linspace(0, 1, 72)
YY = _sigmoid2(XX, *OPT)
print('[{0}]'.format(','.join(['{:.4f}'.format(a) for a in YY])))
print(len(YY))

pylab.plot(X, Y, 'o', label='data')
pylab.plot(XX, YY, label='fit')
pylab.ylim(0, 1.05)
pylab.legend(loc='best')
pylab.show()
