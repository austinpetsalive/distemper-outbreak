import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

t = np.array(list(range(0, 24 * 5))) # in hours

def iterative_exp(xs, a, b):
    start = 0
    result = [start]
    for x in range(1, len(xs)):
        result.append(result[-1]*a+b)
    return [result[0], result[59], result[119]]

immunity = 3.134625261e-1 * np.exp(9.671767868e-3 * t)
risk = 1 - immunity

popt, pcov = curve_fit(iterative_exp, t, [0.0, 0.5, 1.0])
iterative_immunity = [0]
for i in range(1, len(t)):
    iterative_immunity.append(iterative_immunity[-1]*1.03 + 0.001)
print(popt)
print(pcov)
#y = np.array([4, 1, 2, 3]) # 
# Four axes, returned as a 2-d array
f, axarr = plt.subplots(2, 2)
axarr[0, 0].plot(t, immunity)
axarr[0, 0].plot(t, iterative_immunity)
axarr[0, 0].set_title('Immunity Over Time')
axarr[0, 1].plot(t, risk)
axarr[0, 1].set_title('Risk Over Time')
axarr[1, 0].plot(risk, immunity)
axarr[1, 0].set_title('Axis [1,0]')
axarr[1, 1].plot([0], [0])
axarr[1, 1].set_title('Axis [1,1]')
# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
plt.show()