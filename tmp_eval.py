import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()
sns.set_style("whitegrid")

labels = ['Agent', 'Snake', 'None']
means =  [0.042593, 0.043415, 0.057471]
stds =   [0.015516, 0.017665, 0.019588]

plt.bar(list(range(len(means))), means, yerr=stds/np.sqrt(30), label=labels, color=list(reversed(sns.color_palette()[0:len(means)])))
plt.xticks(list(range(len(means))), labels)
plt.xlabel('Intervention')
plt.ylabel('Single Hour Infection Rate')
plt.title('Comparison of Inverventions of Single Graph/Hour/State')
plt.gcf().set_size_inches((10, 5))

plt.show()
