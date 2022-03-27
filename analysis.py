import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from 

modified = None

baseline = None

diff = np.subtract(modified,baseline)

mapping = {'pop':0,'country':1,'blues':2,'jazz':3,'reggae':4,'rock':5,'hip hop':6}

sns.heatmap(diff, xticklabels = mapping.keys(),yticklabels = mapping.keys(),ax=ax)

ax = plt.axes()
ax.set_title('SET TITLES')
plt.xlabel ("Target")
plt.ylabel ("Predicted")
plt.savefig("SETNAME.png") 