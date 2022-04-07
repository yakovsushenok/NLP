"""
    file : analysis
    authors : 21112254, 16008937, 20175911, 21180859

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


modified = np.array([[394.33333333, 102.        , 101.        , 132.33333333,
        149.66666667],
       [ 64.66666667, 427.33333333, 125.33333333,  67.66666667,
         65.66666667],
       [ 79.66666667, 149.33333333, 307.        ,  91.66666667,
        120.66666667],
       [ 81.33333333,  69.66666667, 104.        , 361.66666667,
         84.33333333],
       [192.33333333,  57.66666667, 163.33333333, 117.33333333,
        359.        ]])

baseline = np.array([[258.5,  64. ,  72.5, 120. , 109.],
       [125. , 406. , 142. , 122.5, 122.],
       [181. , 197.5, 373. , 181. , 220.5],
       [ 99. ,  68.5, 104. , 315.5,  94.5],
       [120.5,  34. ,  97. ,  78. , 263.5]])

diff = np.subtract(modified,baseline)
#cmap = sns.color_palette("coolwarm", as_cmap=True)
mapping = {'pop':0,'country':1,'blues':2,'jazz':3,'rock':4}
ax = plt.axes()
#,cmap = cmapcenter=0
sns.heatmap(baseline, xticklabels = mapping.keys(),yticklabels = mapping.keys(),ax=ax,vmax=500,vmin=0)


ax.set_title('CNN baseline (Total test count 3972)')
plt.xlabel ("Target")
plt.ylabel ("Predicted")
plt.savefig("cnn_baseline.png") 