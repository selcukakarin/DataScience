# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 21:06:12 2019

@author: selcuk
"""

import matplotlib.pyplot as plt
import numpy as py
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns



data = pd.read_csv("satislar.csv")



sns.set_style('whitegrid')

sns.FacetGrid(data,height = 5).map(plt.scatter,'Aylar','Satislar').add_legend()
