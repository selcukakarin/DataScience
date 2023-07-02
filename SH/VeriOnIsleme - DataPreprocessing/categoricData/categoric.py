# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:38:34 2019

@author: selcuk
"""

import numpy as py
import matplotlib.pyplot as plt
import pandas as pd

#kodlar
#veri yukleme
veriler=pd.read_csv("eksikveriler.csv")

print(veriler)

#kategorik veriler
#sci - kit learn
from sklearn.preprocessing import Imputer

ulke=veriler.iloc[:,0:1].values
print(ulke)

# LabelEncoder her değeri sayısal değere çevirebilir
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
# fit_transform hem veriye uygular hem veri üzerine yazar
ulke[:,0]=le.fit_transform(ulke[:,0])
print(ulke)

#### farklı bir şekilde sayısal değere çevirme işlemi(kolon bazlı çevirme yapar)
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)