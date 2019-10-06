# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 15:56:42 2019

@author: selcuk
"""

import numpy as py
import matplotlib.pyplot as plt
import pandas as pd

#kodlar
#veri yukleme
veriler=pd.read_csv("veriler.csv")

print(veriler)

#veri on isleme
boy=veriler[["boy"]]
print(boy)

boyKilo=veriler[["boy","kilo"]]
print(boyKilo)

x =10

class insan:
    