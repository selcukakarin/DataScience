# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:12:44 2019

@author: selcuk
"""

import numpy as py
import matplotlib.pyplot as plt
import pandas as pd

#kodlar
#veri yukleme
veriler=pd.read_csv("eksikveriler.csv")

print(veriler)

# =============================================================================
# #veri on isleme
# boy=veriler[["boy"]]
# print(boy)
# 
# boyKilo=veriler[["boy","kilo"]]
# print(boyKilo)
# 
# 
# =============================================================================
#eksik veriler
#sci - kit learn
from sklearn.preprocessing import Imputer
# strategy="mean" demek ortalama al demek
# axis satırdaki veriler için mi ortalama alınacak sütun bazında mı ortalama alınacak onu belirler
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
# iloc pandasın tablomuzdaki hangi değerleri alacağımıza işaret edebilen bir al fonksiyonu
yas=veriler.iloc[:,1:4].values   # 1,2 ve 3. indexli kolonları aldık
print(yas)
imputer=imputer.fit(yas[:,1:4])    #  aldığımız kolonlara imputer fonksiyonunu uyguladık
yas[:,1:4]=imputer.transform(yas[:,1:4])     #  impute ediler veriyi aldığımız kolonların üzerine yazdık
print(yas)
# =============================================================================
# ########################## Diğer yöntem
# yas=veriler.iloc[:,3:4].values    #yas sütununu çektik
# imputer=imputer.fit(yas)    #imputer fonksiyonunu yeni çektiğimiz yaş sütununa uyguladık
# veriler["yas"]=imputer.transform(yas) #impute edilen veriyi veriler tablomuza transform ettik
# print(veriler)
# =============================================================================
