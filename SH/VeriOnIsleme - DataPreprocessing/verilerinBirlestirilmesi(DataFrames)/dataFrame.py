# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:50:53 2019

@author: selcuk
"""

import numpy as py
import matplotlib.pyplot as plt
import pandas as pd

#kodlar
#veri yukleme
veriler=pd.read_csv("eksikveriler.csv")

print(veriler)

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

sonucUlke=pd.DataFrame(data=ulke, index=range(22),columns=["fr","tr","us"])    
# ulke listemizde 22 değer olduğu için 22 verdik
print(sonucUlke)

sonucYas=pd.DataFrame(data=yas,index=range(22),columns=["boy","kilo","yas"])
print(sonucYas)

cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)
sonucCinsiyet=pd.DataFrame(data=cinsiyet,index=range(22),columns=["cinsiyet"])
print(sonucCinsiyet)

# axis kolon bazlı değil satır bazlı dataFrame birleştirmesi yapar
s=pd.concat([sonucUlke,sonucYas],axis=1)
print(s)

s2=pd.concat([s,sonucCinsiyet],axis=1)
print(s2)

