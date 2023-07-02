# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:50:53 2019

@author: selcuk
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler=pd.read_csv("maaslar.csv")
# TEST 
print(veriler)

#data frame dilimleme (slice)
# iloc fonksiyonu dataframe tipinde değer dönüyor
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

#NumPY dizi (array) dönüşümü
# biz ise lin_reg fonksiyonu için bu dataframe'deki değerleri istiyoruz
X=x.values
Y=y.values
print("------- x")
print(x)
print("------- y")
print(y)

# linear regression
# doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
## Linear regression metodlarına ulaşabilmek için aşağıda bir obje oluşturduk
lin_reg=LinearRegression()
## objeden model oluşturduk
# X'den Y'i öğrenir
lin_reg.fit(X,Y)

plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg.predict(x.values),color="blue")
# plt.show()

# polynomial regression
# doğrusal olmayan (nonlinear model) oluşturma
# 2. dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree = 2)
x_poly=poly_reg.fit_transform(X)
print(x_poly)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

# X ve Y değerlerini noktalar halinde grafikte gösterir
plt.scatter(X,Y,color="red")
# Tahmin çizgisini grafikte gösterir
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color="blue")
# plt.show()
# show yapılmazsa diğer çizilen plt değerleri de aynı plt üzerine yazılır

# 4. dereceden polinomal regresyon
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree = 4)
x_poly=poly_reg.fit_transform(X)
print(x_poly)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
# X ve Y değerlerini noktalar halinde grafikte gösterir
plt.scatter(X,Y,color="red")
# Tahmin çizgisini grafikte gösterir
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color="blue")
# plt.show()

# tahminler

print(lin_reg.predict(np.array([[11]])))
print(lin_reg.predict(np.array([[6.6]])))
 
print(lin_reg2.predict(poly_reg.fit_transform(np.array([[11]]))))
print(lin_reg2.predict(poly_reg.fit_transform(np.array([[6.6]]))))