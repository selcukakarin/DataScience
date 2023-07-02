#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: selcuk
"""

# linear svm doğrusal olarak birbirinden ayrılabilecek iki sınıfı doğrusal bir çizgi ile bölmek için kullanılır
# svm deki mantık maksimum aralığı bulmaya dayalıdır
# svm de dikkat edilmesi gereken bi konu marjinal verilere karşı koruyuculuğunun olmamasıdır. 
# Bunun için svm de mutlaka scaling yapılmalıdır.

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

#data frame dilimleme (slice)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#NumPY dizi (array) dönüşümü
X = x.values
Y = y.values

#linear regression
#doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#polynomial regression
#doğrusal olmayan (nonlinear model) oluşturma
#2. dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

# 4. dereceden polinom
poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)

# Gorsellestirme
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)), color = 'blue')
plt.show()

#tahminler
print(lin_reg.predict(np.array([[11]])))
print(lin_reg.predict(np.array([[6.6]])))
 
print(lin_reg2.predict(poly_reg.fit_transform(np.array([[11]]))))
print(lin_reg2.predict(poly_reg.fit_transform(np.array([[6.6]]))))

from sklearn.preprocessing import StandardScaler
####### VERİLERİN STANDARDİZE EDİLMESİ (ÖLÇEKLENMESİ) ######
sc1=StandardScaler()
X_olcekli=sc1.fit_transform(X)
# X'e StandardScaler() uygulanır X e atılır
sc2=StandardScaler()
Y_olcekli=sc2.fit_transform(Y)
print("----- X")
print(X)
print("----- Y")
print(Y)
print("----- ")

# standardScaler'den geri dönüş
print(sc1.inverse_transform(X_olcekli))
print(sc2.inverse_transform(Y_olcekli))

#############################################


from sklearn.svm import SVR
# kernel='polynomial' ve yahut kernel='linear' olabilir
svr_reg=SVR(kernel='rbf')
svr_reg.fit(X_olcekli,Y_olcekli)

plt.scatter(X_olcekli,Y_olcekli,color="red")
plt.plot(X_olcekli,svr_reg.predict(X_olcekli),color="blue")



print(sc2.inverse_transform(svr_reg.predict(sc1.transform(11))))
print(sc2.inverse_transform(svr_reg.predict(sc1.transform(6.6))))




    

