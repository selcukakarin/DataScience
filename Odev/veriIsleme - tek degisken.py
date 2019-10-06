# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:50:53 2019

@author: selcuk
"""
################ 1. KUTUPHANELER ################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm

################ 2. VERi ÖN İŞLEME ################
################ 2.1. VERİ YÜKLEME ################
dataset=pd.read_csv("dataset.csv")

# EKSİK VERİLER
#sci - kit learn
# iloc pandasın tablomuzdaki hangi değerleri alacağımıza işaret edebilen bir fonksiyonu index location
veriOnisleme=dataset.iloc[:,[1,2]].values   # işlem yapılacak kolonları aldık. (hasta yaşı, BI-RADS sonucu,Kitle yoğunluğu, akraba tarihi,kanser sonucu)
eksikVeri=pd.DataFrame(data=veriOnisleme, columns=["BI-RADS_sonuc","Kanser_Sonucu"])  
# eksikVeri=pd.DataFrame(data=veriOnisleme,index=range(100), columns=["Hasta_yas","BI-RADS_sonuc","Kitle_yogunlugu","Akraba_tarih","Kanser_Sonucu"])  
# 9 değerini alan missing value'lerin değerini işleyebilmek için NaN olarak değiştirdik
veriler=eksikVeri.replace(9, np.nan)
# Akraba tarih alanındaki 9 değerleri NaN'a çevrilmişti ve bu değerleri içeren satırları datasetimizden çıkardık
#dene = veriler[np.isfinite(veriler['Akraba_tarih'])]
# eksik veri içerek satırlar silindi
veriler = veriler.dropna()

print(veriler.corr()) 
# Korelasyon matrisi
cormatris = veriler.corr()

# VERİLERİN EĞİTİM VE TEST İÇİN BÖLÜNMESİ
from sklearn.cross_validation import train_test_split
# x bağımsız değişkenler işlenecek veriler
# y bağımlı değişken - kanser sonucu verisi
# test_size = genel kabul görmüş percentage split test verisi 1/3 -- Eğitim verisi 2/3
# random state verilerimizin ilk 50 veriyi almak yerine rastgele alınmasını sağlar
x_train, x_test, y_train, y_test=train_test_split(veriler.iloc[:,:-1],veriler.iloc[:,-1:],test_size=0.33,random_state=0)

#####################################
#####################################

#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
# x_train ile y_train verileri arasında linear bir model kurduk. x_train'den y_train'i öğrendik
lin_reg.fit(x_train,y_train)

print("linear reg ols")
model1 = sm.OLS(lin_reg.predict(x_test),x_test)
print(model1.fit().summary())

# y'yi tahmin ettik - x'ten gelen verilerin y'deki çıktısı
lin_reg_tahmin = lin_reg.predict(x_test)

# x_train=x_train.sort_index()
# y_train=y_train.sort_index()

# plt.scatter(x_train,y_train,color="red")
# plt.plot(x_test,lin_reg.predict(x_test),color="blue")

# plt.title("kanser sonuç")
# plt.xlabel("bağımsız değişkenler ")
# plt.ylabel("bağımlı deişken kanser sonuc")


#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x_train)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y_train)

poly_reg_tahmin=lin_reg2.predict(poly_reg.fit_transform(x_test))

print("poly reg ols")
model2 = sm.OLS(lin_reg.predict(x_test),x_test)
print(model2.fit().summary())

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(x_train)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(y_train)

from sklearn.svm import SVR
svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli,y_olcekli)

print('svr ols')
model3 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())

#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(x_test,y_test)

print('dt ols')
model4 = sm.OLS(r_dt.predict(x_test),x_test)
print(model4.fit().summary())

#Random Forest Regresyonu
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state=0)
rf_reg.fit(x_test,y_test)

print('rf ols')
model5 = sm.OLS(rf_reg.predict(x_test),x_test)
print(model5.fit().summary())

#ozett R2 degerleri
print('----------------')
print("Linear R2 degeri:")
print(r2_score(y_test, lin_reg.predict((x_test))))


print("Polynomial R2 degeri:")
print(r2_score(y_test, lin_reg2.predict(poly_reg.fit_transform(x_test)) ))


print("SVR R2 degeri:")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)) )


print("Decision Tree R2 degeri:")
print(r2_score(y_test, r_dt.predict(x_test)) )

print("Random Forest R2 degeri:")
print(r2_score(y_test, rf_reg.predict(x_test)) )

