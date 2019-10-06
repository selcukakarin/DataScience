# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:50:53 2019

@author: selcuk
"""
################ 1. KUTUPHANELER ################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

################ 2. VERi ÖN İŞLEME ################
################ 2.1. VERİ YÜKLEME ################
veriler=pd.read_csv("odev_tenis.csv")


# ENCODER  NOMİNAL VEYA ORDİNAL'DEN NUMERİC VERİ OLUŞTURMA ((KATEGORIC)NOMİNAL ORDİNAL --> NUMERİC)

# =============================================================================
# # play kolonunu aldık
# play=veriler.iloc[:,-1:].values
# print(play)
# # LabelEncoder her değeri sayısal değere çevirebilir
# from sklearn.preprocessing import LabelEncoder
# le=LabelEncoder()
# ## play kolonunu labelEncoder ile numeric veriye çevirdik
# # fit_transform hem veriye uygular hem veri üzerine yazar
# play[:,0]=le.fit_transform(play[:,0])
# print(play)
# 
# # windy kolonunu aldık
# windy=veriler.iloc[:,-2:-1].values
# print(windy)
# # LabelEncoder her değeri sayısal değere çevirebilir
# from sklearn.preprocessing import LabelEncoder
# le=LabelEncoder()
# ## windy kolonunu labelEncoder ile numeric veriye çevirdik
# # fit_transform hem veriye uygular hem veri üzerine yazar
# windy[:,0]=le.fit_transform(windy[:,0])
# print(windy)
# =============================================================================

## yukarıdaki yapıda play ve windy kolonlarını labelEncoder ile numeric'e çevirdik
# aşağıda ise veriler.apply diyerek bütün kolonlara LabelEncoder uyguladık
#encoder:  Kategorik -> Numeric
from sklearn.preprocessing import LabelEncoder
veriler2 = veriler.apply(LabelEncoder().fit_transform)
# TRICK - Tüm kolonlara tek seferde labelencoder uyguladık



outlook = veriler2.iloc[:,:1]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
outlook=ohe.fit_transform(outlook).toarray()
print(outlook)

havadurumu = pd.DataFrame(data = outlook, index = range(14), columns=['o','r','s'])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis = 1)
# humidity ve temperature veriler'den gelir
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler], axis = 1)


# VERİLERİN EĞİTİM VE TEST İÇİN BÖLÜNMESİ
from sklearn.cross_validation import train_test_split
# sonveriler.iloc[:,:-1] = işlenecek veriler
# sonveriler.iloc[:,-1:] = hedef column
# test_size = genel kabul görmüş percentage split test verisi 1/3 -- Eğitim verisi 2/3
# random state verilerimizin ilk 50 veriyi almak yerine rastgele alınmasını sağlar
x_train, x_test, y_train, y_test=train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33,random_state=0)

# LinearRegression yapıyoruz
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# x_train ile y_train verileri arasında linear bir model kurduk. x_train'den y_train'i öğrendik
regressor.fit(x_train,y_train)
# y'yi tahmin ettik
y_pred = regressor.predict(x_test)

# ilk prediction değerini variable explorer da görmek için değişkene attık
ilk_pred=y_pred


# BACKWARD ELIMINATION
import statsmodels.formula.api as sm
# Beta0 değerlerini ekledik - np.ones((22,1)).astype(int) ile 1 lerden oluşan 22satır bir dizi oluşturduk ve veriye ekledik
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1)
# tüm kolonları p_value değerlerini  hesaplamak üzere aldık
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
# endog parametresi tahmin edilmesi gereken değer - bağımlı değişken
# exog diğer kolonlar - bağımsız değişkenler
# sm.OLS ile bağımsız değişkenlerin bağımlı değişken üzerindeki etkilerini ölçeriz
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog = X_l)
r = r_ols.fit()
print(r.summary())

# en büyük p-value değerine sahip ilk kolonu windy kolonunu sonverilerden çıkardık
sonveriler=sonveriler.iloc[:,1:]

# BACKWARD ELIMINATION
import statsmodels.formula.api as sm
# Beta0 değerlerini ekledik - np.ones((22,1)).astype(int) ile 1 lerden oluşan 22satır bir dizi oluşturduk ve veriye ekledik
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1)
# tüm kolonları p_value değerlerini  hesaplamak üzere aldık
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
# endog parametresi tahmin edilmesi gereken değer - bağımlı değişken
# exog diğer kolonlar - bağımsız değişkenler
# sm.OLS ile bağımsız değişkenlerin bağımlı değişken üzerindeki etkilerini ölçeriz
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog = X_l)
r = r_ols.fit()
print(r.summary())

# yeniden bir tahmin modeli oluşturmak için sonveriler'den çıkardığımız kolonu x_train ve x_test'ten de çıkardık
x_train=x_train.iloc[:,1:]
x_test=x_test.iloc[:,1:]

# x_train ile y_train verileri arasında linear bir model kurduk. x_train'den y_train'i öğrendik
# eğitime katılcak veriler elemeden sonra güncellendi
regressor.fit(x_train,y_train)
# y'yi tahmin ettik
# son tahmin
y_pred = regressor.predict(x_test)






