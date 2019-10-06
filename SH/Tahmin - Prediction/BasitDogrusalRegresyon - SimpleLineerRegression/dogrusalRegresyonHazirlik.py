# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:50:53 2019

@author: selcuk
"""
################ 1. KUTUPHANELER ################
import numpy as py
import matplotlib.pyplot as plt
import pandas as pd

################ 2. VERi ÖN İŞLEME ################
################ 2.1. VERİ YÜKLEME ################
veriler=pd.read_csv("satislar.csv")
# TEST 
print(veriler)

# veri ön işleme
aylar=veriler[['Aylar']]
# test 
print(aylar)

satislar=veriler[['Satislar']]
print(satislar)

satislar2=veriler.iloc[:,0:1].values
print(satislar2)


# VERİLERİN EĞİTİM VE TEST İÇİN BÖLÜNMESİ
# from sklearn.cross_validation import train_test_split kütüphanesini yerine yeni sürümde aşağıdaki kütüphane tanımlamasını kullanıyoruz
from sklearn.model_selection import train_test_split
# aylar = bağımsız değişken
# satislar = bağımlı değişken
# test_size = genel kabul görmüş percentage split test verisi 1/3 -- Eğitim verisi 2/3
# random state verilerimizin ilk 50 veriyi almak yerine rastgele alınmasını sağlar 
x_train, x_test, y_train, y_test=train_test_split(aylar,satislar,test_size=0.33,random_state=0)
"""
from sklearn.preprocessing import StandardScaler
####### VERİLERİN STANDARDİZE EDİLMESİ (ÖLÇEKLENMESİ) ######
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
# x_traine StandardScaler() uygulanır X_train e atılır
X_test=sc.fit_transform(x_test)
## verileri aynı dünyaya uyarladık. aynı düzleme çektik
Y_train=sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)
############################################
"""
## MODEL İNŞASI  (linear regression)
# simple linear regression kategorik verilerin sınfılandırılması için kullanılamaz
from sklearn.linear_model import LinearRegression
## Linear regression metodlarına ulaşabilmek için aşağıda bir obje oluşturduk
lr =  LinearRegression()
## objeden model oluşturduk
# X_Train'den Y_train'i öğrenir
lr.fit(x_train,y_train)
## Tahmin işlemini yaparız - X_test'ten Y_test'i tahmin ederiz
tahmin = lr.predict(x_test)
# data frame'deki index değerine göre sıralama yapıldı
x_train=x_train.sort_index()
y_train=y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))

plt.title("ayalara göre satış")
plt.xlabel("aylar")
plt.ylabel("satışlar")


















