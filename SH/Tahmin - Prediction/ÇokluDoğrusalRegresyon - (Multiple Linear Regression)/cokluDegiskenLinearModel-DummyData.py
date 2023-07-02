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
veriler=pd.read_csv("veriler.csv")
# TEST 
print(veriler)

# EKSİK VERİLER
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

# ENCODER  NOMİNAL VEYA ORDİNAL'DEN NUMERİC VERİ OLUŞTURMA ((KATEGORIC)NOMİNAL ORDİNAL --> NUMERİC)
ulke=veriler.iloc[:,0:1].values
print(ulke)
# LabelEncoder her değeri sayısal değere çevirebilir
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
# fit_transform hem veriye uygular hem veri üzerine yazar
ulke[:,0]=le.fit_transform(ulke[:,0])
print(ulke)

#### farklı bir şekilde sayısal değere çevirme işlemi(kolon bazlı çevirme yapar)  0 0 1 - 1 0 0 - 0 1 0
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

########################
# -1: ile son kolonu alabiliriz
c=veriler.iloc[:,-1:].values
print(c)
# LabelEncoder her değeri sayısal değere çevirebilir
### DUMMY DATA ( kukla veri ) problemi açısından ordinal verilerde sadece labelEncoder yeterli olabilir.
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
# fit_transform hem veriye uygular hem veri üzerine yazar
c[:,0]=le.fit_transform(c[:,0])
print(c)

#### farklı bir şekilde sayısal değere çevirme işlemi(kolon bazlı çevirme yapar)  0 0 1 - 1 0 0 - 0 1 0
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features='all')
c=ohe.fit_transform(c).toarray()
print(c)
########################

# NUMPY DİZİLERİNİ DATAFRAME DÖNÜŞTÜRDÜK
sonucUlke=pd.DataFrame(data=ulke, index=range(22),columns=["fr","tr","us"])    
# ulke listemizde 22 değer olduğu için 22 verdik
print(sonucUlke)

sonucYas=pd.DataFrame(data=yas,index=range(22),columns=["boy","kilo","yas"])
print(sonucYas)

cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)

# Burada c'deki cinsiyet değerini gösteren E-K kolonlarından sadece birini aldık
# c[:,:1] = tüm satırları al : , :1 0 ile1 arasındaki kolonu al demektir
sonucCinsiyet=pd.DataFrame(data=c[:,:1],index=range(22),columns=["cinsiyet"])
print(sonucCinsiyet)


# DATAFRAME BİRLEŞTİRME İŞLEMİ
# axis kolon bazlı değil satır bazlı dataFrame birleştirmesi yapar
# cinsiyetsiz sonuc = s
s=pd.concat([sonucUlke,sonucYas],axis=1)
print(s)
# cinsiyetli sonuc= s2
s2=pd.concat([s,sonucCinsiyet],axis=1)
print(s2)


# VERİLERİN EĞİTİM VE TEST İÇİN BÖLÜNMESİ
# from sklearn.cross_validation import train_test_split kütüphanesini yerine yeni sürümde aşağıdaki kütüphane tanımlamasını kullanıyoruz
from sklearn.model_selection import train_test_split
# s = işlenecek veriler
# sonuccinsiyet = hedef column
# test_size = genel kabul görmüş percentage split test verisi 1/3 -- Eğitim verisi 2/3
# random state verilerimizin ilk 50 veriyi almak yerine rastgele alınmasını sağlar
x_train, x_test, y_train, y_test=train_test_split(s,sonucCinsiyet,test_size=0.33,random_state=0)
print("----- x_train")
print(x_train)
print("----- x_test")
print(x_test)
print("----- y_train")
print(y_train)
print("----- y_test")
print(y_test)
print("-----")

from sklearn.preprocessing import StandardScaler
####### VERİLERİN STANDARDİZE EDİLMESİ (ÖLÇEKLENMESİ) ######
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
# x_traine StandardScaler() uygulanır X_train e atılır
X_test=sc.fit_transform(x_test)
print("----- X_train")
print(X_train)
print("----- X_test")
print(X_test)
print("----- ")



#############################################

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# x_train ile y_train verileri arasında linear bir model kurduk. x_train'den y_train'i öğrendik
regressor.fit(x_train,y_train)
# y'yi tahmin ettik
y_pred = regressor.predict(x_test)
print("-------")
print("Tahmin")
print(y_pred)
print("-------")
print("Gerçek")
print(y_test)
print("-------")

# s2 ( sonuclar ) değişkeninden boy kolonunu aldık
boy = s2.iloc[:,3:4].values
print(boy)
# boy kolonunun solundaki değerleri aldık
sol = s2.iloc[:,:3]
# boy kolonunun sağındaki değerleri aldık
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)
# veri içerisindeki değerlerden boyu tahmin etmeye çalışıyoruz
x_train, x_test, y_train, y_test=train_test_split(veri,boy,test_size=0.33,random_state=0)
print("----- x_train")
print(x_train)
print("----- x_test")
print(x_test)
print("----- y_train")
print(y_train)
print("----- y_test")
print(y_test)
print("-----")

regressor2 = LinearRegression()
# x_train ile y_train verileri arasında linear bir model kurduk. x_train'den y_train'i öğrendik
regressor2.fit(x_train,y_train)
# y'yi tahmin ettik
y_pred2 = regressor2.predict(x_test)
print("-------")
print("Tahmin")
print(y_pred)
print("-------")
print("Gerçek")
print(y_test)
print("-------")

