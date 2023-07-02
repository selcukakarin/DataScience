# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 23:23:26 2018

@author: user
"""

# importing
import numpy as np

# numpy basics
array = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])  # 1*15 vector

print(array.shape)

a = array.reshape(3,5)
print(a)
print("shape: ",a.shape)
print("dimension: ", a.ndim)

print("data type: ",a.dtype.name)
print("size: ",a.size)

print("type: ",type(a))

array1 = np.array([[1,2,3,4],[5,6,7,8],[9,8,7,5]])
print(array1)

zeros = np.zeros((3,4))
print(zeros)

zeros[0,0] = 5
print(zeros)

ones=np.ones((3,4))
print(ones)

emp=np.empty((2,3))
print(emp)


a = np.arange(10,50,5)
print(a)

a = np.linspace(10,50,20)
print(a)



# %% numpy basic operations

a = np.array([1,2,3])
b = np.array([4,5,6])

print(a)
print(b)
print(a+b)
print(a-b)
print(a**2)

print(np.sin(a))

print(a<2)


a = np.array([[1,2,3],[4,5,6]])
b = np.array([[1,2,3],[4,5,6]])

print(a)
print(b)

# element wise prodcut
print(a*b)

# matrix prodcut
a.dot(b.T)

print(np.exp(a))

a = np.random.random((5,5))

print(a)

print(a.sum())
print(a.max())
print(a.min())

print(a.sum(axis=0))
print(a.sum(axis=1))

print(np.sqrt(a))
print(np.square(a)) # a**2

print(np.add(a,a))


# %% indexing and slicing
import numpy as np
array = np.array([1,2,3,4,5,6,7])   #  vector dimension = 1
print(array)

print(array[0])

print(array[0:4])

reverse_array = array[::-1]
print(reverse_array)


array1 = np.array([[1,2,3,4,5],[6,7,8,9,10]])

print(array1[1,1])

print(array1[:,1])


print(array1[1,1:4])


print(array1[-1,:])
print(array1[:,-1])

# %%
# shape manipulation
array = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(array)

# flatten
a = array.ravel()
print(a)

array2 = a.reshape(3,3)
print(array2)

arrayT = array2.T
print(arrayT)

print(arrayT.shape)


array5 = np.array([[1,2],[3,4],[4,5]])
print(array5)

array5 = np.column_stack((array1,array2))
print(array5)


# %% stacking arrays

array1 = np.array([[1,2],[3,4]])
array2 = np.array([[-1,-2],[-3,-4]])
print(array1)
print(array2)

# veritical
#array([[1, 2],
#       [3, 4]])
#array([[-1, -2],
#       [-3, -4]])
array3 = np.vstack((array1,array2))
print(array3)

# horizontal
#array([[1, 2],[-1, -2],
#       [3, 4]],[-3, -4]]
array4 = np.hstack((array1,array2))
print(array4)

#%% convert and copy

liste = [1,2,3,4]   # list

array = np.array(liste) #np.array

liste2 = list(array)

a = np.array([1,2,3])

b = a
b[0] = 5
c = a
print(a)
print(b)
print(c)

d =  np.array([1,2,3])
e = d.copy()
f = d.copy()
d[0] = 5

print(d)
print(e)
print(f)




















