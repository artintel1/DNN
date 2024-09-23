# import numpy as np

# arr = np.zeros((10,1))
# bias = 5

# arr+=bias
# print(arr)

# class A:
#     c = 0
#     def __init__(self) -> None:
#         self.a = 2
#         self.b = 3
#         global c
#         c = self.a+self.b

#     def add(a,b):
#         return a+b
    
# # a = A()
# print(A.add(2,4))

# dout = 0 if 1== else 5
# print(dout)
# import numpy as np
# arr = []

# def printer(a):
#     for element in a:
#         print(element," ")

# for _ in range(5):
#     arr.append(np.zeros((3,1)))

# printer(arr)

# for i in range(5):
#     v = np.random.randint(0,100,(3,1))
#     arr[i] = v

# printer(arr)
import numpy as np
# arr = np.asarray([1,2,3]).reshape(-1,1)
# print(arr)

# a = np.array([[1,2,3,4,5,6]]).transpose()
# b = np.array([[1,2,3,4,5,6]]).transpose()
# print(a*b)

# l = [1,2,3,4,5]
# l.pop()
# print(l)

arr = []
for _ in range(5):
    arr.append(np.ones((5,1)))

for element in arr:
    element+= np.ones((5,1))

print(arr)
