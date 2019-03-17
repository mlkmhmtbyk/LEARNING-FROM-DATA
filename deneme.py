import numpy as np
import matplotlib.pyplot as plt

#reading from regression_data.txt and appends in a reg_data list)
file=open("regression_data.txt","r")
reg_data= list()
for x in file:
    reg_data.append(x)
file.close()

#splits sizes and weights in one line and makes them a list
array_length = len(reg_data)
for i in range(array_length):
    reg_data[i]= reg_data[i].split()

reg_data.pop(0)#deleting first line in txt
reg_data.pop()#deleting empty line at the end of txt

#definition of lists named x and y
x = list() # size
y = list() #weight

#appends the proper data to proper list
for i in reg_data:
    x.append(i[0])
    y.append(i[1])

plt.scatter(x,y)
plt.show()