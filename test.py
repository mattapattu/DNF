#!/usr/bin/bin/env python
import math

a = list()
for i in range(0, 100):  
    a.append((1,20000))
    print 1,20000
for i in range(100,150):
    a.append((1,100))
    print 1,100
for i in range(150,160):
    a.append((1,1))
    print 2,1

#print(a)
for item in a[::-1]:
        print item[1], item[0]

