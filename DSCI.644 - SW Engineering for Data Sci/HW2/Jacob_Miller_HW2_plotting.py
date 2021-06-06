#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 09:39:25 2020

@author: jmiller
"""


import matplotlib.pyplot as plt

workers = [2, 5, 10, 50, 100]
time_1000 = [.073, .048, .045, .053, .138]
time_10000 = [3.43, 1.44, .844, .388, .610]
time_20000 = [.316, .185, .125, .117, .187]

plt.plot(workers, time_1000, label = '1000', linestyle = '--', marker = 'o')
plt.plot(workers, time_10000, label = '10000', linestyle = '-.', marker = 'v')
plt.plot(workers, time_20000, label = '20000', linestyle = ':', marker = 's')
plt.xlabel('Workers')
plt.ylabel('Time (s)')
plt.legend(title = 'Lines per file')
plt.title('Jacob Miller - HW2\nTime to execute vs. number of workers')
plt.show()