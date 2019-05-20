# -*- coding: utf-8 -*-
"""
Created on Sun May 19 13:18:51 2019

@author: marco
"""

import matplotlib.pyplot as plt
import numpy as np

from core import get_txt_data


ch1_2 = []
ch1_3 = []
ch1_4 = []
ch2_1 = []
ch2_3 = []
ch2_4 = []
ch3_1 = []
ch3_2 = []
ch3_4 = []
ch4_1 = []
ch4_2 = []
ch4_3 = []

for i in range(984):
    txt_path = f'checkpoints/IMS/IMS{(i+1)*10}_granger_matrix.txt'
    data = get_txt_data(txt_path, delimiter=' ')
    ch1_2.append(data[0,1])
    ch1_3.append(data[0,2])
    ch1_4.append(data[0,3])
    ch2_1.append(data[1,0])
    ch2_3.append(data[1,2])
    ch2_4.append(data[1,3])
    ch3_1.append(data[2,0])
    ch3_2.append(data[2,1])
    ch3_4.append(data[2,3])
    ch4_1.append(data[3,0])
    ch4_2.append(data[3,1])
    ch4_3.append(data[3,2])
    
final = [[ch1_2, ch1_3, ch1_4], [ch2_1, ch2_3, ch2_4],
         [ch3_1, ch3_2, ch3_4], [ch4_1, ch4_2, ch4_3]] 

x = np.linspace(10, 9840, 984)/60

def plot_result():
    for i in range(4):
        idx= list(set(range(1,5)) - {i+1})
        plt.figure(dpi=200)
        plt.subplot(2,2,1)
        plt.plot(x,final[i][0], color='blue')
        plt.xlim([0, 164])
        plt.ylim([0, 0.4])
        plt.title(f'bearing {i+1} to {idx[0]}')
        plt.xlabel('time(hr)')
        plt.ylabel('casuality')
        plt.xticks(np.linspace(0,164,16),rotation=90)
        plt.yticks(np.linspace(0.0,0.4,9))
        plt.grid()  
        plt.subplot(2,2,2)
        plt.plot(x,final[i][1], color='red')
        plt.xlim([0, 164])
        plt.ylim([0, 0.4])
        plt.title(f'bearing {i+1} to {idx[1]}')
        plt.xlabel('time(hr)')
        plt.ylabel('casuality')
        plt.xticks(np.linspace(0,164,16),rotation=90)
        plt.yticks(np.linspace(0.0,0.4,9))
        plt.grid()  
        plt.subplot(2,2,3)
        plt.plot(x,final[i][2], color='green')
        plt.xlim([0, 164])
        plt.ylim([0, 0.4])
        plt.title(f'bearing {i+1} to {idx[2]}')
        plt.xlabel('time(hr)')
        plt.ylabel('casuality')
        plt.xticks(np.linspace(0,164,16),rotation=90)
        plt.yticks(np.linspace(0.0,0.4,9))
        plt.grid()  
        plt.subplot(2,2,4)
        plt.plot(x,final[i][0], color='blue')
        plt.plot(x,final[i][1], color='red')
        plt.plot(x,final[i][2], color='green')
        plt.xlim([0, 164])
        plt.ylim([0, 0.4])
        plt.title(f'bearing {i+1} to {idx[0]},{idx[1]},{idx[2]}')
        plt.xlabel('time(hr)')
        plt.ylabel('casuality')
        plt.xticks(np.linspace(0,164,16),rotation=90)
        plt.yticks(np.linspace(0.0,0.4,9))
        plt.grid()  
        plt.tight_layout()
        plt.savefig(f'images/result/IMS_GrangerCasulaity_Channel{i+1}.jpg')

plot_result()