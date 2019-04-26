# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 15:20:08 2018

@author: ju357
"""

import matplotlib.pylab as plt

def fill_plot(x,mean,error,color,label='my_data'):
    
    plt.plot(x, mean, 'k', color=color, label=label)
    plt.fill_between(x, mean-error, mean+error, 
                 alpha=0.4, edgecolor=color, facecolor=color, 
                linewidth=4, linestyle='dashdot', antialiased=True)  
                
    
    
    
    