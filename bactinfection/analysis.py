"""
Class implementing analysis of segmentation data
"""

# Author: Guillaume Witz, Science IT Support, Bern University, 2019
# License: BSD3


import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipywidgets as ipw
import pickle
from sklearn import mixture

from . import utils
from .segmentation import Bact

class Analysis(Bact):
    
    def __init__(self):

        """Standard __init__ method.
        
        Parameters
        ----------
        bact : Bact object
            Bact object
        
        
        Attributes
        ----------
            
        all_files = list
            list of files to process
        
        """
        Bact.__init__(self)
        
        self.out = ipw.Output()
        self.out_plot = ipw.Output()
        self.load_button = ipw.Button(description = 'Load segmentation')
        self.load_button.on_click(self.load_infos)
        
        self.sel_channel = ipw.SelectMultiple(options = [],
                                     layout = {'width': '200px'},style = {'description_width': 'initial'})
        self.sel_channel.observe(self.plot_byhour_callback, names = 'value')
        
        self.button_plotbyhour = ipw.Button(description = 'Plot by hour')
        self.button_plotbyhour.on_click(self.plot_byhour_callback)
        
        self.bin_width = ipw.IntSlider(min = 10, max = 1000, value = 100)
        self.bin_width.observe(self.plot_byhour_callback, names = 'value')
        self.hour_select = ipw.Select(options = [], value = None)
        self.hour_select.observe(self.plot_byhour_callback, names = 'value')
        
        self.load_analysis_button = ipw.Button(description = 'Load analysis')
        self.load_analysis_button.on_click(self.load_analysis)
        
        self.save_analysis_button = ipw.Button(description = 'Save analysis')
        self.save_analysis_button.on_click(self.save_analysis)
        
        self.GM = None
          
    def load_infos(self,b = None):
        
        with self.out:
            self.load_segmentation()
            
        self.sel_channel.options = self.channels
        self.create_result()
        self.hour_select.options = self.result.hour.unique()

        
        
    def create_result(self):
        
        empty_dict = {x:[] for x in self.channels if x is not None}
        empty_dict['hour'] = []
        empty_dict['replicate'] = []
        empty_dict['filename'] = []

        for x in self.bacteria_channel_intensities:
            if self.bacteria_channel_intensities[x] is not None:
                for c in self.channels:
                    if c is not None:
                        numvals = len(self.bacteria_channel_intensities[x][c])
                        empty_dict[c]+=list(self.bacteria_channel_intensities[x][c])
                empty_dict['hour']+=numvals*[int(re.findall('\_(\d+)h\_', x)[0])]
                empty_dict['filename']+=numvals*[x]
                index = re.findall('\_(\d+)\.', x)
                if len(index)== 0:
                    index = 0
                else:
                    index = int(index[0])
                empty_dict['replicate']+=numvals*[index]
        
        empty_dict = pd.DataFrame(empty_dict)
        self.result = empty_dict
        
    def pool_intensities(self):
        pooled = {k: np.concatenate([self.bacteria_channel_intensities[x][k] for x in self.bacteria_channel_intensities.keys()])
 for k in self.channels if k is not None}
        
        return pooled
    
    def plot_byhour_callback(self, b= None):
        
        self.out_plot.clear_output()
        with self.out_plot:
            
            self.plot_split_intensities(bin_width = self.bin_width.value,
                         channel = self.sel_channel.value,
                         hour = self.hour_select.value)
            
    def plot_split_intensities(self, channel = None, hour = None, bin_width = 10, min = 0, max = 3000):
        if (len(channel) == 0) or (hour is None) :
            print('select a channel and an hour')
        else:
            grouped = self.result.groupby('hour')
            sel_group = grouped.get_group(hour)
            fig, ax = plt.subplots(figsize=(10,7))
            for c in channel:
                self.split(sel_group[c].values)
                hist_val, xdata = np.histogram(sel_group[c].values,bins = np.arange(min,max,bin_width),density=True)
                xdata = np.array([0.5*(xdata[x]+xdata[x+1]) for x in range(len(xdata)-1)])
                ind1 = 0
                ind2 = 1
                ax.bar(x=xdata, height=hist_val, width=xdata[1]-xdata[0],color = 'gray',label='Data')

                ax.plot(xdata,self.normal_fit(xdata,self.GM.weights_[ind1], 
                                                  self.GM.means_[ind1,0], self.GM.covariances_[ind1,0,0]**0.5),
                                     'b',linewidth = 2, label='Cat1')
                ax.plot(xdata,self.normal_fit(xdata,self.GM.weights_[ind2], 
                                                  self.GM.means_[ind2,0], self.GM.covariances_[ind2,0,0]**0.5),
                                     'r',linewidth = 2, label='Cat2')
                #ax.hist(sel_group[c], label = c, alpha = 0.5, bins = np.arange(min,max,bin_width))
            ax.legend()
            ax.set_title('Hour '+str(hour))
            plt.show()
        
    def plot_result_groupedby_hour(self, min = 0, max = 3000, bin_width = 100, channels = None):
        
        if channels is None:
            channels = self.channels
        grouped = self.result.groupby('hour')
        for x,y in grouped:
            fig, ax = plt.subplots(figsize=(10,7))
            for c in channels:
                if c is not None:
                    self.GM = self.split(y[c].values)
                    hist_val, xdata = np.histogram(y[c].values,bins = np.arange(min,max,bin_width),density=True)
                    xdata = np.array([0.5*(xdata[x]+xdata[x+1]) for x in range(len(xdata)-1)])
                    ind1 = 0
                    ind2 = 1
                    ax.bar(x=xdata, height=hist_val, width=xdata[1]-xdata[0],color = 'gray',label='Data')

                    ax.plot(xdata,self.normal_fit(xdata,self.GM.weights_[ind1], 
                                                  self.GM.means_[ind1,0], GM.covariances_[ind1,0,0]**0.5),
                                     'b',linewidth = 2, label='Cat1')
                    ax.plot(xdata,self.normal_fit(xdata,self.GM.weights_[ind2], 
                                                  self.GM.means_[ind2,0], GM.covariances_[ind2,0,0]**0.5),
                                     'r',linewidth = 2, label='Cat2')
                    #ax.hist(y[c], label = c, alpha = 0.5, bins = np.arange(min,max,bin_width))
            ax.legend()
            ax.set_title(x)

    def split(self, data):
        X = np.reshape(data,(-1,1))
        GM = mixture.GaussianMixture(n_components=2)
        GM.fit(X)
        self.GM = GM
     
    def normal_fit(self, x, a, x0, s): 
        return (a/(s*(2*np.pi)**0.5))*np.exp(-0.5*((x-x0)/s)**2)
    
    def plot_hist_single_file(self):
        fig, ax = plt.subplots(figsize=(10,7))
        for k in self.bacteria_channel_intensities[self.select_file.value].keys():
            ax.hist(self.bacteria_channel_intensities[self.select_file.value][k],bins = np.arange(0,3000,100),
                    density = True,alpha = 0.5, label = k)
        ax.legend()
        ax.set_title(self.select_file.value)
        plt.show()
        
        
    def save_analysis(self, b = None):
        if not os.path.isdir(self.folder_name+'/Analyzed/'):
            os.makedirs(self.folder_name+'/Analyzed/',exist_ok=True)
        file_to_save = self.folder_name+'/Analyzed/'+os.path.split(self.folder_name)[-1]+'.pkl'
        with open(file_to_save, 'wb') as f:
            to_export = {'bact_channel':self.bact_channel,
                         'nucl_channel':self.nucl_channel,
                         'bacteria_channel_intensities':self.bacteria_channel_intensities,
                         'channels':self.channels, 'all_files':self.all_files,
                        'result': self.result}
            pickle.dump(to_export, f)
            
            
    def load_analysis(self, b = None):
        
        file_to_load = self.folder_name+'/Analyzed/'+os.path.split(self.folder_name)[-1]+'.pkl'
        if not os.path.isfile(file_to_load):
            print('No analysis found')
        else:
            with open(file_to_load, 'rb') as f:
                temp = pickle.load(f)

            for k in temp.keys():
                setattr(self, k, temp[k])
            
            print('Loading Done')
        