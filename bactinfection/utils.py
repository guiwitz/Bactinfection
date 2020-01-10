"""
Core functions for data import and analysis
"""

# Author: Guillaume Witz, Science IT Support, Bern University, 2019
# License: BSD3

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

from scipy.optimize import leastsq
import skimage.io as io
from skimage.morphology import binary_closing, watershed, label, binary_opening, disk
from skimage.filters import threshold_li, gaussian
from skimage.feature import match_template, peak_local_max
import skimage
import scipy.ndimage as ndi
import oiffile

import javabridge
import bioformats as bf
javabridge.start_vm(class_path=bf.JARS)

#import spotdetection.spot_detection as sp


#loads an lsm file from filepath and keeps only full resolution image
#returns a list of arrays for each wavelength

def oif_import(filepath, channels = True):

    #image = oiffile.imread(filepath)

    oibfile = oiffile.OifFile(filepath)
    image = oibfile.asarray()
        
    all_channels = []
    for x in oibfile.mainfile.keys():
        channel = re.findall('^Channel.*',x)
        if len(channel)>0:
            all_channels.append(x)
    
    if channels:
        print([oibfile.mainfile[x]['DyeName'] for x in all_channels])
    
    return image

def oir_import(filepath):
    
    with bf.ImageReader(filepath) as reader:
        image = reader.read(series = 0, rescale = False)
    return image
    
    
    
def oif_get_channels(filepath):
    
    oibfile = oiffile.OifFile(filepath)
        
    all_channels = []
    for x in oibfile.mainfile.keys():
        channel = re.findall('^Channel.*',x)
        if len(channel)>0:
            all_channels.append(x)
    
    print([oibfile.mainfile[x]['DyeName'] for x in all_channels])


def segment_nuclei(image, radius = 15):
    
    sigma = radius/np.sqrt(2)
    im_gauss = gaussian(image.astype(float),sigma=1)

    #create a global mask where nuclei migth be fused
    th_nucl = threshold_li(im_gauss)
    mask_nucl = im_gauss > 1*th_nucl
    mask_nucl = binary_opening(mask_nucl, disk(10))
    
    mask_label = label(mask_nucl)

    '''#find local maxima in LoG filterd image
    logim = ndi.filters.gaussian_laplace(im_gauss.astype(float),sigma=sigma)
    peaks = peak_local_max(-logim,min_distance=10,indices=False)
    peaks = peaks*mask_nucl

    #use the blobal and the refined maks for watershed
    im_water = watershed(-image, markers=label(peaks), mask=mask_nucl)#, compactness=1)'''
    
    return mask_label


#create beacteria template
def create_template(length = 7, width = 3):
    #create the bacteria template and its rotated verions

    #template = np.zeros((7,7))
    #template[1:6,2:5]=1

    template = np.zeros((length+2,length+2))
    template[1:1+length,int((length+1)/2)-int((width-1)/2):int((length+1)/2)+int((width-1)/2)+1]=1

    #create a list of rotated templates
    rot_templ = []
    for ind,alpha in enumerate(np.arange(0,180,18)):
        rot_templ.append(skimage.transform.rotate(template,alpha, order = 0))
        
    return rot_templ


#detect bacteria in image. This function needs to be improved so that bacteria size is a parameter
#and not hardcoded
def detect_bacteria(image, rot_template, mask = None):

    #do template matching with all rotated templates
    all_match = np.zeros((len(rot_template),image.shape[0],image.shape[1]))
    for ind in range(len(rot_template)):
        all_match[ind, :,:] = match_template(image,rot_template[ind],pad_input=True)

    #calculate max proj of those images matched with templates at different angles
    #maxarg contains for each pixel the plane index of best match and hence the angle
    maxproj = np.max(all_match,axis=0)
    
    if mask is not None:
        maxproj = maxproj * (1-mask)
    
    return maxproj


def remove_small(image, minsize):
        
    labeled = skimage.morphology.label(image>1)
    regions = skimage.measure.regionprops(labeled)
    indices = np.array([0]+[x.label if (x.area>minsize) else 0 for x in regions])
    mask = indices[labeled]>0
    mask = mask+1
    
    return mask



'''def detect_spots(image, sigmaXY, sigmaZ):
    
    #create filters
    gfilt = sp.make_g_filter(modelsigma=sigmaXY,modelsigmaZ=sigmaZ)
    gfilt_log = sp.make_laplacelog(modelsigma=sigmaXY,modelsigmaZ=sigmaZ)
    
    #pad image and filter it
    im_exo_pd = np.pad(image,((0,0),(0,0),(5,5)),mode = 'mean')
    filtered = sp.spot_filter_convfft(im_exo_pd, gfilt, gfilt_log, alpha = 0.05, loc_max_dist = 2)
    
    #find regions that match the template shape and find local maxima
    matched = match_template(im_exo_pd, gfilt, pad_input=True)
    match_locmax = peak_local_max(matched,min_distance = 2, indices = False,threshold_abs=0.5)

    #recover amplitude and background of maxima in filtered image
    amp_0 = filtered['amplitude'][match_locmax]
    b_0 = filtered['background'][match_locmax]

    #create a dataframe with all spot information. Remove spots too close to the edges
    spot_coord = np.where(match_locmax)
    spot_coord = np.stack(spot_coord).T

    spot_prop = pd.DataFrame(np.c_[spot_coord,amp_0, b_0],columns=['x','y','z','A','b'])
    spot_prop = spot_prop[(spot_prop.x-6>=0)&(spot_prop.x+7<im_exo_pd.shape[0])]
    spot_prop = spot_prop[(spot_prop.y-6>=0)&(spot_prop.y+7<im_exo_pd.shape[1])]
    spot_prop = spot_prop[(spot_prop.z-6>=0)&(spot_prop.z+7<im_exo_pd.shape[2])]
    
    return spot_prop'''


def fit_gaussian_hist(data, plotting = True):

    fitfunc  = lambda p, x: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2)
    errfunc  = lambda p, x, y: (y - fitfunc(p, x))


    ydata, xdata = np.histogram(data, bins=np.arange(0,4000,30))
    xdata = [0.5*(xdata[x]+xdata[x+1]) for x in range(len(xdata)-1)]
    init  = [np.max(ydata), np.mean(data),
             np.std(data)]

    out   = leastsq( errfunc, init, args=(xdata, ydata))
    
    fig=[]
    if plotting == True:
        fig,ax = plt.subplots()
        plt.bar(x = xdata,height=ydata, width=30,color='r')
        plt.plot(xdata, fitfunc(out[0], xdata))
        plt.plot([out[0][1]+3*out[0][2],out[0][1]+3*out[0][2]],[0,np.max(ydata)],'green')
        ax.set_ylabel('Counts')
        ax.set_xlabel('Pixel intensity')
        ax.legend(['Background fit','Threshold','Pixel intensity'])
        plt.show()
        
    return out, fig


def filter_sets(image):
    im_gauss = skimage.filters.gaussian(image,sigma = 10, preserve_range = True)
    im_gauss2 = skimage.filters.gaussian(image,sigma = 20, preserve_range = True)
    im_frangi = skimage.filters.frangi(image)
    im_prewitt = skimage.filters.prewitt(image)
    im_meijering = skimage.filters.meijering(image)
    im_gauss_laplace = skimage.filters.laplace(skimage.filters.gaussian(image,sigma = 5,preserve_range=True),ksize=10)
    im_gradient = skimage.filters.rank.gradient(image,skimage.morphology.disk(8))
    im_entropy = skimage.filters.rank.entropy(image,skimage.morphology.disk(8))
    im_roberts = skimage.filters.roberts(skimage.filters.gaussian(image,sigma = 5,preserve_range=True))
    
    filter_list = [im_gauss,im_gauss2,im_frangi,im_prewitt,im_meijering,
     im_gauss_laplace,im_gradient,im_entropy,im_roberts]
    filter_names = ['Gauss $\sigma$=10', 'Gauss $\sigma$=20','Frangi','Prewitt','Meijering','Gauss+Laplace', 'Gradient',
                'Entropy','Roberts']
    
    return filter_list, filter_names
