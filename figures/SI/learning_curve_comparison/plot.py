#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 22:09:03 2019

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import glob
import pickle
from cycler import cycler
from scipy.stats import kendalltau
from scipy.stats import pearsonr
import pandas as pd

plt.close('all')

FS = 14
LW = 3

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.size'] = FS
rcParams['axes.labelsize'] = FS
rcParams['xtick.labelsize'] = FS
rcParams['ytick.labelsize'] = FS
rcParams['font.sans-serif'] = ['Arial']
upper_lim = 1500

########## LOAD DATA ##########
# Load predictions
lifetime_filename = 'y_test1.csv'
lifetimes = pd.read_csv(lifetime_filename, sep=',', header=None).to_numpy()

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
c1 = default_colors[0]
c2 = default_colors[1]
c3 = default_colors[2]
c4 = default_colors[3]
custom_cycler = (cycler(color=    [c1 , c2, c2, c2, c3, c1, c1, c1, c3]) +
                 cycler(marker=   ['v','^','<','>','o','p','h','8','s']) +
                 cycler(linestyle=['' , '', '', '', '', '', '', '', '']))

domhan_train_mae = np.array([2109.7225052494855, 1849.6594645867638, 1212.585891425674, 635.6415053274644, 736.4823524684366, 982.9295517446604, 815.0153516193024, 182.1740969957656, 39.26671639504283, 963.9705046307255, 278.1233982727113, 809.9918017851779, 214.47717379245182, 373.61617715971505, 782.1409915242443, 652.1409619184559, 541.4208000790744, 914.2647723217043, 653.845237920406, 210.86480457427322, 57.24575120439577, 389.4157655237717, 94.65957430767732, 459.5290549169848, 339.8275453559294, 326.7065238129744, 679.1995509676041, 410.3718431648451, 57.079993411632415, 192.71364213307274, 411.6424010143676, 408.6435024825014, 916.8936753415427, 735.7578409085011, 963.5001409748955, 409.75772249900103, 928.7912480271002, 838.0202255528388, 547.8514411465768, 755.5909467639559, 436.224344293054])
domhan_train_rmse = np.sqrt(np.mean(np.square(domhan_train_mae)))

domhan_heldout_mae = np.array([np.nan, 2493.068296928594, 2094.8197810081306, 586.0209237538261, 1002.8239594685774, 828.0886161707591, 1237.8282455886356, np.nan, 481.6387982920691, 961.4891619522864, 801.9955995836486, 802.0558212579014, 866.0247684749514, 822.6510169945109, 400.66977086687217, 652.1602767031299, 597.1575946486878, 308.92394879579984, 993.737541966869, 600.3325568689738, 468.0030496288503, 219.24430041153883, 431.5080154543924, 511.02414260157826, 309.74864155949746, 436.3664583113754, 9.884396163888573, 422.7439663888581, 467.1424830589698, 445.7571320642128, 166.9446296431381, 457.8987028120321, 429.78596308521577, 468.2807373985792, 446.55785059312376, 1052.703333755956, 855.4357489859956, 290.9851377192449, 517.2176011452538, 406.10507071420625, 859.8587339155433, 663.1606373604295])
domhan_heldout_rmse = np.sqrt(np.nanmean(np.square(domhan_heldout_mae)))
print(domhan_train_rmse, domhan_heldout_rmse)


domhan_train_preds = [91.94136211141023, 93.18051678356767, -197.684101123298, 50.38697403257821, 50.876731770968746, -220.38650031693876, 47.605715685240256, 817.5573088473077, 415.7712919995543, 99.03644872691467, 36.179861577172524, 597.3869754510247, 231.56013570153206, 240.2434211555524, 539.7226220898285, 24.322628626381743, -281.710990505023, 208.67876644594628, 257.77638510306207, 570.1747790710767, 476.0403889270692, 115.313839697023, 45.78359107037038, 590.4490334249813, 444.964786728798, 110.10408031931313, 689.2460445873136, 139.2818015010784, 88.85487465188736, -4.8252002671927965, 372.8114980278332, -92.70346821197333, 36.10305018920719, 26.39549374630571, -92.28794056268988, -407.17686792580866, -74.53453191250247, 196.4422552341619, 23.13301630802914, -220.36804996341772, -227.86734160863028, 131.62066293192328]



klein_train_preds = '2201.45405644 1689.92293845 1075.41615329  619.85156499  926.6970837 901.89631245  994.67052202  749.43245083  342.69752497 1049.28333041 935.17213365  875.76464568  718.18843693  740.64829633  861.68705867 826.42333538  806.27629305 1057.78943942  758.07576455  656.98828478 243.39328872  252.67803542  440.13933993  433.1564084   399.76865432 511.0392272   504.77117734  378.11588469  565.26286836  530.61824059 619.85156499  508.23618453  682.77163116  439.86721104  490.50366343 582.1759733   512.24689864  514.22531902  471.57451814  574.92264446 430.46603782'
klein_train_preds = np.array([float(i) for i in klein_train_preds.split()])
klein_train_true = '2160 1434 1074  870  788  719  857  788  559 1017  870  860  709  731 742  704  617  966  702  616  300  438  444  511  477  483  494  461 489  527  461  468  498  492  520  463  478  459  429  462  487'
klein_train_true = np.array([float(i) for i in klein_train_true.split()])

klein_heldout_preds = '2201.45405644 2201.45405644 1689.92293845  619.85156499 1043.70264552 901.89631245  994.67052202  749.43245083  342.69752497  913.999803 1049.28333041  935.17213365  875.76464568  718.18843693  740.64829633 861.68705867  826.42333538  806.27629305 1057.78943942  758.07576455 656.98828478  410.82764335  284.66850779  180.34422158  564.43498352 672.18290094  403.01423546  511.51792187  474.97178596  619.85156499 619.85156499  970.19634743  905.93374243  543.2604352  479.75393149 395.69699955  688.18813234  886.25763185  651.99292369 801.99061218 745.38218905  740.64829633'
klein_heldout_preds = np.array([float(i) for i in klein_heldout_preds.split()])
klein_heldout_true = '1852 2237 1709  636 1054  880  862  691  534 1014  854  842  917  876 757  703  648  625 1051  651  599  335  480  561  458  485  487  502 513  495  471  509  481  519  499  535  465  499  466  457  429  713'
klein_heldout_true = np.array([float(i) for i in klein_heldout_true.split()])

def format_lifetimes_plot(r):
    """Format the plot with specific axis limits, aspect ratio, tick marks, and annotation.
    Parameters:
        - r (float): The correlation coefficient to be displayed in the plot annotation.
    Returns:
        - None: The function modifies the plot directly and does not return a value.
    Processing Logic:
        - Sets both x and y axis limits to the predefined variable 'upper_lim'.
        - Sets aspect ratio to 'equal' within a box layout for consistent visual scaling.
        - Defines specific tick marks for both axes.
        - Annotates the plot with the correlation coefficient value formatted to two decimal places."""
    plt.xlim([0,upper_lim])
    plt.ylim([0,upper_lim])
    ax0.set_aspect('equal', 'box')
    ax0.set_xticks([0,750,1500])
    ax0.set_yticks([0,750,1500])
    #plt.legend(validation_pol_leg, bbox_to_anchor=(1.01, 0.85))
    plt.annotate('r = {:.2}'.format(r),(75,1250))

def init_plot(ax):
    ax.plot((-100,upper_lim+100),(-100,upper_lim+100), ls='--', c='.3',label='_nolegend_')
    ax.set_prop_cycle(custom_cycler)

fig = plt.figure(figsize=(11,10))
ax0 = plt.subplot(1,2,1)
init_plot(ax0)
plt.scatter(klein_train_true, klein_train_preds)
plt.xlabel('True final cycle lives', fontsize=FS)
plt.ylabel('Predicted final cycle lives', fontsize=FS)
plt.title('Train Set', loc='left', weight='bold', fontsize=FS)
r = pearsonr(klein_train_preds, klein_train_true)[0]
format_lifetimes_plot(r)

ax0 = plt.subplot(1,2,2)
init_plot(ax0)
plt.scatter(klein_heldout_true, klein_heldout_preds)
plt.xlabel('True final cycle lives', fontsize=FS)
plt.ylabel('Predicted final cycle lives', fontsize=FS)
plt.title('Held-Out Set', loc='left', weight='bold', fontsize=FS)
r = pearsonr(klein_heldout_preds, klein_heldout_true)[0]
format_lifetimes_plot(r)


plt.tight_layout()
plt.savefig('learning_curve_comparison.png',bbox_inches='tight')
plt.savefig('learning_curve_comparison.pdf',bbox_inches='tight',format='pdf')

plt.show()
