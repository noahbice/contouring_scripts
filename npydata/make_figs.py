import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

structs = ['Spinal Cord', 'Neck Right', 'Neck Left', \
        'Submandibular Gland Right', 'Submandibular Gland Left', 'Parotid Right', \
        'Parotid Left', 'Oral Cavity', 'Medulla Oblongata', 'Brain']
struct_dict = {structs[i]: i for i in range(len(structs))}

dice = np.load('dice.npy')
counts = np.load('count.npy')
index = np.linspace(0., 1., 100)

#-------------------------
#plotting options
make_dice = True #dice by threshold

make_counts = True #optimum dice by fraction of trues
regression = True #linear regression on above?

make_deviation = True #change in dice for +/- 0.01 to thresh
mode = 'boxplot' #boxplot or histogram
deviation = 2 #how many hundreths
#-------------------------

if make_dice:
    for i in range(dice.shape[0]):
        plt.plot(index, dice[i], label=structs[i])
    plt.legend(fontsize='small')
    plt.xlabel('Threshold')
    plt.ylabel('Dice Score')
    plt.title('Validation Dice by Threshold')
    plt.savefig('dice_by_thresh.png')
    plt.show()

if make_counts:
    opt_threshes = []
    for i in range(dice.shape[0]):
        idx = np.where(dice[i] == np.amax(dice[i]))
        opt_threshes.append(index[idx][0])
        plt.scatter([counts[i]],[index[idx]], label=structs[i])
    opt_threshes = np.array(opt_threshes)
    if regression:
        m, b, R, p, _ = linregress(counts, opt_threshes)
        R2 = R**2
        x = np.linspace(0,0.1, 10)
        y = m*x + b
        plt.plot(x, y, lw=0.75, \
            label = '$y = {}*x + ${}'.format(np.round(m, decimals=2), np.round(b, decimals=2)))
        plt.text(0., 0.3, '$R^2 = ${}'.format(np.round(R2, decimals=2)))
        plt.text(0., 0.28, '$p = ${}'.format('2e-7'))
    plt.legend(fontsize='small', loc='lower right')
    plt.xlabel('Fraction \"True\" in Masks')
    plt.ylabel('Optimum Threshold')
    plt.title('Optimum Threshold by Dataset')
    plt.savefig('opt_thresh.png')
    plt.show()
    
if make_deviation: 
    absolute_deviations = []
    for i in range(dice.shape[0]):
        idx = np.where(dice[i] == np.amax(dice[i]))
        minus = np.abs(dice[i][idx] - dice[i][idx[0]-deviation])
        absolute_deviations.append(minus)
        plus = np.abs(dice[i][idx] - dice[i][idx[0]+deviation])
        absolute_deviations.append(plus)
    absolute_deviations = np.array(absolute_deviations)
    avg = np.median(absolute_deviations)
    if mode == 'boxplot':
        plt.boxplot(absolute_deviations, vert=False)
        plt.xlabel('Median absolute deviation: {}'.format(np.round(avg, decimals=3)))
        plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    if mode == 'histogram':
        plt.hist(absolute_deviations)
        plt.text(0.3, 14, 'Median absolute deviation: {}'.format(np.round(avg, decimals=3)))
        plt.xlabel('Dice Deviation')
        plt.ylabel('Frequency')
    plt.title('Dice Deviation for $\pm${} Hundredths to Threshold'.format(deviation))
    plt.savefig('deviations.png')
    plt.show()
        