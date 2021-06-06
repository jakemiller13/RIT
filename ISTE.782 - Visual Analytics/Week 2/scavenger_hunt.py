import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

def read_waveforms(LA, RV, RA) :
    infile = open('waveforms.csv', 'r')

    line = infile.readline()
    wf = 0 # which waveform are we trying to read (0 = LA, 1 = RV, 2 = RA)

    while line :
        line = line.strip()
        data = line.split(',')

        for i in range(0, len(data)) : 
            data[i] = float(data[i])

        if(wf == 0) :
            LA.append(data)
        elif(wf == 1) :
            RV.append(data)
        elif(wf == 2) :
            RA.append(data)
        
        wf = (wf + 1) % 3
        line = infile.readline()

    infile.close()

def read_times(TL, TR) :
    infile = open('times.csv', 'r')
    line = infile.readline()
    data = line.strip().split(',')

    for i in range(0, len(data)) : 
        data[i] = float(data[i]) * 1000

    TL.append(data)

    line = infile.readline()
    data = line.strip().split(',')

    for i in range(0, len(data)) : 
        data[i] = float(data[i]) * 1000

    TR.append(data)
    infile.close()

def plot_waveforms(LA, RV, RA, TL, TR) :   
    for i in range(len(LA)):
        fig, ax = plt.subplots(3, 1, sharex = True)
        ax[0].plot(TL, LA[i])
        ax[0].set_ylabel('Lin Accel\n(g)')
        ax[1].plot(TR, RV[i])
        ax[1].set_ylabel('Rot Vel\n(rad/sec)')
        ax[2].plot(TR, RA[i])
        ax[2].set_ylabel('Rot Accel\n(rad/sec^2)')
        plt.xlabel('Time (ms)')
        plt.suptitle('Waveforms for Impact {}'.format(i + 1))
        plt.show()
    plt.close('all')

def sum_stats(LA, RV, RA, TL, TR):
    MLA, ALA, PLA = [], [], []
    MRV, ARV, PRV = [], [], []
    MRA, ARA, PRA = [], [], []
    
    for i in range(len(LA)):
        
        MLA.append(np.min(LA[i]))
        ALA.append(np.mean(LA[i]))
        PLA.append(np.max(LA[i]))
        
        MRV.append(np.min(RV[i]))
        ARV.append(np.mean(RV[i]))
        PRV.append(np.max(RV[i]))
        
        MRA.append(np.min(RA[i]))
        ARA.append(np.mean(RA[i]))
        PRA.append(np.max(RA[i]))
    
    return {'MLA': MLA, 'ALA': ALA, 'PLA': PLA,
            'MRV': MRV, 'ARV': ARV, 'PRV': PRV,
            'MRA': MRA, 'ARA': ARA, 'PRA': PRA}
    
def plot_sum_stats(stats):
    for i in stats:
        
        if i == 'MLA' or i == 'ALA' or i == 'PLA':
            ylab = 'Lin Accel (g)'
        if i == 'MRV' or i == 'ARV' or i == 'PRV':
            ylab = 'Rot Vel (rad/sec)'
        if i == 'MRA' or i == 'ARA' or i == 'PRA':
            ylab = 'Rot Accel (rad/sec^2)'
        
        plt.plot(range(len(stats[i])), stats[i])
        plt.xlabel('Instance')
        plt.ylabel(ylab)
        plt.title(i)
        plt.show()
        plt.close()

def plot_vs_stats(combo_list):
    # PLA, ARV, ARA
    for i in combinations(combo_list, 2):
        plt.scatter(combo_list[i[0]], combo_list[i[1]])
        plt.xlabel(i[0])
        plt.ylabel(i[1])
        plt.title('{} vs {}'.format(i[0], i[1]))
        plt.show()
        plt.close()

def find_max(combo_list):
    top_impacts = {}
    for i in combo_list:
        top = np.argsort(combo_list[i])[-5:] + 1
        top_impacts[i] = sorted(top)
        print('{}: {}'.format(i, sorted(top)))
        
    return top_impacts

# make empty data and time Lists
LA_list = []
RV_list = []
RA_list = []
TL_list = []
TR_list = []

read_waveforms(LA_list, RV_list, RA_list)
read_times(TL_list, TR_list)

# convert all data and time lists to numpy arrays for plotting
LA = np.array(LA_list)
RV = np.array(RV_list)
RA = np.array(RA_list)
TL = np.array(TL_list[0])
TR = np.array(TR_list[0])
