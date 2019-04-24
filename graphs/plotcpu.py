import matplotlib.pyplot as plt
import numpy as np

def plot_cpu(data):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    t = np.arange(0.01, len(data[0]), 1)
    
    ax1 = plt.subplot(511)
    plt.plot(t, data[0]) # container 0
    # make these tick labels invisible
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    # share x only
    ax2 = plt.subplot(512, sharex=ax1)
    plt.plot(t, data[1]) # container 1
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    ax3 = plt.subplot(513, sharex=ax1)
    plt.plot(t, data[2]) # cotainer 2
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax4 = plt.subplot(514, sharex=ax1)
    plt.plot(t, data[3]) # cotainer 3
    plt.setp(ax4.get_xticklabels(), visible=False)
    ax5 = plt.subplot(515, sharex=ax1)
    plt.plot(t, data[4]) # cotainer 4
    
    plt.show()


