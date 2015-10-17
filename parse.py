import arrow
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
import operator as op
import pylab
import matplotlib
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

def plot_sec_hist(seconds):
    YBASE = 2; XBASE = 2
    MIN_VAL = 1; MAX_VAL = np.ceil(max(seconds))
    start_power = np.floor(np.log(MIN_VAL) / np.log(XBASE))
    end_power = np.ceil(np.log(MAX_VAL) / np.log(XBASE))
    num_bins = (end_power - start_power) + 1
    bins = np.logspace(start_power, end_power, num_bins, base=XBASE)
    hist = np.histogram(seconds, bins=bins)
    plt.loglog(hist[1][:-1], hist[0], 'x-', basey=YBASE, basex=XBASE)
    plt.show()

def plot_seconds(seconds):
    plt.plot(seconds)
    plt.show()

def mi(x,y, bins=11):
    """Given two arrays x and y of equal length, return their mutual information in bits
    """
    Hxy, xe, ye = pylab.histogram2d(x,y,bins=bins)
    Hx = Hxy.sum(axis=1)
    Hy = Hxy.sum(axis=0)
    Pxy = Hxy/float(x.size)
    Px = Hx/float(x.size)
    Py = Hy/float(x.size)
    pxy = Pxy.ravel()
    px = Px.repeat(Py.size)
    py = pylab.tile(Py, Px.size)
    idx = pylab.find((pxy > 0) & (px > 0) & (py > 0))
    return (pxy[idx]*pylab.log2(pxy[idx]/(px[idx]*py[idx]))).sum()

def takens_embedding(data, tau_max=100, mode="2d"):
    mis = []
    for tau in range(1, tau_max):
        unlagged = data[:-tau]
        lagged = np.roll(data, -tau)[:-tau]
        mis.append(mi(unlagged, lagged))

        if len(mis) > 1 and mis[-2] < mis[-1]: # return first local minima
            tau -= 1
            print tau, mis
            break
    data_lag0 = data[:-2].flatten()
    data_lag1 = np.roll(data, -tau)[:-2].flatten()
    data_lag2 = np.roll(data, -2 * tau)[:-2].flatten()
    if mode =="3d":
        # 3d plot
        figure = pylab.figure()
        axes = Axes3D(figure)
        axes.plot3D(data_lag0, data_lag1, data_lag2, linestyle='-', lw=5, alpha=0.4)
        figure.add_axes(axes)
        pylab.show()
    elif mode == "2d":
        plt.plot(data_lag0, data_lag1)
        plt.show()
    elif mode == "anim":
        fig = plt.figure()
        ax = plt.axes(xlim=(0, 90000), ylim=(0, 90000))
        line, = ax.plot([], [], lw=0.8)
        def init():
            line.set_data([], [])
            return line,
        def animate(i):
            line.set_data(data_lag0[:i], data_lag1[:i])
            return line,
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=data_lag0.size, interval=5, blit=True)
        anim.save("anim_%d.mp4" % data_lag0.size, fps=60)
        plt.show()

def seconds_from_sched(sched_filename):
    seconds = []
    with open(sched_filename) as sched_file:
        sched_lines = sched_file.readlines()
        #sched_lines = sched_lines[:4000]
        print "num lines: ", len(sched_lines)
        curr_date = arrow.get(sched_lines[0].strip()[4:], "MMM D HH:mm:ss YYYY Z")
        i = 0
        for line in sched_lines[1:]:
            i += 1
            if i % 10000 == 0:
                print i
            next_date = arrow.get(line.strip()[4:], "MMM D HH:mm:ss YYYY Z")
            diff = curr_date - next_date
            if diff.days != 0:
                curr_date = next_date
                continue
            seconds.append(diff.seconds)
            curr_date = next_date
    return seconds

def categorize_infelicitous_commits(sched_filename):
    """
    Look at the commits that bypass that inverse dynamic relation
    """
    infelicitous = []
    seconds = seconds_from_sched(sched_filename)
    members = []
    prods = []
    for (idx, first) , second in zip(enumerate(seconds), seconds[1:]):
        members.append((idx, (first, second)))
        prods.append(first * second)
    prods_mean = np.mean(np.array(prods))
    prods_mean = prods_mean * np.log(prods_mean)
    infelicitous = filter(lambda x: x[1][0] * x[1][1] > prods_mean, members)
    return infelicitous

def predict_next_times(seconds):
    members, prods = [], []
    for (idx, first) , second in zip(enumerate(seconds), seconds[1:]):
        members.append((idx, (first, second)))
        prods.append(first * second)
    firsts = [member[1][0] for member in members]
    # try dynamic ratio?
    errors = []
    for idx, tup in members:
        if idx in (0, 1, 2):
            continue
        first, second = tup
###### is there a contingency on your markovianness? sometimes markovian, sometimes inverse?
##### is there enough information to be deciding between the two, to poke at when this thing explodes deterministically?
        second_pred = first
        #second_pred = (prods[idx-1]) / float(first)
        errors.append(np.abs(second_pred - second))
    plt.plot(errors)
    plt.show()
    print errors
    print "error sum: ", sum(errors)

if __name__ == "__main__":
    #with open("/home/curuinor/data/linux_sched") as sched_file:
    seconds = seconds_from_sched("dio_sched")
    predict_next_times(seconds)
    #infelicitous = categorize_infelicitous_commits("dio_sched")
    #print infelicitous
    #plt.scatter(map(op.itemgetter(1), infelicitous), map(op.itemgetter(0), infelicitous))
    #plt.show()
    #seconds = seconds_from_sched("sched")
    #print "finished reading"
    #lagged = np.roll(seconds, -1)
    #plt.scatter(seconds, lagged, s=4, alpha=0.7)
    #plt.show()
    #takens_embedding(np.array(seconds), mode="anim")
