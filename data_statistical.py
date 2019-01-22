import matplotlib
from matplotlib import pyplot as plt
import numpy as np


def plot_samples(sample_file):

    samples_time = []
    with open(sample_file, 'r') as f:
        for line in f:
            travel_time = int(line.strip().rsplit(' ', 1)[1])
            samples_time.append(travel_time)
    array = np.array(samples_time)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.hist(array, bins=65, density=False, facecolor="blue", edgecolor="black", alpha=0.7, range=(5, 1200))
    plt.xlabel("区间")
    plt.ylabel("频数/频率")
    plt.title("频数/频率分布直方图")
    plt.show()


if __name__ == '__main__':

    plot_samples('sanfrancisco/node/sf_travel_time_21.samples')
