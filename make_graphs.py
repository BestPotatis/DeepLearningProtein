import json
import matplotlib.pyplot as plt

def plot_stat(data, condition, stat):
    plot_data = []
    for fold in data[condition]:
        fold_data = []
        for epoch in data[condition][fold]:
            fold_data.append( data[condition][fold][epoch][stat.lower()])
        plot_data.append(fold_data)
    
    for fold_data, i in zip(plot_data,range(len(plot_data))):
        label="Fold " + str(i+1)
        plt.plot(fold_data, label=label)
    plt.ylabel(stat)
    plt.xlabel("Epoch")
    plt.title(condition)
    plt.legend()
    plt.show()

def plot_substat(data, condition, stat, substat):
    plot_data = []
    for fold in data[condition]:
        fold_data = []
        for epoch in data[condition][fold]:
            fold_data.append( data[condition][fold][epoch][stat.lower()][substat.lower()])
        plot_data.append(fold_data)
    
    for fold_data, i in zip(plot_data,range(len(plot_data))):
        label="Fold " + str(i+1)
        plt.plot(fold_data, label=label)
    plt.ylabel(stat + substat)
    plt.xlabel("Epoch")
    plt.title(condition)
    plt.legend()
    plt.show()

def plot_confusion(data, condition):
    for fold in data[condition]:
        for epoch in data[condition][fold]:
            confusion_matrix = data[condition][fold][epoch]["confusion_matrix"]


if __name__ == "__main__":
    f = open("stat_data_300.json")
    data = json.load(f)

    #plot_stat(data, "train", "Loss")
    #plot_stat(data, "val", "Loss")
    plot_substat(data, "train", "tm", "topology")
    