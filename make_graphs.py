import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_fn(data, title, ylabel, labels):
    for fold_data, i in zip(data,range(len(data))):
        label=labels[i]
        plt.plot(fold_data, label=label)
    plt.ylabel(ylabel)
    plt.xlabel("Epoch")
    plt.title(title)
    plt.legend()
    plt.show()

def average_arrays(arrays):
    average = arrays[0]
    for i in range(1, len(arrays)):
        for j in range(len(arrays[i])):
            average[j] += arrays[i][j]
    for i in range(len(average)):
        average[i] = average[i] / len(arrays)
    return average

def plot_stat(data, stat, conditions):
    plot_data = []
    for condition in conditions:
        condition_data = []
        for fold in data[condition]:
            fold_data = []
            for epoch in data[condition][fold]:
                fold_data.append( data[condition][fold][epoch][stat.lower()])
            condition_data.append(fold_data)
        condition_data = average_arrays(condition_data)
        plot_data.append(condition_data)
    
    plot_fn(plot_data, stat, stat, conditions)

def plot_substat(data, condition, stat, substat):
    plot_data = []
    for fold in data[condition]:
        fold_data = []
        for epoch in data[condition][fold]:
            fold_data.append( data[condition][fold][epoch][stat.lower()][substat.lower()])
        plot_data.append(fold_data)
    
    plot_fn(plot_data, condition, stat+substat)

def plot_acc(data, condition, stats):
    stats_data = []

    stat_data = []
    for split in data[condition]:
        split_data = []
        for epoch in data[condition][split]:
            split_data.append( data[condition][split][epoch]["type"])
        stat_data.append(split_data)
    stats_data.append(average_arrays(stat_data))

    for stat in stats:
        stat_data = []
        for split in data[condition]:
            split_data = []
            for epoch in data[condition][split]:
                split_data.append( data[condition][split][epoch][stat]["type"])
            stat_data.append(split_data)
        stats_data.append(average_arrays(stat_data))
    
    plt.plot(stats_data[0], label="Total Accuracy", color="black")

    for i in range(1,len(stats_data)):
        label=stats[i-1]
        plt.plot(stats_data[i], label=label, alpha=0.5)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.title("Accuracy for " + condition + " data")
    plt.legend()
    plt.show()


def plot_confusion(data):
    plot_data = None
    for split in data["test"]:
        confusion_matrix = data["test"][split]["0"]["confusion_matrix"]
        plot_data = confusion_matrix

    row_labels = []
    column_labels = []
    for i in ["tm", "sptm", "sp", "glob", "beta", "Topology"]: 
        row_labels.append(str(i))
        column_labels.append(str(i))

    fig, ax = plt.subplots(figsize=(11.5,8))
    sns.heatmap(plot_data, ax=ax, annot=True, xticklabels=column_labels, yticklabels=row_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()




if __name__ == "__main__":
    f = open("stat_data_300.json")
    data = json.load(f)

    conditions = ["train", "val"]
    #Plot loss average across splits/folds
    #plot_stat(data, "loss", conditions)

    #Plot total and each accuracy avarged across splits/folds
    # Train
    #plot_acc(data, "train", ["tm", "sptm", "sp", "glob", "beta"])
    # Validation
    #plot_acc(data, "val", ["tm", "sptm", "sp", "glob", "beta"])

    # Plot confusion matrix as table 
    plot_confusion(data)
    