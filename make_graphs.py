import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from cycler import cycler

def plot_fn(data, conf_int, early_stops, title, ylabel, labels):
    for fold_data, conf_data, i in zip(data, conf_int, range(len(data))):
        label=labels[i]
        x = np.arange(len(fold_data))
        plt.plot(fold_data, label=label) # label for the best model    
        plt.fill_between(x, (fold_data - conf_data), (fold_data + conf_data), alpha = .1)
    plt.vlines(early_stops, ymin = 0, ymax = np.max(fold_data), linestyles = "dashed", color = "red", label = "early stop", alpha = .3)
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
    condition_loss = []
    condition_conf = []
    early_stops = []
    for _, condition in enumerate(conditions):
        avg_epoch_loss = []
        conf_epoch = []
        
        for epoch in range(100):
            epoch_loss_per_fold = []

            for fold in data[condition]:
                if str(epoch) in data[condition][fold].keys():
                    epoch_loss_per_fold.append(data[condition][fold][str(epoch)][stat.lower()])

            if epoch_loss_per_fold:
                avg_epoch_loss.append(np.mean(epoch_loss_per_fold))
                conf_epoch.append(1.96 * np.std(epoch_loss_per_fold) / np.sqrt(len(epoch_loss_per_fold)))
        
           
        condition_loss.append(avg_epoch_loss)   
        condition_conf.append(conf_epoch) 

    for fold in data[condition]:
        early_stops.append(np.argmin([data["val"][fold][epochs][stat.lower()] for epochs in data["val"][fold].keys()]))
             
    plot_fn(np.asarray(condition_loss), np.asarray(condition_conf), np.asarray(early_stops), stat, stat, conditions)

def plot_substat(data, condition, stat, substat):
    plot_data = []
    for fold in data[condition]:
        fold_data = []
        for epoch in data[condition][fold]:
            fold_data.append( data[condition][fold][epoch][stat.lower()][substat.lower()])
        plot_data.append(fold_data)
    
    plot_fn(plot_data, condition, stat+substat)

def plot_acc(data, condition, stats):
    condition_acc = []
    condition_conf = []
    early_stops = []
    for _, condition in enumerate(conditions):
        avg_epoch_acc = np.zeros((2, 100, 5))
        conf_epoch_acc = np.zeros((2, 100, 5))
        
        for epoch in range(100):
            epoch_acc_type_per_fold = [[], [], [], [], []]
            epoch_acc_top_per_fold = [[], [], [], [], []]

            for fold in data[condition]:
                count = 0
                if str(epoch) in data[condition][fold].keys():
                    for i, stat in enumerate(stats):
                        epoch_acc_type_per_fold[i].append(data[condition][fold][str(epoch)][stat.lower()]["type"])
                        epoch_acc_top_per_fold[i].append(data[condition][fold][str(epoch)][stat.lower()]["topology"])
                        count += 1

            if epoch_acc_type_per_fold:
                avg_epoch_acc[0, epoch, :] = np.mean(epoch_acc_type_per_fold, axis = 1)
                avg_epoch_acc[1, epoch, :] = np.mean(epoch_acc_top_per_fold, axis = 1)
                
                conf_epoch_acc[0, epoch, :] = (1.96 * np.std(epoch_acc_type_per_fold, axis = 1) / np.sqrt(len(epoch_acc_type_per_fold)))
                conf_epoch_acc[1, epoch, :] = (1.96 * np.std(epoch_acc_top_per_fold, axis = 1) / np.sqrt(len(epoch_acc_top_per_fold)))
        
        nans_acc = avg_epoch_acc[~np.isnan(avg_epoch_acc)].shape[0]
        nans_conf = avg_epoch_acc[~np.isnan(avg_epoch_acc)].shape[0]
        
        avg_epoc_acc = avg_epoch_acc[~np.isnan(avg_epoch_acc)].reshape((2, nans_acc//(2*5), 5))
        conf_epoch_acc = conf_epoch_acc[~np.isnan(avg_epoch_acc)].reshape((2, nans_conf//(2*5), 5))
        
        condition_acc.append(avg_epoc_acc)   
        condition_conf.append(conf_epoch_acc) 

    for fold in data[condition]:
        early_stops.append(np.argmin([data["val"][fold][epochs]["loss"] for epochs in data["val"][fold].keys()]))
        
    # plot_fn(condition_acc, np.asarray(condition_conf), np.asarray(early_stops), stat, stat, conditions)

    titles = ["type", "topology"]
    labels = ["train", "val"]
    colors = ["darkred", "darkorange", "darkgreen", "darkviolet", "darkblue"]
    for i, train_val in enumerate(condition_acc):
        for j, types in enumerate(train_val):
            plt.subplot(1, 2, j + 1)
            plt.gca().set_prop_cycle(cycler('color', colors))
            x = np.arange(types.shape[0])
            plt.plot(types, label = [stat + f"_{labels[i]}" for stat in stats], linestyle = "dashed" if labels[i] == "val" else "solid")
            
            for k, line in enumerate(types.T):
                l_bound =  (line - condition_conf[i][j, :, k])
                l_bound[(line - condition_conf[i][j, :, k]) < 0] = 0
                u_bound =  (line + condition_conf[i][j, :, k])
                u_bound[(line + condition_conf[i][j, :, k]) > 1] = 1
                
                plt.fill_between(x, l_bound, u_bound, alpha = .1)
            
            if i == 1:
                plt.ylabel("accuracy")
                plt.xlabel("epoch")
                plt.title(f"{titles[j]} accuracy")
                plt.legend(loc = "lower left")
                plt.vlines(early_stops, ymin = 0, ymax = np.max(np.max(train_val, axis = 0)), linestyles = "dashed", color = "red", label = "early stop", alpha = .3)
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
    f = open("stat_data_512.json")
    data = json.load(f)

    conditions = ["train", "val"]
    #Plot loss average across splits/folds
    # plot_stat(data, "loss", conditions)

    #Plot total and each accuracy avarged across splits/folds
    # Train
    plot_acc(data, "train", ["tm", "sptm", "sp", "glob", "beta"])
    # Validation
    #plot_acc(data, "val", ["tm", "sptm", "sp", "glob", "beta"])

    # Plot confusion matrix as table 
    # plot_confusion(data)
    