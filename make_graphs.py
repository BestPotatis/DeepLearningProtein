import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from cycler import cycler

def plot_fn(data, conf_int, early_stops, title, ylabel, labels):
    plt.figure(figsize=(8, 4))
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

    for fold in data["val"]:
        early_stops.append(np.argmax([data["val"][fold][epochs]["topology"] for epochs in data["val"][fold].keys()]))
             
    plot_fn(np.asarray(condition_loss), np.asarray(condition_conf), np.asarray(early_stops), stat, stat, conditions)


def plot_acc(data, conditions, stats):
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
        early_stops.append(np.argmax([data["val"][fold][epochs]["topology"] for epochs in data["val"][fold].keys()]))
     
    titles = ["type", "topology"]
    labels = ["train", "val"]
    colors = ["darkred", "darkorange", "darkgreen", "darkviolet", "darkblue"]
    plt.figure(figsize=(12, 7))
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
    plot_data = np.zeros((5,7))
    for split in data["test"]:
        confusion_matrix = data["test"][split]["0"]["confusion_matrix"]
        plot_data += confusion_matrix
    
    #plot_data = plot_data // len(data["test"].keys())
    
    # normalize row-wise
    plot_data = plot_data / (np.sum(plot_data, axis = 1, keepdims = True) + 1e-9)

    row_labels = []
    column_labels = []
    for i in ["tm", "sptm", "sp+glob", "glob", "beta", "topology", "invalid"]: 
        row_labels.append(str(i) if i != "topology" and i != "invalid" else None)
        column_labels.append(str(i))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(plot_data, ax=ax, annot=True, xticklabels=column_labels, yticklabels=row_labels, fmt='.3g', robust = True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


if __name__ == "__main__":
    f = open("stat_data_hs_512_l_2.json")
    data = json.load(f)

    conditions = ["train", "val"]
    #Plot loss average across splits/folds
    #plot_stat(data, "loss", conditions)

    #Plot total and each accuracy avarged across splits/folds
    #plot_acc(data, conditions, ["tm", "sptm", "sp", "glob", "beta"])

    # Plot confusion matrix as table 
    plot_confusion(data)
    