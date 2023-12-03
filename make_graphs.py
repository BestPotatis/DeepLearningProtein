import json
import matplotlib.pyplot as plt

def plot_fn(data, title, ylabel):
    for fold_data, i in zip(data,range(len(data))):
        label="Fold " + str(i+1)
        plt.plot(fold_data, label=label)
    plt.ylabel(ylabel)
    plt.xlabel("Epoch")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_stat(data, condition, stat):
    plot_data = []
    for fold in data[condition]:
        fold_data = []
        for epoch in data[condition][fold]:
            fold_data.append( data[condition][fold][epoch][stat.lower()])
        plot_data.append(fold_data)
    
    plot_fn(plot_data, condition, stat)

def plot_substat(data, condition, stat, substat):
    plot_data = []
    for fold in data[condition]:
        fold_data = []
        for epoch in data[condition][fold]:
            fold_data.append( data[condition][fold][epoch][stat.lower()][substat.lower()])
        plot_data.append(fold_data)
    
    plot_fn(plot_data, condition, stat+substat)

def plot_confusion(data, condition):
    plot_data = []
    for fold in data[condition]:
        fold_data = []
        for epoch in data[condition][fold]:
            confusion_matrix = data[condition][fold][epoch]["confusion_matrix"]
            # confusion_matrix[actual_type][predicted]
            TP = [0] * 6
            FN = [0] * 6
            for actual in range(len(confusion_matrix)):
                for predicted in range(len(confusion_matrix[actual])):
                    if actual == predicted:
                        TP[predicted] += confusion_matrix[actual][predicted]
                    else:
                        FN[predicted] += confusion_matrix[actual][predicted]
            fold_data.append(TP[0])
        plot_data.append(fold_data)

    plot_fn(plot_data, condition, "True positives")

def sum_confusion_matrix(data, condition):
    sum_confusion_matrix = [[0 for x in range(6)] for y in range(6)] 
    for fold in data[condition]:
        for epoch in data[condition][fold]:
            confusion_matrix = data[condition][fold][epoch]["confusion_matrix"]
            for actual in range(len(confusion_matrix)):
                for predicted in range(len(confusion_matrix[actual])):
                    sum_confusion_matrix[actual][predicted] += confusion_matrix[actual][predicted]
    print(sum_confusion_matrix)


if __name__ == "__main__":
    f = open("stat_data_300.json")
    data = json.load(f)

    #plot_stat(data, "train", "Loss")
    #plot_stat(data, "val", "Loss")
    sum_confusion_matrix(data, "train")
    