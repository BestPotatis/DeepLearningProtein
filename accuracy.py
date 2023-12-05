'''
metric_utils.py
===============

Code provided by Jeppe Hallgren.

All functions were written in pytorch 1.5 - best if you check whether 
there are any changes/warnings that you should consider for pytorch 2.0+.
'''
import torch
from typing import List, Union, Dict
import json
from collections import defaultdict
import os
import numpy as np

# The following are the label mapping is used in the metrics.
LABELS: Dict[str,int] = {'I': 0, 'O':1, 'P': 2, 'S': 3, 'M':4, 'B': 5}

def update_nested_dict(d, keys, value):
    for idx, key in enumerate(keys[:-1]):
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value

def custom_encoder(obj):
    if 'confusion_matrix' in obj:
        return json.dumps(obj, separators=(',', ':'), indent=4)
    else:
        return None

def type_from_labels(label):
    """
    Function that determines the protein type from labels

    Dimension of each label:
    (len_of_longenst_protein_in_batch)

    # Residue class
    0 = inside cell/cytosol (I)
    1 = Outside cell/lumen of ER/Golgi/lysosomes (O)
    2 = beta membrane (B)
    3 = signal peptide (S)
    4 = alpha membrane (M)
    5 = periplasm (P)

    B in the label sequence -> beta
    I only -> globular
    Both S and M -> SP + alpha(TM)
    M -> alpha(TM)
    S -> signal peptide

    # Protein type class
    0 = TM
    1 = SP + TM
    2 = SP
    3 = GLOBULAR
    4 = BETA
    """

    if 2 in label:
        ptype = 4

    elif all(element == 0 for element in label):
        ptype = 3

    elif 3 in label and 4 in label:
        ptype = 1

    elif 3 in label:
       ptype = 2

    elif 4 in label:
        ptype = 0

    elif all(x == 0 or x == -1 for x in label):
        ptype = 3

    else:
        ptype = None

    return ptype

def label_list_to_topology(labels: Union[List[int], torch.Tensor]) -> List[torch.Tensor]:
    """
    Converts a list of per-position labels to a topology representation.
    This maps every sequence to list of where each new symbol start (the topology), e.g. AAABBBBCCC -> [(0,A),(3, B)(7,C)]

    Parameters
    ----------
    labels : list or torch.Tensor of ints
        List of labels.

    Returns
    -------
    list of torch.Tensor
        List of tensors that represents the topology.
    """

    if isinstance(labels, list):
        labels = torch.LongTensor(labels)

    if isinstance(labels, torch.Tensor):
        zero_tensor = torch.LongTensor([0])
        if labels.is_cuda:
            zero_tensor = zero_tensor.cuda()

        unique, count = torch.unique_consecutive(labels, return_counts=True)
        top_list = [torch.cat((zero_tensor, labels[0:1]))]
        prev_count = 0
        i = 0
        for _ in unique.split(1):
            if i == 0:
                i += 1
                continue
            prev_count += count[i - 1]
            top_list.append(torch.cat((prev_count.view(1), unique[i].view(1))))
            i += 1
        return top_list


def is_topologies_equal(topology_a, topology_b, minimum_seqment_overlap=5):
    """
    Checks whether two topologies are equal.
    E.g. [(0,A),(3, B)(7,C)]  is the same as [(0,A),(4, B)(7,C)]
    But not the same as [(0,A),(3, C)(7,B)]

    Parameters
    ----------
    topology_a : list of torch.Tensor
        First topology. See label_list_to_topology.
    topology_b : list of torch.Tensor
        Second topology. See label_list_to_topology.
    minimum_seqment_overlap : int
        Minimum overlap between two segments to be considered equal.

    Returns
    -------
    bool
        True if topologies are equal, False otherwise.
    """

    if isinstance(topology_a[0], torch.Tensor):
        topology_a = list([a.cpu().numpy() for a in topology_a])
    if isinstance(topology_b[0], torch.Tensor):
        topology_b = list([b.cpu().numpy() for b in topology_b])
    if len(topology_a) != len(topology_b):
        return False
    for idx, (_position_a, label_a) in enumerate(topology_a):
        if label_a != topology_b[idx][1]:
            if (label_a in (1,2) and topology_b[idx][1] in (1,2)): # assume O == P
                continue
            else:
                return False
        if label_a in (3, 4, 5):
            if idx == (len(topology_a) - 1): # it's impossible to end in 3, 4 or 5
                return False
            else:
                overlap_segment_start = max(topology_a[idx][0], topology_b[idx][0])
                overlap_segment_end = min(topology_a[idx + 1][0], topology_b[idx + 1][0])
                    
            if label_a == 5:
                # Set minimum segment overlap to 3 for Beta regions
                minimum_seqment_overlap = 3
            if overlap_segment_end - overlap_segment_start < minimum_seqment_overlap:
                return False
    return True

def calculate_acc(correct, total):
    total = total.float()
    correct = correct.float()
    if total == 0.0:
        return torch.tensor(1)
    return correct / total

def test_predictions(model, loader, loss_function, cv, experiment_file_path, device, condition = "test", epoch = 0):
    # either loads an empty dict or the dict of the previous fold 
    experiment_json = json.loads(open(experiment_file_path, 'r').read())

    confusion_matrix = torch.zeros((6, 6), dtype=torch.int64)
    protein_names = []
    protein_label_actual = []
    protein_label_prediction = []
    
    with torch.no_grad():
        model.eval()
        test_batch_loss = []
        
        for _, batch in enumerate(loader):
            inputs, labels, lengths = batch['data'], batch['labels'], torch.tensor(batch['lengths'])
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
            
            output = model(inputs)
            
            predict_label = []
            predict_prot_type = []
            ground_truth_label = []
            ground_truth_prot_type = [] 
            
            loss = 0
            for l in range(output.shape[0]):
                # masking the zero-padded outputs
                batch_output = output[l][:lengths[l]]
                batch_labels = labels[l][:lengths[l]]
                
                # compute cross-entropy loss
                loss += loss_function(batch_output, batch_labels) * len(inputs)
                
                # predict labels and type for the masked outputs
                predictions_batch_mask = batch_output.max(-1)[1]
                predict_prot_type_batch = type_from_labels(predictions_batch_mask)
                ground_truth_prot_type_batch = type_from_labels(batch_labels)
                
                predict_label.append(predictions_batch_mask)
                ground_truth_label.append(batch_labels)
                predict_prot_type.append(predict_prot_type_batch)
                ground_truth_prot_type.append(ground_truth_prot_type_batch)
                
                # used later for the accuracy computation
                protein_names.extend(ground_truth_prot_type)
                protein_label_actual.extend(ground_truth_label)
            
            # go over each protein and compute accuracy
            for idx, actual_type in enumerate(ground_truth_prot_type):
                predicted_type = predict_prot_type[idx]
                predicted_topology = label_list_to_topology(predict_label[idx])
                predicted_labels_for_protein = predict_label[idx]
                
                # convert output and label to topology sequences    
                ground_truth_topology = label_list_to_topology(ground_truth_label[idx])
                
                prediction_topology_match = is_topologies_equal(ground_truth_topology, predicted_topology, 5)
                
                if actual_type == predicted_type:
                    # if we guessed the type right for SP+GLOB or GLOB,
                    # count the topology as correct
                    if actual_type == 2 or actual_type == 3 or prediction_topology_match:
                        confusion_matrix[actual_type][5] += 1
                    else:
                        confusion_matrix[actual_type][predicted_type] += 1

                else:
                    confusion_matrix[actual_type][predicted_type] += 1
                
                protein_label_prediction.append(predicted_labels_for_protein)
                
                
            # receive loss from current batch
            test_batch_loss.append(loss.item())
        
    model.train()
    
    type_correct_ratio = \
    calculate_acc(confusion_matrix[0][0] + confusion_matrix[0][5], confusion_matrix[0].sum()) + \
    calculate_acc(confusion_matrix[1][1] + confusion_matrix[1][5], confusion_matrix[1].sum()) + \
    calculate_acc(confusion_matrix[2][2] + confusion_matrix[2][5], confusion_matrix[2].sum()) + \
    calculate_acc(confusion_matrix[3][3] + confusion_matrix[3][5], confusion_matrix[3].sum()) + \
    calculate_acc(confusion_matrix[4][4] + confusion_matrix[4][5], confusion_matrix[4].sum())
    type_accuracy = float((type_correct_ratio / 5).detach())

    tm_accuracy = float(calculate_acc(confusion_matrix[0][5], confusion_matrix[0].sum()).detach())
    sptm_accuracy = float(calculate_acc(confusion_matrix[1][5], confusion_matrix[1].sum()).detach())
    sp_accuracy = float(calculate_acc(confusion_matrix[2][5], confusion_matrix[2].sum()).detach())
    glob_accuracy = float(calculate_acc(confusion_matrix[3][5], confusion_matrix[3].sum()).detach())
    beta_accuracy = float(calculate_acc(confusion_matrix[4][5], confusion_matrix[4].sum()).detach())
    
    tm_type_acc = float(calculate_acc(confusion_matrix[0][0] + confusion_matrix[0][5], confusion_matrix[0].sum()).detach())
    tm_sp_type_acc = float(calculate_acc(confusion_matrix[1][1] + confusion_matrix[1][5], confusion_matrix[1].sum()).detach())
    sp_type_acc = float(calculate_acc(confusion_matrix[2][2] + confusion_matrix[2][5], confusion_matrix[2].sum()).detach())
    glob_type_acc = float(calculate_acc(confusion_matrix[3][3] + confusion_matrix[3][5], confusion_matrix[3].sum()).detach())
    beta_type_acc = float(calculate_acc(confusion_matrix[4][4] + confusion_matrix[4][5], confusion_matrix[4].sum()).detach())
    
    # add data to dictionary
    key_list = [condition, cv, epoch, 'confusion_matrix']
    update_nested_dict(experiment_json, key_list, confusion_matrix.tolist())
    
    experiment_json[condition][cv][epoch].update({
        'type': type_accuracy
    })
    
    # Topology 
    experiment_json[condition][cv][epoch].update({
        'tm': {
            'type': tm_type_acc,
            'topology': tm_accuracy
        }
    })
    
    experiment_json[condition][cv][epoch].update({
        'sptm': {
            'type': tm_sp_type_acc,
            'topology': sptm_accuracy
        }
    })
    
    experiment_json[condition][cv][epoch].update({
        'sp': {
            'type': sp_type_acc,
            'topology': sp_accuracy
        }
    })
    
    experiment_json[condition][cv][epoch].update({
        'glob': {
            'type': glob_type_acc,
            'topology': glob_accuracy
        }
    })
    
    experiment_json[condition][cv][epoch].update({
        'beta': {
            'type': beta_type_acc,
            'topology': beta_accuracy
        }
    })
    
    experiment_json[condition][cv][epoch].update({
        'loss': np.mean(test_batch_loss)
    })
    
    if condition == "test":
        experiment_json[condition][cv].update({
            "hyperparameter": model.hidden_size
        })
    
    open(experiment_file_path, 'w').write(json.dumps(experiment_json, indent = 4, default = custom_encoder))
      
    return np.mean(test_batch_loss)
    
    # return (protein_names, protein_label_actual, protein_label_prediction, np.mean(test_batch_loss), experiment_json[condition][cv][epoch]['loss']) 