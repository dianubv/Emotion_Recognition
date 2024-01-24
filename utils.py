import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import math
import os


parent_path = '/home/diane/Documents/Projects/ER/projects_from_others/EPiC-2023-main'


def Average(lst : list) -> float:
    """ Calculate the average of a list"""
    return sum(lst) / len(lst)


def to_categorical(y : np.ndarray) -> np.ndarray:
    """ 1-hot encodes a tensor """
    num_classes = len(np.unique(y))     # The number of unique values in the tensor
    return np.eye(num_classes, dtype='uint8')[y.astype(int)]


class WeightClipper(object):
    """ Clips the weights of the network to be in the range [-1, 1] """
    def __init__(self, frequency=5) -> None:
        self.frequency = frequency

    def __call__(self, module : nn.Module) -> None:
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1,1)

class WeightInit(object):
    """ Initialize the weights of the network using a normal distribution """
    def __init__(self, frequency=5) -> None:
        self.frequency = frequency

    def __call__(self, module : nn.Module) -> None:
        # filter the variables to get the ones you want
        torch.manual_seed(0)
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = nn.init.normal_(w, 0.0, 0.02)


def adjust_learning_rate(args, optimizer, epoch : int) -> None:
    #  Learning rate scheduling helps control the learning rate at different stages of training to improve convergence and generalization
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# load_dataset_to_device becomes ini_data

def unique(sequence : list) -> list:
    """ Return a list of unique items in a sequence"""
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def min_max_scale(label : np.ndarray) -> np.ndarray:
    """ Scale the label to range [0, 1]"""
    return np.asarray([(x - 0.5) / (9.5 - 0.5) for x in label])


def min_max_inverse_scale(scaled_label : np.ndarray) -> np.ndarray:
    """ Inverse scale the label to the orignal scale [0, 10]"""
    return np.asarray([x  * (9.5 - 0.5) + 0.5 for x in scaled_label])


def rmse(y_pred : np.array, y_true : np.array) -> float:
    """ Calculate the Root Mean Squared Error between predicted and true arrays"""
    mse = np.mean((y_pred - y_true)**2)
    rmse = np.sqrt(mse)
    return rmse


def make_dir(path : str):
    """ Create a directory if it does not exist"""
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass



# check format and details of final annotations files 




def write_result(emotion_name : str, prediction_label, destination_file : str) -> None:
    """ Write the prediction label to the destination file"""
    destination = pd.read_csv(destination_file)
    destination = np.array(destination)
    print(len(prediction_label))

    assert len(prediction_label) == len(destination)

    if emotion_name == 'valence': 
        destination[:, 1] = prediction_label
    elif emotion_name == 'arousal':
        destination[:, 2] = prediction_label
    elif emotion_name == 'dominance':
        destination[:, 3] = prediction_label

    updated_df = pd.DataFrame(destination, columns=['time', 'valence', 'arousal'])
    updated_df.to_csv(destination_file, index=False)




def set_permissions_recursive(path : str) -> None:
    """ Set the permissions of all files and folders recursively to 700"""
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o600)
        for f in files:
            os.chmod(os.path.join(root, f), 0o700)



