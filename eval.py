# First, let's define a function to load the model checkpoints and gather the validation accuracies.

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import fnmatch

def load_model_checkpoints(folder_path, activation_fns, repeats):
    """Load model checkpoints and gather validation accuracies.

    Args:
        folder_path (str): Path to the folder where the checkpoints are saved.
        activation_fns (list): List of activation function names as strings.
        repeats (int): Number of repeats for each activation function.

    Returns:
        dict: A dictionary with activation function names as keys and lists of accuracies as values.
    """
    all_accuracies = {act_fn: [] for act_fn in activation_fns}

    for act_fn in activation_fns:
        for i in range(1, repeats + 1):
            file_name_pattern = f"{i}_ckpt_*_{act_fn}_*.pth"
            for file_name in os.listdir(folder_path):
                if fnmatch.fnmatch(file_name, file_name_pattern):
                    checkpoint_path = os.path.join(folder_path, file_name)
                    checkpoint = torch.load(checkpoint_path)
                    val_accuracy = checkpoint['top1']  # Assuming 'top1' is the key for validation accuracy.
                    all_accuracies[act_fn].append(val_accuracy)
    
    return all_accuracies

# Next, let's create a boxplot to compare the performances.

def plot_accuracies_boxplot(accuracies_dict):
    """Plot a boxplot of the accuracies for each activation function.

    Args:
        accuracies_dict (dict): A dictionary with activation function names as keys and lists of accuracies as values.
    """
    labels = list(accuracies_dict.keys())
    accuracies = [accuracies_dict[act_fn] for act_fn in labels]
    
    fig, ax = plt.subplots()
    ax.boxplot(accuracies, labels=labels)
    ax.set_title('Activation Functions Comparison')
    ax.set_xlabel('Activation Function')
    ax.set_ylabel('Validation Accuracy')

    plt.show()

# Usage example
folder_path = './checkpoint'  # Replace with your path
activation_fns = ['GELU', 'SiLUT']  # Replace with your list of activation functions
repeats = 5  # Number of repeats

# Load model checkpoints and gather accuracies
accuracies_dict = load_model_checkpoints(folder_path, activation_fns, repeats)

# Plot the boxplot for comparison
plot_accuracies_boxplot(accuracies_dict)

