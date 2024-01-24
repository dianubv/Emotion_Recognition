# Emotion_Recognition

# TO DO 
    - fix train.py problem 
            -> shape problem

# TO ADD
    - a Data scaling or normalization.  Consider scaling or normalizing the ECG data to bring it within a reasonable range, which could enhance model performance.


""" 
Pb open_dataset.py : la taille est pas la même alors que ça devrait être 154 pour les deux
    print(f"Number of files: {len(data_dict)}")
    print(f"Number of labels: {len(labels)}")
     """


# Architecture 
I'm working on an emotion recognition project using ecg and deep learning. You will help me for this project. Ask question every time when you need to know more information about the project
parent-dir/
├── Dataset/
|    ├── ECG/
|    |    └──.dat files
|    └── Self-annotation.xlsx
├──ecg_net.py           # Defines a neural network architecture using PyTorch for processing ECG data, focusing on emotion recognition by utilizing a convolutional neural network (CNN) with multiple layers.
├──open_dataset.py      # Extracts data from files of the folder Dataset, processes it into labeled datasets, and provides functionality to split this data into training and testing sets.
├──parsing.py           # Contains functionalities to create a parser for command-line arguments, possibly related to configuring and running the emotion recognition model.
├──README.md
├──train.py             # Orchestrates the training of an emotion recognition model on ECG data by iterating through epochs, and evaluating its performance.
└──utils.py             # A collection of various utility functions for data manipulation, preprocessing, model weight handling, learning rate adjustment, and file operations, likely used across the project for diverse tasks.
 

Somatique data, biodata -> more aware about what they feel
interpretate data to emotion 
talk about future in the defense 
tel the story is the most important 
talk about the future thing I want to implement, explain with many details to show that I know, design side 