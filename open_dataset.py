import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split


def ini_dataset(data_folder_path: str, xlsx_file_path : str) -> (TensorDataset, TensorDataset):
    """ Extract the data from the .dat and the .xlsx files and create a tuple of data and label dataset """
    
    columns_to_extract = ['Participant Id', 'Session Id', 'Video Id' ,'Valence level', 'Arousal level', 'Dominance level']
    data = pd.read_excel(xlsx_file_path, sheet_name='Sheet1', usecols=columns_to_extract)
    label_data = data.values

    data_list = []
    label_list = []

    for row in label_data:
        participant_id, session_id, video_id = row[:3]  # Extracting Participant Id, Session Id, and Video Id
        file_name = f"ECGdata_s{session_id}p{participant_id}v{video_id}.dat"
        file_path = os.path.join(data_folder_path, file_name)

        if os.path.exists(file_path):   # Check if the file exists, add its data into the mapping
            file_data = np.loadtxt(file_path, delimiter=',')
            #print(f"File length: {len(file_data)}")
            label = row[3:6]  # Extracting label data
            label = np.array(label, dtype=np.float32)  
            data_list.append(file_data)
            label_list.append(label)

        else :
            print(f"File {file_name} does not exist")


    # Convert lists to PyTorch tensors
    data_tensor = torch.tensor(data_list)
    label_tensor = torch.tensor(label_list)

    # Create TensorDatasets for data and labels
    combined_dataset = TensorDataset(data_tensor, label_tensor)
    return combined_dataset
  

def split_data(dataset: TensorDataset) -> (TensorDataset, TensorDataset):
    """ Split the dataset into training and testing sets """

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    return train_data, test_data


if __name__ == '__main__' : 
    folder_path = './Dataset/ECG'
    file_path = './Dataset/Self-annotation.xlsx'
    
    combined_dataset = ini_dataset(folder_path, file_path)
    # Printing the first element of the dictionary

    print(f"Data Dataset: {combined_dataset}")
    print(f"Data type: {type(combined_dataset)}")
    print(f"Dataset shape: {combined_dataset[0][0].shape}")
    train_data, test_data = split_data(combined_dataset)
    print(f"Train Data lenght : {len(train_data)}")
    print(f"Train data: {train_data}")
    print(f"Train data[0]: {train_data[0]}")
    print(f"Train data[0] ecg: {train_data[0][0]}")
    print(f"Train data[0] emotions: {train_data[0][1]}")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
    print(f"Train loader: {train_loader}")
    # print(f"Test Data: {test_data}")
    



