import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from utils import adjust_learning_rate, rmse, write_result
from parsing import create_parser



def train_model(model : nn.Module, train_loader : DataLoader, test_loader : DataLoader, criterion, optimizer, args) -> None :
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        print(f"Data length: {len(train_loader)}")
        print(f"train loader: {train_loader}")
        for i, data in enumerate(train_loader):
            print(f"Data: {data}")
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            print(f"Input : {inputs}")
            print(f"Size input : {inputs.size()}")
            inputs = inputs.unsqueeze(0)
            print(f"Size input after : {inputs.size()}")
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        adjust_learning_rate(args, optimizer, epoch)

        model.eval()
        test_loss = 0.0
        predictions = []
        true_labels = []
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item()
                predictions.extend(outputs.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        test_loss /= len(test_loader)
        rmse_value = rmse(np.array(predictions), np.array(true_labels))
        
        print(f"Epoch [{epoch + 1}/{args.epochs}], "
              f"Train Loss: {running_loss / len(train_loader):.4f}, "
              f"Test Loss: {test_loss:.4f}, "
              f"RMSE: {rmse_value:.4f}")

        # Save model weights and write it into a file
        torch.save(model.state_dict(), 'ecg_model_epoch{epoch}.pth')
        write_result('valence', predictions, 'destination_file.csv')



if __name__ == "__main__":
    from ecg_net import EmotionCNN
    from open_dataset import ini_dataset, split_data

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    # Initialize dataset and dataloader
    folder_path = './Dataset/ECG'
    file_path = './Dataset/Self-annotation.xlsx'

    combined_dataset = ini_dataset(folder_path, file_path)
    train_data, test_data = split_data(combined_dataset)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = EmotionCNN(input_channels=1)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    train_model(model, train_loader, test_loader, criterion, optimizer, args)
