import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Step 1: Read data from Excel file
excel_file = "E:/posture_detections/csv_withleaning.xls"
data = pd.read_excel(excel_file)

# Assuming your data has columns: Frame_1, Frame_2, ..., x_5, x_6, ..., y_5, y_6, ..., label

# Step 2: Define your dataset class
class PoseDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x_cols = ['x_5', 'x_6', 'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16',
                  'y_5', 'y_6', 'y_11', 'y_12', 'y_13', 'y_14', 'y_15', 'y_16']
        
        x_values = self.data.loc[idx, x_cols].values.astype(float)
        y_label = self.data.loc[idx, 'label']  # Assuming 'label' contains the class label
        
        # Map label string to numerical index
        label_map = {'sitting': 0, 'standing': 1, 'squatting': 2, 'leaning':3}
        y_target = label_map[y_label]
        
        return torch.tensor(x_values, dtype=torch.float32), torch.tensor(y_target, dtype=torch.long)

# Step 3: Initialize your dataset and dataloader
dataset = PoseDataset(data)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Step 4: Define your MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Step 5: Instantiate your model, loss function, and optimizer
input_size = 16  # Assuming 16 features (x_5 to x_16 and y_5 to y_16)
hidden_size = 128
num_classes = 4   # Number of classes (sitting, standing, squatting, leaning) #trained on four classes(leaning added)

model = MLP(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Train your model
num_epochs = 90
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 7: Save the trained model
torch.save(model.state_dict(), 'pose_classification_model_90.pth')
print('Model saved successfully.')