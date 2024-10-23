import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import pickle
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import tensorflow as tf


class EdgeGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, fc_hidden_size):
        super(EdgeGCN, self).__init__()
        
        #1D convolutional layers
        self.conv1d_1 = nn.Conv1d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=2)
        self.conv1d_2 = nn.Conv1d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.relu = nn.ReLU()

        self.fc =  nn.Linear(32*120, in_channels)

        #Graph convolutional Layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        # Fully connected layers
        self.fc1 = nn.Linear(out_channels, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, out_channels)  
        
    def forward(self, edge_attr, edge_index, batch):
        #forward pass of neural network
        x = self.conv1d_1(edge_attr)
        x = self.relu(x)
        x = self.conv1d_2(x)
        x = self.relu(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

      
        row, col = edge_index
        out = torch.zeros((batch.max().item() + 1, x.size(1)), device=x.device)
        out.index_add_(0, batch[row], x)

        x = F.relu(self.fc1(out))
        x = self.fc2(x)

        return x
    
def train(model, data, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in data:
        optimizer.zero_grad()  # Zero the gradients
        out = model(batch.edge_attr.float(), batch.edge_index, batch.batch)  # Forward pass
        loss = criterion(out, batch.y.long())  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        total_loss += loss.item()
    return total_loss/ len(data)

def evaluate(model, data, predictions, actual):
    model.eval()  # Set model to evaluation mode
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in data:
            out = model(batch.edge_attr.float(), batch.edge_index, batch.batch)  # Forward pass
            pred = out.argmax(dim=1)  # Get predictions
            predictions.append(pred)
            actual.append(batch.y.long())
            correct = (pred == batch.y.long()).sum().item()  # Boolean array of correct predictions
            total+= batch.y.size(0)  # Count total graphs
            total_correct += correct
    acc = total_correct/total
    return acc, predictions, actual

if __name__ =='__main__':
    with open("GraphList_Edges_classify1.pickle","rb") as f:
        graph_list_edges = pickle.load(f)
    print("Loaded Data")
    
    graph_list_edges_train, graph_list_edges_test = train_test_split(graph_list_edges, test_size=0.2, random_state=42, shuffle=True) #split the data into training and testing

    train_loader = DataLoader(graph_list_edges_train, batch_size=32)
    test_loader = DataLoader(graph_list_edges_test, batch_size=32)

    num_epochs = 500
    learning_rate = 0.0002
    in_channels = 64 # Number of input features
    hidden_channels = 128 # Number of hidden units
    fc_hidden_channels = 256
    out_channels = 25  # Number of classes

    modelEdge = EdgeGCN(in_channels, hidden_channels, out_channels, fc_hidden_channels)
    optimizer = torch.optim.Adam(modelEdge.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()



    #Training loop
    losses = []
    epochs = []
    for epoch in range(num_epochs):
        loss = train(modelEdge, train_loader, optimizer, criterion)
        losses.append(loss)
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')
        epochs.append(epoch+1)

    # Evaluation   

    predictions = []
    actual = []
    acc, predictions, actual = evaluate(modelEdge, test_loader, predictions, actual)
    print(f'Test Accuracy: {acc:.4f}')
    
    torch.save(modelEdge, 'model_edge_2.pth')

    plt.plot(epochs, losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Model Loss")
    plt.show()

    with open("graph_edges_ordinal_encoder1.pkl", 'rb') as f:
        ordinal_encoder = pickle.load(f)

    actual = torch.cat(actual).cpu().numpy()  # Convert to 1D array
    predictions = torch.cat(predictions).cpu().numpy()  # Convert to 1D array
 
    actual = ordinal_encoder.inverse_transform(actual.reshape(-1,1))
    predictions = ordinal_encoder.inverse_transform(predictions.reshape(-1,1))
    
    #display confusion_matrix
    cm = confusion_matrix(actual, predictions)
    plt.figure(figsize=(10, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    #Display Classification Metrics
    accuracy = accuracy_score(actual, predictions)
    precision = precision_score(actual, predictions, average='macro')
    recall = recall_score(actual, predictions, average='macro')
    f1 = f1_score(actual, predictions, average='macro')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

