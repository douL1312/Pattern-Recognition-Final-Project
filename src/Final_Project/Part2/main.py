import pandas as pd
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Φόρτωση δεδομένων
train_features = pd.read_csv('Data_Receptors/Train_Features.csv', header=None)
train_labels = pd.read_csv('Data_Receptors/Train_Labels.csv', header=None)
test_features = pd.read_csv('Data_Receptors/Test_Features.csv', header=None)

# Διαχωρισμός σε training_set και validation_set
train_features, val_features, train_labels, val_labels = train_test_split(
  train_features,
  train_labels,
  test_size=0.2,
  random_state=42)

# Μετατροπή δεδομένων σε PyTorch tensors
train_features_tensor = tc.tensor(train_features.values, dtype=tc.float32)
train_labels_tensor = tc.tensor(train_labels.values, dtype=tc.float32)
val_features_tensor = tc.tensor(val_features.values, dtype=tc.float32)
val_labels_tensor = tc.tensor(val_labels.values, dtype=tc.float32)
test_features_tensor = tc.tensor(test_features.values, dtype=tc.float32)

# Ορισμός των datasets και των dataloaders
train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Δημιουργία μοντέλου
class RecModel(nn.Module):
  def __init__(self):
    super(RecModel, self).__init__()
    # Επίπεδα για συνεχή χαρακτηριστικά
    self.fc_cont_1 = nn.Linear(1425, 1024)
    self.fc_cont_2 = nn.Linear(1024, 512)
    self.fc_cont_3 = nn.Linear(512, 256)
    self.fc_cont_4 = nn.Linear(256, 128)
    self.fc_cont_5 = nn.Linear(128, 64)

    # Επίπεδα για δυαδικά χαρακτηριστικά
    self.fc_bin_1 = nn.Linear(2048, 1024)
    self.fc_bin_2 = nn.Linear(1024, 512)
    self.fc_bin_3 = nn.Linear(512, 256)
    self.fc_bin_4 = nn.Linear(256, 128)
    self.fc_bin_5 = nn.Linear(128, 64)

    # Συνδυαστικά επίπεδα
    self.fc_comb_1 = nn.Linear(128, 64)
    self.fc_comb_2 = nn.Linear(64, 32)
    self.fc_comb_3 = nn.Linear(32, 16)

    # Επίπεδο εξόδου
    self.output = nn.Linear(16, 1)

  def forward(self, x):
    # Διαχωρισμός των χαρακτηριστικών σε συνεχή(από 0 έως 1424) και δυαδικά(από 1425 έως 3472)
    x_cont = x[:, :1425]
    x_bin = x[:, 1425:]

    # Επεξεργασία των συνεχών χαρακτηριστικών
    x_cont = F.relu(self.fc_cont_1(x_cont))
    x_cont = F.relu(self.fc_cont_2(x_cont))
    x_cont = F.relu(self.fc_cont_3(x_cont))
    x_cont = F.relu(self.fc_cont_4(x_cont))
    x_cont = F.relu(self.fc_cont_5(x_cont))

    # Επεξεργασία των δυαδικών χαρακτηριστικών
    x_bin = F.relu(self.fc_bin_1(x_bin))
    x_bin = F.relu(self.fc_bin_2(x_bin))
    x_bin = F.relu(self.fc_bin_3(x_bin))
    x_bin = F.relu(self.fc_bin_4(x_bin))
    x_bin = F.relu(self.fc_bin_5(x_bin))
    
    # Συνδυασμός συνεχών και δυαδικών χαρακτηριστικών
    x_comb = tc.cat((x_cont, x_bin), dim=1)

    # Επεξεργασία των συνδυασμένων χαρακτηριστικών
    x_comb = F.relu(self.fc_comb_1(x_comb))
    x_comb = F.relu(self.fc_comb_2(x_comb))
    x_comb = F.relu(self.fc_comb_3(x_comb))

    # Τελική πρόβλεψη
    x_out = tc.sigmoid(self.output(x_comb))
    return x_out

model = RecModel()
criterion = nn.BCELoss() # Παίρνουμε ως κριτήριο την Binary CrossEntropy Loss
optimizer = tc.optim.Adam(model.parameters(), lr=0.001) # Χρησιμοποιούμε ως Optimizer τον Adam και ρυθμός εκμάθησης=0.001
epochs = 10

for epoch in range(epochs):
  model.train()
  for inputs, labels in train_loader:
    # Μηδενισμός gradient
    optimizer.zero_grad()

    # Πρόβλεψη από το μοντέλο
    outputs = model(inputs)

    # Υπολογισμός απώλειας
    loss = criterion(outputs, labels)

    # Backpropagation
    loss.backward()

    # Ενημέρωση βαρών
    optimizer.step()

  print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

model.eval()

with tc.no_grad():
  correct = 0
  total = 0
  for inputs, labels in val_loader:
    outputs = model(inputs)
    predicted = (outputs > 0.5).float()
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
  print(f'Validation accuracy: {100 * correct / total}%')