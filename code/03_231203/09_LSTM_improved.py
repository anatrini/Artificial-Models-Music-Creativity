import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Improve performance: 
# 1. using a LSTM model
# 2. multiple layers, dropout, weight_decay, early stopping

midi_file = './data/agnusdei_1.mid'  # Replace with the path to your MIDI file
midi_data = pretty_midi.PrettyMIDI(midi_file)

# Extract the melody (e.g., by choosing a specific instrument track)
instrument_number = 0  # Adjust this to choose the desired instrument track
instrument = midi_data.instruments[instrument_number]
note_features = np.array([(note.pitch, note.start, note.end, note.velocity) for note in instrument.notes])

# Define the sequence length for training
sequence_length = 5

# Prepare data sequences for training
sequences = []
targets = []
for i in range(len(note_features) - sequence_length*2):
    sequences.append(note_features[i:i+sequence_length])
    targets.append(note_features[i+sequence_length:i+sequence_length*2])

# Convert sequences and targets to NumPy arrays
sequences = np.array(sequences)
targets = np.array(targets)

# Split data into training and testing sets
test_size = 0.2  # Define the portion of data to use for testing
X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=test_size, random_state=42)

# Get the input size dynamically based on the first sequence
input_size = 4 # Each MIDI note is a feature vector of size 4
hidden_size = 64 # 
output_size = 4

# Create a LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_rate=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.fc(out)
        return out

model = LSTMModel(input_size, hidden_size, output_size)

# Define the loss function (MSE: Mean Squared Error)
loss_fn = nn.MSELoss()

# Define the optimizer (Adam)
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Training loop to predict the next MIDI note
num_epochs = 1000
train_loss_history = []  # Store training loss over epochs
val_loss_history = []    # Store validation loss over epochs
predictions = [] # Store predictions

# Set the maximum number of epochs without improvements before stopping training
patience = 20

# Track the best validation loss value and the number of epochs since it improved
best_val_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(num_epochs):
    epoch_loss = []

    indices = np.random.permutation(len(X_train))

    for idx in indices:
        optimizer.zero_grad()
    
        input_seq = torch.tensor(X_train[idx], dtype=torch.float32).view(1, sequence_length, input_size)
        target_note = torch.tensor(y_train[idx], dtype=torch.float32).unsqueeze(0)
    
        prediction = model(input_seq)

        predictions.append(prediction.detach().numpy())
    
        loss = loss_fn(prediction, target_note)
        
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

    avg_loss = np.mean(epoch_loss)
    train_loss_history.append(avg_loss)


    # Evaluate the model on the validation set 
    val_losses = []
    for i in range(len(X_test)):
        input_seq_val_set = torch.tensor(X_test[i], dtype=torch.float32).view(sequence_length, input_size).unsqueeze(0)
        target_note_val = torch.tensor(y_test[i], dtype=torch.float32).unsqueeze(0)
        prediction_val_set = model(input_seq_val_set)
        val_loss = loss_fn(prediction_val_set, target_note_val)
        val_losses.append(val_loss.item())

    # Calculate and store the average validation loss
    avg_val_loss = np.mean(val_losses)
    val_loss_history.append(avg_val_loss) 

    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f'Early stopping after {epoch+1} epochs.')
        break

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    if epoch >= 200:
        # Cut 1 dimension, attached graidents, convert into numpy array and integer numbers
        predicted_notes = prediction.squeeze().detach().numpy().astype(int)
        predictions.append(predicted_notes)

print(f'Predictions: {predictions}')


# Calculate no. of epochs
num_epochs_actual = len(train_loss_history)

fig, axs = plt.subplots(2, figsize=(12, 8))

# Truncate first N epochs
N = 20  # Choose a proper value for N
axs[0].plot(range(N+1, num_epochs_actual+1), train_loss_history[N:], label='Train Loss')
axs[0].plot(range(N+1, num_epochs_actual+1), val_loss_history[N:], label='Val Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_title('Training and Validation Loss (First {} Epochs Truncated)'.format(N))
axs[0].legend()
axs[0].grid(True)

# Plot using a logarithmic scale
axs[1].plot(train_loss_history, label='Train Loss')
axs[1].plot(val_loss_history, label='Val Loss')  
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss')
axs[1].set_yscale('log')  # Change on log scale along y axis
axs[1].set_title('Training and Validation Loss (Log Scale)')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()