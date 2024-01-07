import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mido
import pretty_midi
import time
import matplotlib.pyplot as plt

# Load a MIDI file
midi_file = '../../data/agnusdei_1.mid'  # Replace with the path to your MIDI file
midi_data = pretty_midi.PrettyMIDI(midi_file)

# Extract the melody (e.g., by choosing a specific instrument track)
instrument_number = 1  # Adjust this to choose the desired instrument track
instrument = midi_data.instruments[instrument_number]
melody_data = np.array([note.pitch for note in instrument.notes])

# Define the sequence length for training
sequence_length = 10

# Prepare data sequences for training
sequences = []
targets = []
for i in range(len(melody_data) - sequence_length):
    # Create sequences of 10 consecutive notes
    sequences.append(melody_data[i:i+sequence_length])
    # Store the next note as the target
    targets.append(melody_data[i+sequence_length])

# Get the input size dynamically based on the first sequence
input_size = 1  # Each MIDI note is a single scalar value
hidden_size = 64
output_size = 1

# Create a simple RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Use only the last output of the RNN
        out = self.fc(out)
        return out

# Create an instance of the SimpleRNN model
model = SimpleRNN(input_size, hidden_size, output_size)

# Define the loss function (MSE)
loss_fn = nn.MSELoss()

# Define the optimizer (Adam)
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop to predict the next MIDI note
num_epochs = 1000
loss_history = []
mae_history = []
predictions = []

outport = mido.open_output('to Max 1')
note_duration = 0.5


for epoch in range(num_epochs):
    epoch_loss = []
    epoch_mae = []

    # Shuffle the sequences at the beginning of each epoch
    indices = np.random.permutation(len(sequences))

    for idx in indices:
        optimizer.zero_grad()
        
        # Convert the input sequence and target note to a 2D PyTorch tensors (seq_len, input_size)
        input_seq = torch.tensor(sequences[idx], dtype=torch.float32).view(sequence_length, 1).unsqueeze(0)
        target_note = torch.tensor(targets[idx], dtype=torch.float32)
        
        # Make a prediction using the model
        prediction = model(input_seq)

        # Calculate the loss between the prediction and target note
        loss = loss_fn(prediction.squeeze(), target_note)
        
        # Backpropagate the gradients and update the model's parameters
        loss.backward()
        optimizer.step()
        
        epoch_loss.append(loss.item())
        
        # Calculate the error (MAE: medium average error)
        mae = abs(prediction - target_note.unsqueeze(0)).item()
        epoch_mae.append(mae)

    avg_loss = np.mean(epoch_loss)
    avg_mae = np.mean(epoch_mae)
    
    loss_history.append(avg_loss)
    mae_history.append(avg_mae)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Error: {avg_mae:.4f}')

    if epoch >= 200:
        predicted_note = int(round(prediction.item()))
        predictions.append(predicted_note)

        # Create a midi message with the predicted note
        note_on = mido.Message('note_on', note=predicted_note)
        note_off = mido.Message('note_off', note=predicted_note)

        # Send midi over virtual output
        outport.send(note_on)
        # Wait for note duration
        time.sleep(note_duration)
        # Set to note off
        outport.send(note_off)

print(f'Predictions: {predictions}')

# Create a plot of the loss and error history
plt.figure(figsize=(10, 4))
plt.plot(range(1, num_epochs+1), loss_history, color='b', label='Loss')
plt.plot(range(1, num_epochs+1), mae_history, color='r', label='Error')
plt.xlabel('Epochs')
plt.ylabel('Loss / Error')
plt.title('Loss and Error During Training')
plt.legend()
plt.grid(True)
plt.show()