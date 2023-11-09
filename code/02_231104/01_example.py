import numpy as np
import pretty_midi
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load MIDI file
midi_file = '../data/agnusdei_1.mid'
midi_data = pretty_midi.PrettyMIDI(midi_file) # Open MIDI file

# Extract the melody by choosing a specific instrument track
instrument_number = 0
instrument = midi_data.instruments[instrument_number]
melody_data = np.array([note.pitch for note in instrument.notes])

# Define sequence lenght for training
sequence_length = 4

# Prepare data sequences for training
sequences = []
targets = []
for i in range(len(melody_data) - sequence_length):
    # Create sequences of sequences_length consecutive notes
    sequences.append(melody_data[i:i+sequence_length])
    # Store the next as the target
    targets.append(melody_data[i+sequence_length])


# # Create a linear model
# model = nn.Sequential(
#     nn.Linear(batch_size, 1)
# )

# # Define the loss function (MSE)
# loss_fn = nn.MSELoss()

# # Define the optimizer (Adam)
# learning_rate = 0.001
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # Training loop to predict the next MIDI note
# num_epochs = 1000
# loss_history = []

# for epoch in range(num_epochs):
#     optimizer.zero_grad()
#     epoch_loss = []

#     # Take a sequence of batch size length notes
#     for start_idx in range(len(data) - batch_size):
#         input_seq = data[start_idx:start_idx + batch_size].unsqueeze(0)
#         target_note = data[start_idx + batch_size]
    
#         # Make the prediction
#         prediction = model(input_seq)
    
#         # Calculate the loss
#         loss = loss_fn(prediction.squeeze(), target_note)
    
#         # Perform backpropagation and optimization
#         loss.backward()
#         optimizer.step()

#         epoch_loss.append(loss.item())

#     avg_loss = np.mean(epoch_loss)
#     loss_history.append(avg_loss)  # Append the loss value for plotting

#     if (epoch + 1) % 100 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# # Make a prediction on the next MIDI note
# input_seq = data[-batch_size:]
# prediction = model(input_seq)
# print(f'Predicted next MIDI note: {int(round(prediction.item()))}')

# # Create a plot of the loss history
# plt.plot(range(1, num_epochs+1), loss_history, color='b')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Loss during training')
# plt.grid(True)
# plt.show()