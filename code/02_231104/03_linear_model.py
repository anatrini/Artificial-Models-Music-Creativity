import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generate random data (sequence of 10 MIDI notes)
data = torch.tensor([60, 62, 64, 60, 60, 62, 64, 60, 64, 65, 67, 64, 65, 67, 67, 69, 67, 65, 64, 60, 67, 69, 67, 65, 64, 60, 67, 55, 60], dtype=torch.float32)
batch_size = 4


# Create a linear model
model = nn.Sequential(
    nn.Linear(batch_size, 1)
)

# Define the loss function (MSE)
loss_fn = nn.MSELoss()

# Define the optimizer (Adam)
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop to predict the next MIDI note
num_epochs = 1000
loss_history = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    epoch_loss = []

    # Take a sequence of batch size length notes
    for start_idx in range(len(data) - batch_size):
        input_seq = data[start_idx:start_idx + batch_size].unsqueeze(0)
        target_note = data[start_idx + batch_size]
    
        # Make the prediction
        prediction = model(input_seq)
    
        # Calculate the loss
        loss = loss_fn(prediction.squeeze(), target_note)
    
        # Perform backpropagation and optimization
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

    avg_loss = sum(epoch_loss) / len(epoch_loss)
    loss_history.append(avg_loss)  # Append the loss value for plotting
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    #if (epoch + 1) % 100 == 0:
    #    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Make a prediction on the next MIDI note
input_seq = data[-batch_size:]
prediction = model(input_seq)
print(f'Predicted next MIDI note: {int(round(prediction.item()))}')

# Create a plot of the loss history
plt.plot(range(1, num_epochs+1), loss_history, color='b')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss During Training')
plt.grid(True)
plt.show()