# train.py
from load_data import train_loader
from tqdm import tqdm

# Define the number of epochs
n_epochs = 10  # Set this to the desired number of epochs

for epoch in range(n_epochs):
    total_loss = 0.0

    # Correctly iterate over the train_loader
    for i, data in enumerate(tqdm(train_loader)):
        images, labels = data  # Unpack images and labels from the data
        print(f"Batch {i}: Image shape: {images.shape}, Labels: {labels}")
        # Here you would include your training step, loss calculation, etc.
        # total_loss += your_loss_function(...)

    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss}")
