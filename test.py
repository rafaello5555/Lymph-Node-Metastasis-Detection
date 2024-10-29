from load_data import get_data_loaders
from tqdm import tqdm
from model import initialize_model
import torch

if __name__ == '__main__':
    # Load the data loaders
    train_loader, val_loader, test_loader = get_data_loaders()
    model, criterion, optimizer = initialize_model()
    

# Test the model
    correct = 0  
    total = 0 


    with torch.no_grad():
        
        
        for data in tqdm(test_loader):
             
            images, labels = data 
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
 
            total += labels.size(0)

        
            correct += (predicted == labels).sum().item()

# Compute and print the model's accuracy on the test set
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))