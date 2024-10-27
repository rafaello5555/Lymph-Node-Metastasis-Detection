from load_data import get_data_loaders
from tqdm import tqdm
from model import initialize_model


n_epochs = 10 

if __name__ == '__main__':
    # Load the data loaders
    train_loader, val_loader, test_loader = get_data_loaders()
    
    for epoch in range(n_epochs):
        running_loss = 0.0

        
        for i, data in enumerate(tqdm(train_loader)):
            inputs, labels = data  
            optimizer.zero_grad()
            
            outputs = initialize_model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()
           
            

    
