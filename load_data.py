# import zipfile
# from tqdm import tqdm

# with zipfile.ZipFile("data_sample.zip", "r") as zip_ref:
#     # get the total number of files in the zip file
#     total_files = len(zip_ref.infolist())

#     # iterate over the files in the zip file and extract them
#     for file in tqdm(zip_ref.infolist(), total=total_files):
#         zip_ref.extract(file, "data_sample")

import os
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image

#Load leabels of data

labels_df = pd.read_csv("labels.csv")
print(labels_df.head())




# for idx, img in enumerate(np.random.choice(train_imgs, 30)):
#     ax = fig.add_subplot(3, 30//3, idx+1)
#     im = Image.open(path2data + "/" + img)
#     print(plt.imshow(im))
#     plt.savefig("new.jpg")
#     label = labels_df.loc[labels_df["id"] == img.split(".")[0], 'label'].values[0]
#     ax.set_title(f"Label:{label}")
    





from torch.utils.data import TensorDataset, DataLoader,Dataset, random_split

import torch
import os
import pandas as pd
from PIL import Image


class cancer_dataset(Dataset):
    def __init__(self, root_dir, transform, data_type=None):
        self.transform = transform
        
        # Path to the directory containing images
        path_of_data = os.path.join(root_dir, "data_sample/data_sample")
        
        # Get list of filenames in the directory
        get_file_names = os.listdir(path_of_data)
        
        # Load labels.csv
        path2label = os.path.join(root_dir, 'labels.csv')
        
        # Full path to each image
        self.full_filenames = [os.path.join(path_of_data, f) for f in get_file_names]
        
        # Load the CSV containing labels
        get_label = pd.read_csv(path2label)
        
        # Set 'id' as the index for easy access
        get_label.set_index("id", inplace=True)
        
        # Subsetting based on data_type
        if data_type == "train":
            self.labels = [get_label.loc[img_files[:-4]].values[0] for img_files in get_file_names][0:3608]
            self.full_filenames = [os.path.join(path_of_data, f) for f in get_file_names][0:2608]
            print("training_dataset")
            
        elif data_type == "val":
            self.labels = [get_label.loc[img_files[:-4]].values[0] for img_files in get_file_names][3608:3648]
            self.full_filenames = [os.path.join(path_of_data, f) for f in get_file_names][3508:3648]
            print("validation_dataset")
            
        elif data_type == "test":
            self.labels = [get_label.loc[img_files[:-4]].values[0] for img_files in get_file_names][3648:-1]
            self.full_filenames = [os.path.join(path_of_data, f) for f in get_file_names][3648:-1]
            print("testing_dataset")
            
        else:
            # Here `get_label` is used instead of `labels_df`
            self.labels = [get_label.loc[filename[:-4]].values[0] for filename in get_file_names]
    
    def __len__(self):
        return len(self.full_filenames)
        
    def __getitem__(self, idx):
        # Open the image file and apply the transform
        img = Image.open(self.full_filenames[idx])
        img = self.transform(img)
        
        # Return the transformed image and its corresponding label
        return img, self.labels[idx]

        
        
        
        
from torchvision import transforms

# data transformation 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
composed_train = transforms.Compose([
                               transforms.Resize((224, 224)),
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomRotation(degrees=5),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, std)
                               ])

# this transformation is for valiadationa and test sets
composed= transforms.Compose([
                               transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, std)
                               ])


# data_dir = "./"
# dataset_full = cancer_dataset(data_dir , transform=composed)

# img, label = dataset_full [19]
# print(img.shape, torch.min(img), torch.max(img))       
        
        
        
        
        
        


# This function will allow us to easily plot tensor data 
from numpy import clip , array
from matplotlib import pyplot as plt
from torch import Tensor

def imshow(inp: Tensor) -> None:
    """Imshow for Tensor."""
    inp = inp.cpu().numpy()
    inp = inp.transpose((1, 2, 0))
    mean = array([0.485, 0.456, 0.406])
    std = array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()








 

    
# for i in range(5):
    
    
#     plt.title("sample of the training data")
#     imshow(training_set[2][0])

 




    
    






import torch
from torch.utils.data import DataLoader
# ... other imports

def get_data_loaders(batch_size=10, num_workers=2):
    data_dir = "./"
    
    training_set = cancer_dataset(data_dir, transform=composed_train, data_type="train")
    validation_set = cancer_dataset(data_dir, transform=composed, data_type="val")
    test_set = cancer_dataset(data_dir, transform=composed, data_type="test")
    
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader





    

    
    


# Define classes
num_classes = 2
print("done")
    


