import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import glob
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
from models import upsampleCNN

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split

from models import upsampleCNN

file_dir = os.path.dirname(os.path.abspath(__file__))
pjt_dir = os.path.dirname(file_dir)

# constructing the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=100,
            help='number of epochs to train the model for')
args = vars(parser.parse_args())

# helper functions
image_dir = os.path.join(file_dir, '../outputs/saved_images')
os.makedirs(image_dir, exist_ok=True)  #creates new dir, doesnt do anything if already exists
    
def save_decoded_image(img, name):
    # img = img.view(img.size(0), 3, 224, 224)
    save_image(img, name)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Device found: {device}")

gauss_blur = os.listdir(os.path.join(file_dir, '../input/gaussian_blurred/'))
gauss_blur.sort()
sharp = os.listdir(os.path.join(file_dir, '../input/sharp'))
sharp.sort()

x_blur = []
for i in range(len(gauss_blur)):
    x_blur.append(gauss_blur[i])

y_sharp = []
for i in range(len(sharp)):
    y_sharp.append(sharp[i])
    
# print(x_blur[10])
# print(y_sharp[10])

(x_train, x_val, y_train, y_val) = train_test_split(x_blur, y_sharp, test_size=0.25)

print(f"No. of training data points: {len(x_train)}")
print(f"No. of validation data points: {len(x_val)}")

# define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class DeblurDataset(Dataset):
    def __init__(self, blur_paths, sharp_paths=None, transforms=None):
        self.X = blur_paths
        self.y = sharp_paths
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        blur_image = cv2.imread(os.path.join(pjt_dir, f"input\gaussian_blurred\{self.X[i]}"))
        
        if self.transforms:
            blur_image = self.transforms(blur_image)
            
        if self.y is not None:
            sharp_image = cv2.imread(os.path.join(pjt_dir, f"input\sharp\{self.y[i]}"))
            if self.transforms:
                sharp_image = self.transforms(sharp_image)
            sample = {'blur': blur_image, 'sharp': sharp_image}
            return sample 
        else:
            sample = {'blur': blur_image}
            return sample

train_data = DeblurDataset(x_train, y_train, transform)
val_data = DeblurDataset(x_val, y_val, transform)

batch_size = 2
 
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

#creates an instance of the "CNN "model and moves the model to the specified device
model = upsampleCNN.CNN().to(device)
print(model)

# the loss function
criterion = nn.MSELoss()
# the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
        optimizer,
        mode='min',
        patience=5,
        factor=0.1,
        verbose=True
    )

print ('hxsdjhsdcd')
def fit(model, dataloader):
    model.train() # puts model in training mode
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        blur_image = data['blur']
        sharp_image = data['sharp']
        if blur_image is None:
            continue
        blur_image = blur_image.to(device)
        sharp_image = sharp_image.to(device)
        optimizer.zero_grad()
        outputs = model(blur_image)
        # loss = criterion(outputs, sharp_image)
        resized_outputs = F.interpolate(outputs, scale_factor=(4, 4)) #this is the new code added to modify the size for new convolution layers
        loss = criterion(resized_outputs, sharp_image) #this is the new code added to modify the size for new convolution layers
        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step()
        running_loss += loss.item()
    
    train_loss = running_loss/len(dataloader.dataset)
    print(f"Train Loss: {train_loss:.5f}")
    
    return train_loss

# the training function
def validate(model, dataloader):
    model.eval() # puts model in evaluation mode
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            blur_image = data['blur']
            sharp_image = data['sharp']
            if blur_image is None:
                    continue
            blur_image = blur_image.to(device)
            sharp_image = sharp_image.to(device)
            outputs = model(blur_image)
            resized_outputs = F.interpolate(outputs, scale_factor=(4, 4)) #this is the new code added to modify the size for new convolution layers
            loss = criterion(resized_outputs, sharp_image) #this is the new code added to modify the size for new convolution layers
            #loss = criterion(outputs, sharp_image)
            running_loss += loss.item()

            # if epoch == 0 and i == (len(val_data)/dataloader.batch_size)-1:
            #     save_decoded_image(sharp_image.cpu().data, name=os.path.join(pjt_dir, f"outputs/saved_images/sharp{epoch}.jpg"))
            #     save_decoded_image(blur_image.cpu().data, name=os.path.join(pjt_dir, f"outputs/saved_images/blur{epoch}.jpg"))

        val_loss = running_loss/len(dataloader.dataset)
        print(f"Val Loss: {val_loss:.5f}")

        print(f"Shape of output {outputs.shape}")
        save_decoded_image(outputs[0].cpu().data, name=os.path.join(pjt_dir, f"outputs/saved_images/val_deblurred{epoch}.png"))
        
        return val_loss


curr_time = str(datetime.now()).replace(':', '-')

train_loss = []
val_loss = []
start = time.time()
for epoch in range(args['epochs']):
    print(f"Epoch {epoch+1} of {args['epochs']}")
    train_epoch_loss = fit(model, trainloader)
    val_epoch_loss = validate(model, valloader)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    scheduler.step(val_epoch_loss)
end = time.time()

print(f"Took {((start-end)/60):.3f} minutes to train")

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(pjt_dir, f"outputs/saved_models/loss_plot_{curr_time}.png"))
plt.show()

# save the model to disk
print('Saving model...')
torch.save(model.state_dict(), os.path.join(pjt_dir, f"outputs/saved_models/model_{curr_time}.pth"))