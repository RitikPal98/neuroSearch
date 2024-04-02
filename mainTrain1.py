import os
import cv2
from PIL import Image
import numpy as np
# Path to the dataset folder
dataset_folder = "brain_tumor_dataset/"

no_tumor_images = os.listdir(dataset_folder + "no/")
yes_tumor_images = os.listdir(dataset_folder + "yes/")
dataset=[]
labels=[]

for i, image_name in enumerate(no_tumor_images):
    if(image_name).split(".")[1]=='jpg':
        image=cv2.imread(dataset_folder+"/no/"+image_name)
        if image is not None:
            image=Image.fromarray(image,'RGB')
            image=image.resize((64,64))
            dataset.append(np.array(image))
            labels.append(0)
        else:
            print(f"Image {image_name} could not be read")
for i, image_name in enumerate(yes_tumor_images):
    if(image_name).split(".")[1]=='jpg':
        image=cv2.imread(dataset_folder+"/yes/"+image_name)
        if image is not None:
            image=Image.fromarray(image,'RGB')
            image=image.resize((64,64))
            dataset.append(np.array(image))
            labels.append(1)
        else:
            print(f"Image {image_name} could not be read")

print(dataset)
print(labels)