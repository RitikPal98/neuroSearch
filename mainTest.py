import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor10EpochsCategorical.h5')

# image=cv2.imread('D:\\workspace\\projects\\hackathon\\IITD\\BTD\\pred\\pred0.jpg')
image=cv2.imread("./pred/pred0.jpg")

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)
# print(img)
input_img=np.expand_dims(img, axis=0)

# result=model.predict_classes(input_img)
# print(result)


result = model.predict(input_img)
print(result)

predicted_class = np.argmax(result, axis=-1)
print(predicted_class)



