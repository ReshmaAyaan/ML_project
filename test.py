from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
model=load_model('cat_dog_model.h5')

img=image.load_img(r"dataset/test_set/dogs/dog.4001.jpg",target_size =(150,150))

#Converts image into numbers (pixels)
img=image.img_to_array(img)

# ###Black = 0
# White = 255
# Colors = values in between
# [200, 150, 50]
# #200 / 255 = 0.78
# 150 / 255 = 0.58
# 50  / 255 = 0.19
# Small values (0–1)
# Faster learning
# Better accuracy
img = img / 255.0
###Simple analogy
# You have 1 photo 📷
# Model wants a folder of photos 📁

# So you do:

# 👉 Put your photo inside a folder
#img shape = (150, 150, 3)
#img shape = (1, 150, 150, 3)

img=np.expand_dims(img,axis=0)

result=model.predict(img)
print(result)

####result[0]=[0.5]
#result[0][0]=0.5  ##batch removing
##cat=0,dog=1 alphabetics


if result[0][0] >0.5:
    print('dog')
else:
    print('cat')

