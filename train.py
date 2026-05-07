from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

##load training data
#rescale 0-balck 1- white ,
#What rescale=1./255 does

#It divides every pixel by 255:
# #200 / 255 = 0.78
# 150 / 255 = 0.58
# 50  / 255 = 0.19
#at_time 32 images=batch_size
#class 2binary,o-cat,1-dogs

train_data=ImageDataGenerator(rescale=1./255).flow_from_directory("dataset/training_set",
    target_size=(150,150),batch_size=32,class_mode="binary")
model=Sequential([Conv2D(16,(3,3),activation ='relu',input_shape =(150,150,3)),
                 MaxPooling2D(2,2),
                 Flatten(),
                 Dense(1,activation='sigmoid')])
model.compile(optimizer='adam',loss='binary_crossentropy')
model.fit(train_data,epochs=2)
print(train_data.class_indices)
model.save("cat_dog_model.h5")