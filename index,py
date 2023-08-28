# Importing the necessary libraries and frameworks
import tensorflow as tf 
from tensorflow import keras 
import numpy as np
import matplotlib.pyplot as plt 

# We used a built in dataset from keras
fashion_mnist = keras.datasets.fashion_mnist  #This loads the data from keras
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() #This helps split the data
print(train_images.shape) #finding out the shape of the training data
print(train_images[0,23,23]) #Let's look at 1 pixel
print(train_labels[:10])  # Let's look at the first 10 training model

#Let's create an array of the label names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", 
               "shirt", "Sneaker", "Bag", "Ankle Boot"]

#Using matplotlib we can visualize our DATA
#plt.figure()
#plt.imshow(train_images[6])
#plt.colorbar()
#plt.grid(False)
#plt.show()
 
#PREPROCESSING OUR DATA
train_images = train_images / 225.0
test_images = test_images / 225.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape= (28, 28)), #input layer 1
    keras.layers.Dense(128, activation='relu'), #Hidden layer 2
    keras.layers.Dense(10, activation='softmax') # output layer 3
    ])

#Compiling the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Training our model
model.fit(train_images, train_labels, epochs=5)

#Evaluating/testing the test_data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print('Test accuracy:', test_acc)

#Predicting our model 
predictions = model.predict(test_images)
print(class_names[np.argmax(predictions[1])])

#Visualize our predicted data 
'''plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()'''

#Now let's get an input from our user and predict
COLOR = "white"
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def get_number():
    while True:
        num = input("Pick a number: ")
        if num.isdigit():
            num = int(num)
            if 0 <= num <= 1000:
                return int(num)
        else:
            print("Try another number...")

num = get_number()
image = test_images[num]
label = test_labels[num] 

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", 
               "shirt", "Sneaker", "Bag", "Ankle Boot"]

def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title("Expected: " + label)
    plt.xlabel("Guess: " + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()

def predict(model, image, correct_label):
    predictions = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(predictions)]
    show_image(image, class_names[correct_label], predicted_class)

print(model, image, label)
