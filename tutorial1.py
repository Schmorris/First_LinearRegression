import tensorflow as tf
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt 

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images = train_images/255.0
test_images = test_images/255.0

#Creating a model
model = keras.Sequential([# Defining a Sequence of layers in order
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation='relu'),#rectified linear unit as activation function 
	keras.layers.Dense(10)#picks values for each neuron so all of the values add up to one
	])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

#train the model
model.fit(train_images, train_labels, epochs=5)#epochs= how often it sees the image for example. Just tweak

#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

#print('\nTest accuracy:', test_acc)

prediction = model.predict(test_images)

for i in range(5):
	plt.grid(False)
	plt.imshow(test_images[i], cmap=plt.cm.binary)
	plt.xlabel('Actual: ' + class_names[test_labels[i]])
	plt.title('Prediction: ' + class_names[np.argmax(prediction[i])])
	plt.show()
