from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from google.colab import files
from IPython.display import Image
import numpy as np
import matplotlib.pyplot as plt

classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']

def train():
  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

  x_train = x_train.reshape(60000, 784)
  x_train = x_train / 255
  y_train = utils.to_categorical(y_train, 10)

  x_test = x_test.reshape(10000, 784)
  x_test = x_test / 255
  y_test = utils.to_categorical(y_test, 10)

  model = Sequential()
  model.add(Dense(736, input_dim=784, activation="tanh"))
  model.add(Dense(10, activation="softmax"))
  model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
  history = model.fit(x_train, y_train, 
                    batch_size=200, 
                    epochs=10,
                    validation_split=0.2,
                    verbose=1)

  scores = model.evaluate(x_test, y_test, verbose=1)
  model.save('fashion_mnist_dense.h5')

#train()
model = load_model('fashion_mnist_dense.h5')

f = files.upload()
image_path = "tshirt.png"
Image(image_path, width=150, height=150)
img = image.load_img(image_path, target_size=(28,28), color_mode="grayscale")

x = image.img_to_array(img)
x = x.reshape(1,784)
x = 255-x
x /= 255

plt.imshow(img.convert('RGBA'))
plt.show()

prediction = model.predict(x)
print(classes[np.argmax(prediction)])
