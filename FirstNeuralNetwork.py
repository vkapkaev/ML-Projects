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

# функция обучения нейронной сети
def train():
  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() # загружаем данные

  classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']
  #данные для обучения - предобработка
  x_train = x_train.reshape(60000, 784)
  x_train = x_train / 255
  y_train = utils.to_categorical(y_train, 10)
  #данные для оценки качества обучения - предобработка
  x_test = x_test.reshape(10000, 784)
  x_test = x_test / 255
  y_test = utils.to_categorical(y_test, 10)
  #построение нейронной сети
  model = Sequential()
  model.add(Dense(800, input_dim=784, activation="relu")) #входной слой, картинка 28*28=784 пикселя. Активатор - relu (REtified Linear Unit)
  model.add(Dense(10, activation="softmax")) # выходной слой - 10 нейронов(классов)
  model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
  history = model.fit(x_train, y_train, 
                    batch_size=200, 
                    epochs=10,
                    validation_split=0.2,
                    verbose=1)
  #оценка качества
  scores = model.evaluate(x_test, y_test, verbose=1)
  #сохранение нейросети в файл
  model.save('fashion_mnist_dense.h5')

#train()
#загрузка
model = load_model('fashion_mnist_dense.h5')
#подача произвольного файла trainers.jpg
f = files.upload()
image_path = "trainers.jpg"
Image(image_path, width=150, height=150)
img = image.load_img(image_path, target_size=(28,28), color_mode="grayscale")
#нормализация изображения
x = image.img_to_array(img)
x = x.reshape(1,784)
x = 255-x
x /= 255

plt.imshow(img.convert('RGBA'))
plt.show()

prediction = model.predict(x)
print(classes[np.argmax(prediction)])
