from keras.constraints import maxnorm
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils

# image1_pathname = 'E:/images/pic1.jpeg'
# image1 = Image.open(image1_pathname)
# image1.show()

# display_image_pathname = input('Enter image pathname')
# display_image = Image.open(display_image_pathname)
# display_image.show()

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# index = int(input('Enter an image index: '))
# display_image = X_train[index]
# display_label = y_train[index][0]  # the 0 makes sure the call returns the number itself and not the array

# # another method using the pil library rather than pyplot
# final_image = Image.fromarray(display_image)
# final_image.show()

# # printing out red, green, or blue colors of an image only
# red_image = Image.fromarray(display_image)
# red, blue, green = red_image.split()
# plt.imshow(green, cmap="Greens")
# plt.show()
#
# plt.imshow(display_image)
# plt.show()
# print(labels[display_label])

new_X_train = X_train.astype('float32')
new_X_test = X_test.astype('float32')
new_X_train /= 255
new_X_test /= 255
new_Y_train = np_utils.to_categorical(y_train)
new_Y_test = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(new_X_train, new_Y_train, epochs=10, batch_size=32)

model.save('Trained_model.h5')
