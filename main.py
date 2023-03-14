from telegram.ext import *
import tensorflow as tf
import cv2
import numpy as np
from io import BytesIO

with open("bot_api_token.txt", "r") as f:
    # gets the API TOKEN
    API_TOKEN = f.read()

(x_train, y_train), (x_test, y_test) = tf.kera.datasets.cifar10.load_data()
x_train, x_test = x_train / 255, x_test / 255

class_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

model = tf.keras.models.Sequential()
# 2D convolutional layer - 32 filters, kernel size 3 by 3, activation relu, 32 by 32 pixels, 3 channel rgb
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation = "relu"))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation = "relu"))
# flatten results
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation="relu"))
# 10 outputs due to 10 possible classes
# softmax is producing a result where if you add up all the individual neuron activations you will get 1 (100%)
# each neuron add how likely it is close to correct classification
model.add(tf.keras.layers.Dense(10, activation = "softmax"))

# When 
def start(update, context):
    update.message.reply_text("Hello!")

def help(update, cntext):
    update.message.reply_text(
        """
        /start - Starts conversation
        /help - Shows this message
        /train - Trains neural network
        """
    )

def train(update, context):
    update.message.reply_text("Model is being trained ...")
    model.compile(optimizer="adam", loss= "sparse_categorical_crossentropy", metrics=["accuracy"])
    # fit on training data with 10 epochs
    model.fit(x_train, y_train, epochs = 10, validation_data = (x_test, y_test))
    model.save("cifar_classifier.model")
    update.message.reply_text("Done! You can now send a photo")

def handle_message(update, context):
    update.message.reply_text("Please train the model and send a picture!")

def handle_photo(update, context):
    file = context.bot.get_file(update.message.photo[-1].file_id) # gets the last photo from user
    # get photo as byte array
    f = BytesIO(file.download_as_bytearray())

# for telegram bot
updater = Updater(API_TOKEN, use_context = True)
dp = updater.dispatcher

dp.add_handler(CommandHandler("start", start))
dp.add_handler(CommandHandler("help", help))
dp.add_handler(CommandHandler("train", train))
dp.add_handler(MessageHandler(filters.text, handle_message))
dp.add_handler(MessageHandler(filters.photo, handle_photo))

updater.start_polling()
updater.idle()