import os
from dotenv import load_dotenv
from telegram.ext import *
import tensorflow as tf
import cv2
import numpy as np
from io import BytesIO

load_dotenv()

API_TOKEN = os.getenv('KEY')

class_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")
model.compile(optimizer="adam", loss= "sparse_categorical_crossentropy", metrics=["accuracy"])

help_text = """
/start - Starts conversation
/help - Shows this message

This is a telegram bot that is capable of identifying a:
plane 
car
bird
cat
deer
dog
frog
horse
ship
or truck.
It was created with Python, and uses Tensorflow to train the CIFAR10 dataset.
Upload any image of the listed objects to get started!
"""

# When 
def start(update, context):
    update.message.reply_text("Hello! Please use /help to see the available commands!")
    
def help(update, cntext):
    update.message.reply_text(help_text)

# def train(update, context):
#     update.message.reply_text("Model is being trained ...")
#     model.compile(optimizer="adam", loss= "sparse_categorical_crossentropy", metrics=["accuracy"])
#     # fit on training data with 10 epochs
#     model.fit(x_train, y_train, epochs = 10, validation_data = (x_test, y_test))
#     model.save("cifar_classifier.model")
#     update.message.reply_text("Done! You can now send a photo")

def handle_message(update, context):
    update.message.reply_text("Please send a picture or press /help for more info!")

# process image from user
def handle_photo(update, context):
    # gets the last photo from user
    file = context.bot.get_file(update.message.photo[-1].file_id) 
    # get photo as byte array
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()), dtype = np.uint8)
    # loading image into script using OpenCV
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # by default, OpenCV uses BGR but our image uses RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (32, 32), interpolation = cv2.INTER_AREA)

    # activation of all neurons. However, we want the neuron with the highest activation
    prediction = model.predict(np.array([img / 255]))
    # np.argmax(prediction) gives the index of the neuron with the highest prediction
    update.message.reply_text(f"This is a {class_names[np.argmax(prediction)]}!")

# for telegram bot
updater = Updater(API_TOKEN, use_context = True)
dp = updater.dispatcher

dp.add_handler(CommandHandler("start", start))
dp.add_handler(CommandHandler("help", help))
# dp.add_handler(CommandHandler("train", train))
dp.add_handler(MessageHandler(Filters.text, handle_message))
dp.add_handler(MessageHandler(Filters.photo, handle_photo))

updater.start_polling()
updater.idle()