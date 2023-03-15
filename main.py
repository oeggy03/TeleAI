import os
from dotenv import load_dotenv
from telegram.ext import *
import tensorflow as tf
import cv2
import numpy as np
from io import BytesIO

load_dotenv()

API_TOKEN = os.getenv('KEY')

class_names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(loaded_model_json)

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

def start(update, context):
    update.message.reply_text("Hello! Loading model...")
    # load weights into new model
    update.message.reply_text("Model loaded! Please use /help to get started!")
    
def help(update, cntext):
    update.message.reply_text(help_text)


def message_handler(update, context):
    update.message.reply_text("Please send a picture or press /help for more info!")

# process image from user
def photo_handler(update, context):
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
    update.message.reply_text(f"This is a {class_names[np.argmax(prediction)]}! Certainty: {np.max(prediction)}")

# for telegram bot
updater = Updater(API_TOKEN, use_context = True)
dp = updater.dispatcher

# for commands
dp.add_handler(CommandHandler("start", start))
dp.add_handler(CommandHandler("help", help))

# for user messages
dp.add_handler(MessageHandler(Filters.text, message_handler))
dp.add_handler(MessageHandler(Filters.photo, photo_handler))

updater.start_polling()
updater.idle()