from telegram.ext import *
import tensorflow as tf
import cv2
import numpy as np
from io import BytesIO

with open("bot_api_token.txt", "r") as f:
    # gets the API TOKEN
    API_TOKEN = f.read()

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

# def train(update, context):
#     update.message.reply_text("Model is being trained ...")
#     model.compile(optimizer="adam", loss= "sparse_categorical_crossentropy", metrics=["accuracy"])
#     # fit on training data with 10 epochs
#     model.fit(x_train, y_train, epochs = 10, validation_data = (x_test, y_test))
#     model.save("cifar_classifier.model")
#     update.message.reply_text("Done! You can now send a photo")

def handle_message(update, context):
    update.message.reply_text("Please train the model and send a picture!")

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
    update.message.reply_text(f"In this imaage I see a {class_names[np.argmax(prediction)]}")

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