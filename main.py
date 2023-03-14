from telegram.ext import *
import tensorflow as tf
import cv2
import numpy as np
from io import BytesIO

with open("bot_api_token.txt", "r") as f:
    # gets the API TOKEN
    API_TOKEN = f.read()

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
    pass

def handle_message(update, context):
    update.message.reply_text("Please train the model and send a picture!")

def handle_photo(update, context):
    pass

# for telegram bot
updater = Updater(API_TOKEN, use_context = True)
dp = updater.dispatcher

dp.add_handler(CommandHandler("start", start))
dp.add_handler(CommandHandler("help", help))
dp.add_handler(CommandHandler("train", train))
dp.add_handler(MessageHandler(Filters.text, start))