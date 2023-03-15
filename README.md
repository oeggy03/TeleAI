# TeleAI
You can test the bot by going to https://t.me/teleaimage_bot, or setting it up yourself. 

The model has been pre-trained, and the bot is ready to go. However, if you wish to see the training in action, you may run the training.py file after installing all the necessary packages.

## About this bot
This bot has been trained using OpenCV, Tensorflow and the keras deep learning API, using the CIFAR10 dataset. Since this is purely a demo project, the smaller and simpler CIFAR10 was chosen over CIFAR100 or other more advanced datasets.

It was built using the Python programming language.

## Capabilities
The teleAI bot is a telegram bot capable of image recognition, and can recognise images from the following categories:

*Plane, 
Car,
Bird,
Cat,
Deer,
Dog,
Frog,
Horse,
Ship,
Truck*

## Set up - If you wish to test it out
Make sure you have Python3 installed. I am using Python 3.11.2

#### Install the following packages:
1. Python telegram bot version 13.7: `pip install python-telegram-bot==13.7`
2. Python dotenv(): `pip install python-dotenv`
3. Tensorflow: `pip install tensorflow`
4. Numpy: `pip install numpy`
5. OpenCV: `pip install opencv-python`

#### Setting up a telegram bot
1. On your telegram app, go to https://t.me/BotFather
2. Type and send `/newbot`.
3. Follow the instructions, and you will get an API key.
4. Create a .env file, and type `KEY = <Your API key here>`
5. Also take note of the bot's telegram link (looks like `t.me/<bot_username>`)

#### Testing out the bot
1. Run the main.py file using `python main.py` in the terminal.
2. Go to the bot's telegram link and type in `/start`.
3. Upload an image of one of the objects in the mentioned categories.
4. Done! The bot should be able to recognise the image.


*Please note that AI is not perfect, and will classify objects wrongly at times, especially if the object is dark / shadowy. However, it should work well as long as the subject of the photo is obvious.*
