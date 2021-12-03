import PySimpleGUI as sg 
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from gtts import gTTS
import os

# /Users/riyapatel/opt/anaconda3/bin/python /Users/riyapatel/Documents/HAII/Final/asl_recognizer_gui.py

def speak(speech_text):
    myobj = gTTS(text=speech_text, lang='en', slow=False)
    myobj.save("speech.mp3")
    os.system("afplay speech.mp3")

sg.theme('Dark Blue 3')  
text = ''

layout = [
            [sg.Text('ASL Translator', size=(30, 1))],
            [sg.Image(filename='', key='image')],
            [sg.Text(text, key='text'), sg.Button('Delete'), sg.Button('Speak')]
        ]
window = sg.Window('HAII Final Project', layout, size=(1400, 775), text_justification='center', auto_size_text=False, element_justification='center')

# Loading the model.
MODEL_NAME = 'asl_alphabet_{}.h5'.format(9575)
model = load_model(MODEL_NAME)

classes_file = open("classes.txt")
classes_string = classes_file.readline()
classes = classes_string.split()
classes.sort()

USE_CAMERA = 0  
cap = cv2.VideoCapture(USE_CAMERA)

IMAGE_SIZE = 200
CROP_SIZE = 400
# Prepare data generator for standardizing frames before sending them into the model.
data_generator = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

prev_guess = ''
guess_count = 0
buffer = 5


while True:
    if window(timeout=20)[0] == sg.WIN_CLOSED:
        break

    ret, frame = cap.read()
    
    cv2.rectangle(frame, (75, 175), (CROP_SIZE+75, CROP_SIZE+175), (0, 0, 0), 3)
    
    # Preprocessing the frame before input to the model.
    cropped_image = frame[175:(CROP_SIZE+175), 75:(CROP_SIZE+75)]
    resized_frame = cv2.resize(cropped_image, (IMAGE_SIZE, IMAGE_SIZE))
    reshaped_frame = (np.array(resized_frame)).reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
    frame_for_model = data_generator.standardize(np.float64(reshaped_frame))

    # Predicting the frame.
    prediction = np.array(model.predict(frame_for_model))
    predicted_class = classes[prediction.argmax()]      # Selecting the max confidence index.

    # Preparing output based on the model's confidence.
    prediction_probability = prediction[0, prediction.argmax()]
    if prediction_probability > 0.5:
        # High confidence.
        cv2.putText(frame, '{} - {:.2f}%'.format(predicted_class, prediction_probability * 100), 
                                    (75, 145), 1, 2, (113,113,113), 2, cv2.LINE_AA)
        if prev_guess == predicted_class:
            guess_count+=1
        else:
            prev_guess = predicted_class
            guess_count = 1
        if guess_count >= buffer:
            text += predicted_class
            guess_count = 0
    elif prediction_probability > 0.2 and prediction_probability <= 0.5:
        # Low confidence.
        cv2.putText(frame, 'Maybe {}... - {:.2f}%'.format(predicted_class, prediction_probability * 100), 
                                    (75, 145), 1, 2, (113,113,113), 2, cv2.LINE_AA)
        if prev_guess == predicted_class:
            guess_count+=1
        else:
            prev_guess = predicted_class
            guess_count = 1
        if guess_count >= buffer:
            text += predicted_class
            guess_count = 0
    else:
        # No confidence.
        cv2.putText(frame, classes[-2], (75, 145), 1, 2, (113,113,113), 2, cv2.LINE_AA)

    # Display the image with prediction.
    window['image'](data=cv2.imencode('.png', frame)[1].tobytes(), size=(1250,700))
    window['text'].update(text)

    event, values = window.read(timeout=0)
    if event == 'Delete':
        text = text[:-1]
    if event == 'Speak':
        speak(text)
    

# When everything done, release the capture.
cap.release()
cv2.destroyAllWindows()