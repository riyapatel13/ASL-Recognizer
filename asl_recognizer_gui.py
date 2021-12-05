# imports
import PySimpleGUI as sg
from PySimpleGUI.PySimpleGUI import I 
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from gtts import gTTS
import os


def speak(speech_text):
    ''' Speaks out given text using Google's Text-to-Speech engine. Converts to 
        an audio file and plays using afplay.

    -- PARAMETERS --
    speech_text : str -> text to be spoken
    '''

    myobj = gTTS(text=speech_text, lang='en', slow=False)
    # saves audio file
    myobj.save("speech.mp3")
    os.system("afplay speech.mp3")


def preprocess_img(frame, CROP_SIZE, IMAGE_SIZE):
    ''' Process image to input into model.

    -- PARAMETERS --
    frame : numpy.ndarray -> full-screen image 
    CROP_SIZE : int -> size of frame that needs to be cropped
    IMAGE_SIZE : int -> size of image that needs to be inputted to the model. The 
                        image should resize from CROP_SIZE to IMAGE_SIZE.

    -- RETURNS --
    model_image : numpy.ndarray -> image to be sent as input to model
    '''
    # Prepare data generator for standardizing frames before sending them into the model.
    data_generator = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

    cropped_image = frame[175:(CROP_SIZE+175), 75:(CROP_SIZE+75)]
    resized_frame = cv2.resize(cropped_image, (IMAGE_SIZE, IMAGE_SIZE))
    reshaped_frame = (np.array(resized_frame)).reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
    model_image = data_generator.standardize(np.float64(reshaped_frame))
    return model_image


def run_application():

    # loading model and classification classes
    model = load_model('models/asl_alphabet_9575.h5')

    classes_file = open("classes.txt")
    classes_string = classes_file.readline()
    classes = classes_string.split()
    classes.sort()

    # setting up GUI 
    sg.theme('Dark Grey 6')  
    translated_text = ''

    layout = [
                [sg.Text('ASL Translator', size=(30, 1))],
                [sg.Image(filename='', key='image')],
                [sg.Text(translated_text, key='text'), sg.Button('Delete'), sg.Button('Speak')]
            ]
    window = sg.Window('HAII Final Project', layout, size=(1400, 775), text_justification='center', 
                        auto_size_text=False, element_justification='center')

    # setting up variables for streaming
    USE_CAMERA = 0  
    cap = cv2.VideoCapture(USE_CAMERA)
    IMAGE_SIZE = 200 # trained on 200x200 images
    CROP_SIZE = 400 # captures 400x400 images
    
    # debouncing variables so it doesn't predict every single frame
    prev_guess = ''
    guess_count = 0
    predict_buffer = 4
    time_count = 0
    time_buffer = 1


    while True: 
        if window(timeout=20)[0] == sg.WIN_CLOSED:
            # end streaming when user hits red x
            break
        
        # stream
        _, frame = cap.read()
        
        # image bounding box
        cv2.rectangle(frame, (75, 175), (CROP_SIZE+75, CROP_SIZE+175), (0, 0, 0), 3)
        
        time_count += 1
        if time_count >= time_buffer:
            
            # Preprocessing the frame before input to the model.
            model_image = preprocess_img(frame, CROP_SIZE, IMAGE_SIZE)

            # Predicting the frame.
            prediction = np.array(model.predict(model_image))
            # Selecting the max confidence index.
            predicted_class = classes[prediction.argmax()]      
            # Preparing output based on the model's confidence.
            prediction_probability = prediction[0, prediction.argmax()]


            if prediction_probability >= 0.25:
                # debounce the predicted letter - must be guessed predict_buffer 
                # times in a row
                if prev_guess == predicted_class:
                    guess_count+=1
                else:
                    prev_guess = predicted_class
                    guess_count = 1
                
                # print prediction probability for users
                cv2.putText(frame, str(predicted_class)+' - '+str(round(prediction_probability*100,3))+'%', 
                            (75, 145), 1, 2, (50,50,50), 2, cv2.LINE_AA)

                if guess_count >= predict_buffer and predicted_class != 'nothing':
                    # change rectangle color to indicate detection
                    cv2.rectangle(frame, (75, 175), (CROP_SIZE+75, CROP_SIZE+175), (0, 255, 0), 3)

                    if predicted_class == 'space':
                        predicted_class = ' '
                    translated_text += predicted_class
                    #guess_count = 0


            else:
                # No confidence.
                cv2.putText(frame, "unable to detect", (75, 145), 1, 2, (50,50,50), 2, cv2.LINE_AA)
            
            # reset variables
            if guess_count >= predict_buffer and time_count >= time_buffer:
                time_count = 0
                guess_count = 0
                

        # Update stream.
        window['image'](data=cv2.imencode('.png', frame)[1].tobytes(), size=(1250,700))
        # Update text with prediction.
        window['text'].update(translated_text)

        # Button event handlers
        event, _ = window.read(timeout=0)
        if event == 'Delete':
            translated_text = translated_text[:-1]
        if event == 'Speak':
            speak(translated_text)

    # When everything done, release the capture.
    cap.release()
    cv2.destroyAllWindows()


run_application()