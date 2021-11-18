import PySimpleGUI as sg 
import cv2

# /Users/riyapatel/opt/anaconda3/bin/python /Users/riyapatel/Documents/HAII/Final/asl_recognizer_gui.py

sg.theme('Dark Blue 3')  

layout = [
            [sg.Text('ASL Translator', size=(30, 1))],
            [sg.Image(filename='', key='image')],
            [sg.Text('Translated Text Goes Here'), sg.Button('Speak')]
        ]
window = sg.Window('HAII Final Project', layout, text_justification='center', auto_size_text=False, element_justification='center')

USE_CAMERA = 0  
cap = cv2.VideoCapture(USE_CAMERA)

while window(timeout=20)[0] != sg.WIN_CLOSED:
    window['image'](data=cv2.imencode('.png', cap.read()[1])[1].tobytes(), size=(800,450))

