# ASL-Recognizer

This repo contains code to perform automatic static American Sign Language recognition. This was created as part of Option B for the final project for 05-318: Human-AI Interaction. My project specifically focuses on finger-spelling, which is used for spelling out names, addresses, and any proper nouns. 

## Files Included

* data 
  * Repo containing data files used for training and testing models.
* construct_models
  * Repo containing Python notebooks to run to create and save models used in application. Did not include trained models because they were too large to store on Github (requires Git LFS).
* classes.txt
  * File to enumerate all the classes that the model trained on - used for prediction. 
* asl_recognizer_gui.py
  * Python file to run application. Loads model created by American_Sign_Language_Recognition.ipynb and allows user to detect characters in real-time. 
* README.md
* requirements.txt
  * Python packages that need to be installed. 

## Installation & Running Code

1. In order to run this code, create a [virtual environment](https://docs.python.org/3/library/venv.html) and run the following in your virtual environment:
```bash
  pip install -r requirements.txt
```
This will install all packages and configure necessary dependencies needed for the code.
2. Since the model was too big to upload to Github, first train the model (approx. 12 hours on my CPU) and save it by running all the cells in American_Sign_Language_Recognition.ipynb.
3. Once the model has been saved, run the following:
```bash
python3 asl_recognizer_gui.py
```

