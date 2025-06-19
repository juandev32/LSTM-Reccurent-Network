# Text Generation Using Long Short-Term Memory (LSTM) Neural Network Architecture
  This project demonstrates one use case of Long Short-Term Memory Architecture.
  
  I created custom neural networks to demonstrate how the variation of properties of the LSTM model produce different results.
   I varied the length of text sequences, dropout rate (fraction of deactivated neurons), number of layers, and number of neurons within a layer.
  The text corpuses utilized are from Cat In The Hat and The Bee Movie Script.

  Within the LSTM_Models Folder, 
  the script to train the model is labeled with the varied LSTM properties.
  The script that produces the output text ends in generate_text

  The variation of these properties produced various qualities of output text.
  ALL of the models are trained on various, results could improve or reflect overfitting if trained on more epochs.

## Table of Contents
   1. [Installation](#installation)
   2. [Usage](#usage)
   3. [Features](#features)
   4. [Contact Info](#contact-info)
   5. [Versions Quick Reference](#versions-quick-reference)

## Installation

  1. **TENSORFLOW GPU (OPTIONAL)**
     
        To accelerate training of the neural network, it is recommended that Cuda (11.2) and CuDnn (8.1) is preinstalled on the system with the GPU Hardware.
        This is not nessesary to trial the project as there are pretraiend model weights already in the weights file. 
        There are also sample text outputs already in `./generated_text_Output/`

  3. **Install Python** (Python 3.10.*) or run it in a virtual environment

  4. This project used these packages and versions.
        keras==2.10.0
        numpy==2.2.2
        tensorflow==2.10.0

  5. **The functins in the package_utilities.py script will ensure that you have all required dependencies installed.**
        If you do not, then you will be prompted, within the terminal, if you want to install the required dependency.

## Usage

  1. **Run the Script** (optional)
     
      You can use the `*/LSTM_Models/select_architecture/filename.py` file to train your own LSTM model with the pre-defined hyperparameters.

     The training will begin automatically after all dependencies have been validated.

     This process is very slow depending on the model architecture and the size of the text corpus.
     You can accelerate this process by having cuda installed

  3. **Load a weights file and generate text**
      
        You change directory into the either `./beeMovie/` or `./LSTM_Models/`
        Run the text generation script ending in "generate_text"

        ex: `python BeeMovie-lstm_ThreeLayer_seqLen16_generate_text.py`

        It will ask you which predefined weights you want to utilize to run through the text corpus.
        
    
  5. **View the output text files**
     
     You can see the model outputs in the terminal or view them as .txt files in generated_text_OUTPUTS

     ex: `LSTM Reccurent Network - Writing Emulator\generated_text_OUTPUTS\OUTPUT - BeeMovie-lstm_ThreeLayer_seqLen16_50-1.2479.txt`

## Features

  1. Both scripts will validate the requirements of the project.
        The package_utilities script in the root directory will check if you are running Cuda and your Graphics card is being utilized properly.
        It will also make sure that you have all required dependencies required for this project to run.
        It is suggested that you create a python virtual environment if you want to train a model, but it is not nessesary aslong as the dependencies in the requirements folder are installed.
    
  2. The generate text script will ask you which set of weights you would like to load. Then you can generate 1000 characters utilizing the trained model.
     
  3. The text will be automatically saved to ./generated_text_OUTPUTS

## Contact Info

  **Email:** [juandev32@gmail.com](mailto:juandev32@gmail.com)  
  **LinkedIn:** [Juan Chavira's Profile](https://www.linkedin.com/in/juan-chavira/)
    
## VERSIONS QUICK REFERENCE
  - CUDA 11.2.* 
  - cuDNN 8.1.*
  - Python 3.10.*
  - Tensorflow==2.10.*
  - keras==2.10.0
  - numpy==2.2.2

    [TENSORFLOW COMPATIABILITY REQUIREMENTS](https://www.tensorflow.org/install/source#gpu)