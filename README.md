# 1. Project Overview
This repository contains a state-of-the-art Text-to-Speech (TTS) system that synthesizes natural-sounding speech from text using Tacotron2 and HiFi-GAN models. The project leverages deep learning techniques to produce high-quality audio output.

# 2. Key Features
High-Quality Audio: Generates clear and natural-sounding speech.
Customizable Parameters: Adjust various parameters to optimize audio output.
Real-time Interaction: Allows for real-time text input and audio playback.
Visualization: Visual representations of Mel spectrograms and alignments for a better understanding of the synthesis process.
# 3. Project Flow
**Environment Setup**: The process begins in the trainingtacotron2.py file, which mounts Google Drive and installs necessary packages. This ensures all dependencies are in place for running the project.

**Logging Configuration**: Configure logging settings to suppress warnings from libraries, maintaining a clean console output.

**Model Configuration**: Define model identifiers for Tacotron2 and HiFi-GAN. Check initialization status to ensure the environment is set up correctly.

**Pronunciation Dictionary Setup**: Download and prepare a pronunciation dictionary to convert text to ARPAbet format, improving the model’s understanding of input text.

**Model Download Functions**: Implement functions to download and initialize the HiFi-GAN and Tacotron2 models, ensuring compatibility.

**End-to-End Inference Function**: Convert input text to speech through a function that processes the text, generates Mel spectrograms with Tacotron2, and synthesizes audio using HiFi-GAN.

**User Input Loop**: Continuously prompt the user for input text and synthesize speech based on the provided input.

**Training Process**: The initialisetraining.py file initiates the training process for the Tacotron2 model using the prepared dataset.

# 4. File Descriptions
**1. trainingtacotron2.py**: Entry point of the project. It mounts Google Drive and downloads necessary dependencies.

**2. loadingdataset.py**: Contains functions for loading and processing the dataset used for training the models.

**3. uploadtranscript.py**: Manages transcription files necessary for training and evaluation.

**4. modelparameters.py**: Defines and manages the model parameters for Tacotron2 and HiFi-GAN.

**5. initialisetraining.py**: Responsible for starting the training process of the Tacotron2 model.

**6. hifiganaudiooutput.py**: Handles audio synthesis and output generation using the HiFi-GAN model.

# 5. Installation Instructions
To set up the project locally, follow these steps:

1. Clone this repository:
```
git clone https://github.com/Ridh1234/AIweek7.git`
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. load the desired dataset for your project(in our case a custom dataset of 200 voices recorded by one of our group member was used)
```
python loadingdataset.py
```
4. upload the filelist.txt corresponding to the audio files which are metadata for our dataset
```
python uploadtranscript.py
```
5. define the model parameters
```
python modelparameters.py
```
6. Start the project by running the trainingtacotron2.py file:
```
python trainingtacotron2.py
```
7. To train the model, prepare your dataset and use the initialisetraining.py file to begin the training process:
```
python initialisetraining.py
```
8. finally use the trained tacotron2 and generate audio by using Hifigan model
```
python hifiganaudiooutput.py
```
