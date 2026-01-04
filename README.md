# Detecting Deepfake Audio Across Multiple AI Generators Using Machine Learning

By: Dylan Tesng

This project uses logistic regression and deep learning to recognize the difference beetwen human and generated speech based on a audio clip.

Both models are trained on the ASVspoof 2019 dataset which covers over real human speech as well as various generation methods such as vocoders and neural waveform model/text to speech. The neural network is also trained on The Fake-or-Real Dataset containing text to speech from various sources.

The goal of this project was to generate a machine learning model accuate enough to distinguish from various AI generation and speech sythesis methods while being light enough to work on a consumer laptop.

## Setup

1. Clone the repositiory

    ```bash
    git clone https://github.com/Deeeeelan/Deepfake-Audio-Detector.git
    ```

2. Set up a virtual environment (optional)

    ```bash
    # Windows
    python -m venv venv 
    venv\Scripts\activate
    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install dependencies

    ```bash
    pip install -r requirements.txt
    ```

Tested on Python 3.13.4.

Used Libaries:

* NumPy
* Pandas
* Librosa
* MatplotLib
* PyTorch
* TorchViz

Jupyter notebooks are located in the `notebooks` folder.

## Datasets

The models in this project were trained off the folowing datasets:

[ASVspoof 2019](https://www.asvspoof.org/index2019.html)

[The Fake-or-Real (FoR) Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset/data)

## Preprocessing

Sample rate for all audio is 16000 (automatically downsampled).

The logistic regression model is trained using the normalized average and standard deviation of 25 MFCC features per audio clip extracted with the Librosa library.

The convolutional neural network extracts log-scaled Mel-spectrograms and normalizes them.

* All spectrograms are timmed to 3 seconds in length
* n_fft = 2048, hop_length = 512, n_mels = 128

## Model Design

The logistic regression model is trained on MFCC features of the ASVspoof 2019 dataset.

The convolutional neural network in trained on the Mel-spectrograms of ASVspoof 2019 dataset and the Fake or Real dataset.

### Outputs

The model outputs logits (raw scores) for the following classes:

* **Class 0:** Fake audio
* **Class 1:** Real audio

Logits can be converted to classes with the sigmoid function:

```python
torch.sigmoid(logits)
```

## Results

### Logistic Regression Model

The model reached a 81% Accuracy, a precision of 98% for fake samples, and aprecision of 34% for real samples.

![Confusion Matrix for the logistic regression model](/images/confusion_matrix2.png)

### Convolutional Neural Network

The model reached 89% Accuracy, a precision of 97% for fake samples, and a precision of 55% for real samples.

![Confusion Matrix for the neural network](/images/confusion_matrix4nn.png)

## Limitations

* The model is trained mostly on clear English speech, and will likely not be accuate in other languages or dirtier sound environments.
* The model is also not trained on leading TTS models (Such as ElevenLabs or Amazon Poly).
* The neural network is limited to the first 3 seconds of spectrogram data.
