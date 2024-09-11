# Language Detection from Audio

This repository provides an implementation for detecting the language from audio samples using a neural network model trained on a dataset of 10 Indian languages. The project uses the **InceptionV3** model for classification and supports data augmentation, preprocessing, and model training.

## Prerequisite

1. **Python Packages**  
   Install the required libraries using the following command:
   ```bash
   pip install -r requirements.txt
   ```
   Packages required:
   - `numpy`
   - `pandas`
   - `matplotlib`
   - `librosa`
   - `soundfile`
   - `tensorflow`
   - `keras`
   - `imageio`
   - `scipy`
   - `IPython`

2. **Kaggle API Setup**
   Ensure you have your Kaggle API key (`kaggle.json`) to download the dataset.
   ```bash
   mkdir ~/.kaggle
   cp kaggle.json ~/.kaggle/
   ```

## Dataset

The dataset used for training is available on Kaggle and can be downloaded using the following command:
```bash
kaggle datasets download -d hbchaitanyabharadwaj/audio-dataset-with-10-indian-languages
```
Extract and move the dataset into your working directory:
```bash
unzip /content/drive/MyDrive/data_Language_final.zip
rm -rf /content/Language_Detection_Dataset
shutil.move("/content/content/Language_Detection_Dataset", "/content/")
```

## Data Preparation

The dataset is split into **train** and **test** categories with support for multiple languages like **English**, **Gujarati**, and **Hindi**. It preprocesses the audio files and generates corresponding spectrogram images.

```python
languages = ['English', 'Gujarati', 'Hindi']
categories = ['Train', 'Test']
```

## Data Augmentation

You can augment the audio data by adding noise using the following function:
```python
def add_noise(audio_segment, gain):
    noise = gain * np.random.normal(size=audio_segment.shape[0])
    return audio_segment + noise
```

## Data Preprocessing

The audio files are converted to fixed-length segments and spectrograms are generated:
```python
def spectrogram(audio_segment):
    spec = lr.feature.melspectrogram(audio_segment, n_mels=128, hop_length=500)
    image_np = lr.core.power_to_db(spec)
    return image_np
```

## Model Architecture

We use the **InceptionV3** model for image classification, which takes the spectrogram images as input.

```python
model = InceptionV3(input_shape=(128, 500, 1), weights=None, classes=3)
model.compile(optimizer=RMSprop(learning_rate=0.045), loss='categorical_crossentropy', metrics=['accuracy'])
```

## Model Training

The model is trained using the following settings:
- Batch size: 128
- Image size: 128x500 (grayscale)
- Initial learning rate: 0.045

```python
history = model.fit(train_generator, validation_data=validation_generator, epochs=20)
```

## Evaluation

The trained model is evaluated on the test set and the accuracy is calculated.

```python
_, test_accuracy = model.evaluate(evaluation_generator)
print(f"Test accuracy: {round(test_accuracy * 100, 1)} %")
```

## User Application (Back-end)

You can use the following script to record audio from the user and predict the language:
```python
audio, sr = get_audio()  # Record audio
audio_to_image_file('/content/recording.wav')  # Convert audio to image
image = load('/content/recording.wav.png')  # Load spectrogram image
preds = model.predict(image)  # Predict language
```

## Deployment

This project can be deployed on **Google Colab** or any environment that supports Python with the necessary libraries installed.
