# 👁️‍🗨️ Multimodal Image Captioning: CNN-LSTM Architecture

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep_Learning-D00000?logo=keras&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project implements a sophisticated multimodal deep learning pipeline designed to automatically generate descriptive text captions for images. This architecture bridges the gap between **Computer Vision (CV)** and **Natural Language Processing (NLP)**, demonstrating the ability to extract spatial hierarchies from visual data and translate them into coherent, sequential language.

## 🏛️ Architecture Details (Encoder-Decoder)
The core of this project is a robust **Encoder-Decoder** framework:

1.  **The Vision Encoder (CNN):**
    * **Model:** Pre-trained **VGG16** (trained on the massive ImageNet dataset).
    * **Mechanism (Transfer Learning):** We leverage VGG16 as a powerful feature extractor. By stripping away the final classification layers, we process each input image through the convolutional base to output a dense, 4096-dimensional feature vector. This vector encapsulates the rich spatial and semantic features of the image.

2.  **The Sequence Decoder (LSTM):**
    * **Model:** Long Short-Term Memory (**LSTM**) Recurrent Neural Network.
    * **Mechanism:** The textual captions are heavily preprocessed (tokenized, cleaned via regex, and padded). The textual sequences pass through an `Embedding` layer. The LSTM then processes this sequence data to learn the temporal dependencies and context of the language.

3.  **The Multimodal Fusion (Decoder Head):**
    * The 4096-d visual feature vector and the output from the LSTM are merged (via an `add` layer). This combined representation is passed through a dense feedforward network, culminating in a `Softmax` layer that outputs a probability distribution over the entire vocabulary to predict the next word in the sequence.

## 🛠️ Key Technical Highlights
* **Optimized Data Handling:** Due to the memory-intensive nature of multimodal data, the training pipeline utilizes **Custom Python Generators**. This allows the model to train on massive datasets in manageable batches without exhausting system RAM.
* **Robust Text Preprocessing:** Implemented strict regex-based text cleaning to sanitize punctuation and enforce lowercase uniformity. Critical sequence boundaries (`startseq` and `endseq`) are injected to manage the LSTM's generation lifecycle.
* **Model Persistence:** Integrated Keras `ModelCheckpoint` callbacks to continuously monitor loss metrics and persist the best-performing model weights during the training cycle.

## 📊 Dataset Context
This implementation is designed to train on the standard **Flickr8k dataset**, utilizing the `captions.txt` file which maps image identifiers to multiple human-annotated descriptive captions.

## 🚀 Getting Started (Local Execution)

### 1. Prerequisites
Ensure you have the required dependencies installed:
```bash
pip install tensorflow numpy tqdm pillow
