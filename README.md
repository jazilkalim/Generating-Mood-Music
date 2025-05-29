# 🎵 DEAM Music Generator using VAE & GAN

This project implements a **deep generative model for music creation** using **Variational Autoencoders (VAE)** and **Generative Adversarial Networks (GAN)**. It leverages the **DEAM (Database for Emotional Analysis of Music)** dataset which includes spectrograms, valence-arousal annotations, and static song-level features.

---

## 🔍 Project Overview

We aim to learn latent representations of music and generate emotionally conditioned audio using:

- A **Convolutional VAE** to compress spectrograms into latent space
- An **Autoregressive GAN-based generator** that generates spectrogram slices over time, conditioned on emotional inputs like valence and arousal
- Conversion of spectrograms back to audio for auditory validation

---

## 📁 Dataset

- **DEAM Dataset**: Includes dynamic and static annotations  
  📦 [Download here](https://www.kaggle.com/datasets/imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music)

- **Inputs**:
  - `mel_spectrograms`: Slices of audio tracks
  - `valence.csv` and `arousal.csv`: Per-second emotional annotations
  - `static_annotations_averaged_songs_*.csv`: Overall mood metadata

---

## 🧠 Model Architecture

### Variational Autoencoder (VAE)

- **Encoder**: 2D CNN layers compress spectrograms into a latent space
- **Latent Dimension**: 100
- **Decoder**: Reconstructs spectrogram from latent embeddings

### Autoregressive Sequence Generator (GAN-based)

- **Input**: Past latent vectors and emotional state (valence, arousal)
- **Architecture**:
  - Dense layers + LSTM cell wrapped in `RNN`
  - Output reshaped to match spectrogram format
- **Output**: Sequence of future spectrograms

---

## 🎓 Training Procedure

1. **Train VAE**: On 128x128 spectrogram slices to learn latent encodings
2. **Train Generator**:
   - Input: Past latent vectors + emotion condition
   - Target: Next spectrogram frame
3. **Loss Functions**:
   - VAE: Reconstruction + KL divergence
   - Generator: L2 loss for prediction accuracy

---

## 🔊 Audio Generation

- Predict the next 30 seconds of music by:
  - Autoregressively generating new spectrograms
  - Concatenating generated frames
  - Inverse Mel-spectrogram transformation to waveform using `librosa`

---

## 📈 Visualizations

- Plot valence/arousal over time for each song
- Display predicted vs original spectrograms
- Listen to generated music samples

---

## 📦 Dependencies

```bash
pip install tensorflow keras librosa numpy pandas matplotlib
# Generating-Mood-Music
