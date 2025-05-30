# handwritten_recognition
# 🖊️ Handwritten Digit & Letter Recognition using CNNs (PyTorch)

This project implements a deep learning-based image classifier that can recognize handwritten **digits (0–9)** using the **MNIST** dataset and handwritten **letters (A–Z)** using the **EMNIST** dataset. Both classifiers are trained using Convolutional Neural Networks (CNNs) built in **PyTorch**.

---

## 📦 Features

* 📥 Automatic downloading and preprocessing of MNIST and EMNIST datasets
* 🧠 CNN model architecture for image classification
* 🔁 Training loops with real-time accuracy and loss reporting
* 📊 Evaluation and prediction on test datasets
* ✅ Proper label remapping for EMNIST (which ranges from 1–26)

---

## 🧠 Model Architecture

A simple CNN architecture used for both digit and letter classifiers:

```
Conv2D (1 → 32) → ReLU → MaxPool
Conv2D (32 → 64) → ReLU → MaxPool
Flatten → FC(3136 → 128) → ReLU → FC(128 → num_classes)
```

---

## 📁 Project Structure

```
handwritten_recognition/
├── handwritten_digit_letter_cnn.py    # Complete PyTorch script
├── README.md                          # This file
└── data/                              # MNIST & EMNIST will be auto-downloaded here
```

---

## ⚙️ Installation

Install the required libraries using pip:

```bash
pip install torch torchvision matplotlib
```

---

## 🚀 How to Run

Run the full script (downloads data, trains models, evaluates):

```bash
python handwritten_digit_letter_cnn.py
```

---

## 📊 Sample Output

```
🎯 Training Digit Classifier (MNIST)
Epoch [1/5], Loss: 169.12, Accuracy: 94.54%
...
Test Accuracy: 98.90%

🔤 Training Letter Classifier (EMNIST)
Epoch [1/5], Loss: 325.18, Accuracy: 76.23%
...
Test Accuracy: 88.65%
```

---

## 📌 Notes

* EMNIST letter labels are originally in the range `1–26`; we shift them to `0–25` for PyTorch compatibility.
* Both models use the same CNN class and training functions for simplicity.

---

## 📚 References

* [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
* [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)
* [PyTorch Docs](https://pytorch.org/)

---

## ✨ Future Work

* Add training visualizations (loss curves)
* Save/load trained models
* Improve CNN architecture with Dropout/BatchNorm
* Build a small web app for interactive handwriting demo (e.g. using Streamlit)

---

## 🤝 Acknowledgments

Developed using PyTorch and open datasets for educational purposes.
