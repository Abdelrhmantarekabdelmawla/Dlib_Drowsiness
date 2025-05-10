# 💤 Drowsiness Detection Model using Dlib

This repository contains a Python-based real-time drowsiness detection system using Dlib's facial landmark predictor. It tracks eye and mouth movements to detect signs of fatigue or yawning.

## 🚀 Features

- Real-time drowsiness detection using webcam.
- Eye Aspect Ratio (EAR) and Mouth Opening Ratio (MOR) logic.
- Evaluation support on a labeled test dataset.
- Visual + quantitative output (confusion matrix, F1-score).

## 📦 Requirements

- Python 3.6+
- Dlib
- OpenCV
- NumPy
- imutils
- scikit-learn

Install dependencies using:

```bash
pip install -r requirements.txt
````

## 📁 File Structure

```
├── drowsiness_detector.py                 # Real-time detection
├── Dlib_eval.py                           # Evaluation on test dataset
├── shape_predictor_68_face_landmarks.dat  # Pretrained Dlib model (see below)
├── README.md
└── requirements.txt
```

## 🔧 How It Works

* Detect face using Dlib.
* Predict 68 facial landmarks.
* Calculate EAR for both eyes and MOR for the mouth.
* If EAR is below a threshold or MOR is above a threshold → trigger drowsiness/yawning alert.

## 📥 Download Pretrained Model

The project depends on `shape_predictor_68_face_landmarks.dat`.

**Download from here**:
👉 [Download shape\_predictor\_68\_face\_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

Then extract it using:

```bash
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

Place the extracted `.dat` file inside the root directory of this repository.

## 🧪 Test on Dataset

You can evaluate the model using labeled image datasets (e.g., with folders `alert/` and `drowsy/`) using the script:

```bash
python Dlib_eval.py
```

Make sure to:

* Set the correct path in the `test_data_dir` variable.
* Place your test images inside subfolders:

  ```
  tests/datatest/test/
  ├── alert/
  └── drowsy/
  ```

This script will:

* Process each test image.
* Predict whether the subject is alert or drowsy.
* Output confusion matrix and F1 score.

### 📊 Evaluation Metrics

* Total detected drowsy and yawning images
* List of file names per category
* Confusion Matrix
* F1 Score

## ▶️ Real-Time Detection

To try real-time detection via webcam:

```bash
python drowsiness_detector.py
```

## 👨‍💻 Author

**Abdelrhman Tarek Abdelmawla**

## 📝 License

This project is licensed under the MIT License.
