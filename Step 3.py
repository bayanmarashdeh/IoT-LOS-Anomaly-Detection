import os
import time
import librosa
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


print("=" * 60)
print("PHASE 3 - IoT ANOMALY DETECTION USING ISOLATION FOREST")
print("=" * 60)


# =====================================================
# Dataset Path
# =====================================================

DATASET_PATH = r"C:\Users\Home\OneDrive\Desktop\-6_dB_fan"

print("Dataset Path:")
print(DATASET_PATH)
print("Path Exists:", os.path.exists(DATASET_PATH))
print()


# =====================================================
# Feature Extraction
# =====================================================

def extract_features(file_path):

    signal, sr = librosa.load(file_path, sr=None)

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=20
    )

    mfcc_mean = np.mean(mfcc, axis=1)

    # RMS Energy
    rms = np.mean(librosa.feature.rms(y=signal))

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(signal))

    # Spectral Centroid
    centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=signal,
            sr=sr
        )
    )

    features = np.hstack((
        mfcc_mean,
        rms,
        zcr,
        centroid
    ))

    return features


# =====================================================
# Read Dataset
# =====================================================

X = []
Y = []

normal_count = 0
abnormal_count = 0

print("Reading Audio Files...")
print()

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.lower().endswith(".wav"):

            path = os.path.join(root, file)

            try:

                features = extract_features(path)

                folder = root.lower()

                if "abnormal" in folder:

                    X.append(features)
                    Y.append(1)
                    abnormal_count += 1

                elif "normal" in folder:

                    X.append(features)
                    Y.append(0)
                    normal_count += 1

            except Exception as e:

                print("Error:", path)
                print(e)


print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print("Normal Files   :", normal_count)
print("Abnormal Files :", abnormal_count)
print("Total Files    :", len(X))
print()

if len(X) == 0:

    print("No audio files found.")
    exit()


X = np.array(X)
Y = np.array(Y)


# =====================================================
# Train Isolation Forest
# =====================================================

print("Training Isolation Forest...")
print()

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

start = time.time()

model.fit(X)

training_time = time.time() - start

print("Training Finished.")
print("Training Time:", round(training_time, 3), "seconds")
print()


# =====================================================
# Predict All Samples
# =====================================================

prediction = model.predict(X)

# Convert
# 1  -> Normal (0)
# -1 -> Abnormal (1)

pred = np.where(prediction == -1, 1, 0)


# =====================================================
# Evaluation
# =====================================================

accuracy = accuracy_score(Y, pred)

print("=" * 60)
print("RESULTS")
print("=" * 60)

print("Accuracy:", round(accuracy * 100, 2), "%")
print()


print("Confusion Matrix")
print(confusion_matrix(Y, pred))
print()


print("Classification Report")
print(classification_report(Y, pred))


# =====================================================
# LoS Channel
# =====================================================

def LoS_channel(features):

    return features


# =====================================================
# NLoS Channel
# =====================================================

def NLoS_channel(features):

    noise = np.random.normal(
        0,
        0.5,
        features.shape
    )

    return features + noise


sample = X[0]
true_label = Y[0]


# =====================================================
# LoS Test
# =====================================================

print("=" * 60)
print("LoS TEST")
print("=" * 60)

start = time.time()

received = LoS_channel(sample)

prediction = model.predict(
    received.reshape(1, -1)
)

prediction = 1 if prediction[0] == -1 else 0

elapsed = time.time() - start

print("True Label    :", true_label)
print("Prediction    :", prediction)
print("Correct       :", prediction == true_label)
print("Analysis Time :", round(elapsed, 6), "seconds")
print()


# =====================================================
# NLoS Test
# =====================================================

print("=" * 60)
print("NLoS TEST")
print("=" * 60)

start = time.time()

received = NLoS_channel(sample)

prediction = model.predict(
    received.reshape(1, -1)
)

prediction = 1 if prediction[0] == -1 else 0

elapsed = time.time() - start

print("True Label    :", true_label)
print("Prediction    :", prediction)
print("Correct       :", prediction == true_label)
print("Analysis Time :", round(elapsed, 6), "seconds")
print()


# =====================================================
# Visualization
# =====================================================

colors = []

for p in pred:

    if p == 0:
        colors.append("blue")
    else:
        colors.append("red")


plt.figure(figsize=(12,6))

plt.scatter(
    range(len(pred)),
    X[:,0],
    c=colors,
    s=12
)

plt.title("Isolation Forest Anomaly Detection")
plt.xlabel("Audio Sample")
plt.ylabel("First MFCC Feature")

plt.grid(True)

plt.show()


print("=" * 60)
print("PROGRAM FINISHED SUCCESSFULLY")
print("=" * 60)