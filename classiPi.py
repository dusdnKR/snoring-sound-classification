import librosa
import numpy as np
import tensorflow as tf
import sounddevice
from sklearn.preprocessing import StandardScaler
import time

tf.compat.v1.disable_eager_execution()

duration = 0.1  # seconds
sample_rate = 44100

def extract_features():
    X = sounddevice.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sounddevice.wait()
    X = np.squeeze(X)
    stft = np.abs(librosa.stft(X))
    mfccs = np.array(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=8).T)
    chroma = np.array(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T)
    mel = np.array(librosa.feature.melspectrogram(y=X, sr=sample_rate).T)
    contrast = np.array(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T)
    tonnetz = np.array(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T)
    ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    features = ext_features
    return features

model_path = "model"
fit_params = np.load('fit_params.npy', allow_pickle=True)
sc = StandardScaler()
sc.fit(fit_params)

n_dim = 161
n_classes = 2
n_hidden_units_one = 256
n_hidden_units_two = 256
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01

X = tf.keras.Input(shape=(n_dim,))
Y = tf.keras.Input(shape=(n_classes,))

W_1 = tf.Variable(tf.random.normal([n_dim, n_hidden_units_one], mean=0, stddev=sd))
b_1 = tf.Variable(tf.random.normal([n_hidden_units_one], mean=0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X, W_1) + b_1)

W_2 = tf.Variable(tf.random.normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd))
b_2 = tf.Variable(tf.random.normal([n_hidden_units_two], mean=0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)

W = tf.Variable(tf.random.normal([n_hidden_units_two, n_classes], mean=0, stddev=sd))
b = tf.Variable(tf.random.normal([n_classes], mean=0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2, W) + b)

init = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver()

y_true, y_pred = None, None

with tf.compat.v1.Session() as sess:
    saver.restore(sess, model_path)
    print("Model loaded")

    snoring_time = 0
    start_time = time.time()
    snoring_detected = False
    snore_start_time = None

    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time >= 8 * 60 * 60:  # 측정 시간
            print("Total snoring time (seconds):", snoring_time)
            break

        feat = extract_features()
        feat = sc.transform(feat)
        y_pred = sess.run(tf.argmax(y_, 1), feed_dict={X: feat})

        if y_pred == 1:
            if not snoring_detected:
                snoring_detected = True
                snore_start_time = current_time
        else:
            if snoring_detected and (current_time - snore_start_time >= 5.0):
                snoring_time += current_time - snore_start_time
                snoring_detected = False

    print("Done")