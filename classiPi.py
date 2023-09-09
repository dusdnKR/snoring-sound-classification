import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

tf.compat.v1.disable_eager_execution()

audio_file_path = "juntae_sleep1.wav"
sleeping_score = 100

duration = 0.1  # seconds
sample_rate = 44100

def extract_features(audio_file_path):
    X, _ = librosa.load(audio_file_path, sr=sample_rate)
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
    snore_record = []
    time_points = []
    sleeping_scores = []

    X, _ = librosa.load(audio_file_path, sr=sample_rate)
    total_duration = len(X) / sample_rate

    while total_duration >= 0:
        current_time = time.time()
        elapsed_time = current_time - start_time
        print(elapsed_time, total_duration)
        print()

        feat = extract_features(audio_file_path)
        feat = sc.transform(feat)
        y_pred = sess.run(tf.argmax(y_, 1), feed_dict={X: feat})

        if y_pred == 1:
            if not snoring_detected:
                snoring_detected = True
                snore_start_time = current_time
                snore_record.append(time.strftime('%H:%M:%S'))
                print(time.strftime('%H:%M:%S'))
        else:
            if snoring_detected and (current_time - snore_start_time >= 5.0):
                snoring_duration = current_time - snore_start_time
                snoring_time += snoring_duration

                # Sleeping Score 감점
                sleeping_score = sleeping_score - snoring_duration * 0.05

                time_points.append(elapsed_time)
                sleeping_scores.append(sleeping_score)

            if elapsed_time >= total_duration:
                break

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, sleeping_scores, marker='o', linestyle='-', color='b')
    plt.title('Sleeping Score')
    plt.xlabel('Time (s)')
    plt.ylabel('Sleeping Score')
    plt.set_ylim(60, 100)
    plt.grid(True)
    plt.savefig('score_graph.png')
    plt.show()

    print("Final Sleeping Score:", sleeping_score)
    print("Total Snoring Time (s):", snoring_time)
    print("Done")
