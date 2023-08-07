import glob
import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# print(tf)

# 그래프 모드로 전환
tf.compat.v1.disable_eager_execution()

'''Folder 1 contains snoring sounds
Folder 0 contains non-snoring sounds'''

# Extract audio file features
def extract_features(file_name):
    X, sample_rate = librosa.load(file_name) # Load audio file, return audio signal X and sample_rate
    stft = np.abs(librosa.stft(X)) # Calc Short-Time Fourier Transform(STFT)
    mfccs = np.array(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=8).T) # Calc Mel-frequency cepstral coefficients(MFCC)
    chroma = np.array(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T) # Calc Chroma feature (Tone Information)
    mel = np.array(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_fft=256).T) # Calc Mel spectrogram (Frequency distribution)
    contrast = np.array(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T) # Calc Spectral Contrast (Frequency band contrast)
    tonnetz = np.array(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T) # Calc Harmonic signal에서 Tonnetz (Harmony)
    return mfccs, chroma, mel, contrast, tonnetz

def get_label(fn):
    return sub_dirs.index(fn)

# Process audio files, extract features and labels
def parse_audio_files(parent_dir, sub_dirs, file_ext='*.wav'):
    ignored = 0
    features, labels, name = np.empty((0, 161)), np.empty(0), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        print("Processing folder..", sub_dir)
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                mfccs, chroma, mel, contrast, tonnetz = extract_features(fn) # Feature extraction
                ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
                features = np.vstack([features, ext_features])
                fnlabel = get_label(sub_dir)
                l = [fnlabel] * (mfccs.shape[0])
                labels = np.append(labels, l)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                print(fn, e)
                ignored += 1
    print("Ignored files: ", ignored)
    return np.array(features), np.array(labels, dtype=int)

# one hot encoding
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

# Path Settings
parent_dir = 'SnoringDataset'
sub_dirs = os.listdir(parent_dir)

try:
    labels = np.load('labels.npy')
    features = np.load('features.npy')
    print("Features and labels found!")
except:
    print("Extracting features...")
    features, labels = parse_audio_files(parent_dir, sub_dirs)
    with open('features.npy', 'wb') as f1:
        np.save(f1, features)
    with open('labels.npy', 'wb') as f2:
        np.save(f2, labels)

labels = one_hot_encode(labels)

print("Splitting and fitting!")

# training set : test set = 9 : 1
train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.1, train_size=0.9, random_state=0)
sc = StandardScaler()
sc.fit(train_x)
with open("fit_params.npy", "wb") as f3:
    np.save(f3, train_x)
train_x = sc.transform(train_x)
test_x = sc.transform(test_x)

print("Training...")

#### Training Neural Network with TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Hyperparameter tuning
training_epochs = 500
n_dim = features.shape[1]
n_classes = 2
n_hidden_units_one = 256
n_hidden_units_two = 256
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.1
model_path = os.path.join(os.getcwd(), "model", "model")

X = tf.compat.v1.placeholder(tf.float32, [None, n_dim])
Y = tf.compat.v1.placeholder(tf.float32, [None, n_classes])

# Layer 1
W_1 = tf.Variable(tf.random.normal([n_dim, n_hidden_units_one], mean=0, stddev=sd))
b_1 = tf.Variable(tf.random.normal([n_hidden_units_one], mean=0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X, W_1) + b_1)  # Use tanh

# Layer 2
W_2 = tf.Variable(tf.random.normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd))
b_2 = tf.Variable(tf.random.normal([n_hidden_units_two], mean = 0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2 ) # Use sigmoid

W = tf.Variable(tf.random.normal([n_hidden_units_two, n_classes], mean=0, stddev=sd))
b = tf.Variable(tf.random.normal([n_classes], mean=0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2, W) + b) # Use softmax

init = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver()

# cost function
cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.math.log(y_), axis=1))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# Calc accuracy
correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_size = 10000
patience_cnt = 0
patience = 16
min_delta = 0.01
stopping = 0

cost_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None
with tf.compat.v1.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        if stopping == 0:
            total_batch = int(train_x.shape[0] / batch_size)  # Convert total_batch to int type
            train_x, train_y = shuffle(train_x, train_y, random_state=42)
            for i in range(total_batch):
                batch_x = train_x[i*batch_size:i*batch_size+batch_size]
                batch_y = train_y[i*batch_size:i*batch_size+batch_size]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, cost = sess.run([optimizer, cost_function], feed_dict={X: batch_x, Y: batch_y})
            cost_history = np.append(cost_history, cost)
            if epoch % 100 == 0:
                print("Epoch: ", epoch, " cost ", cost)
            if epoch > 0 and abs(cost_history[epoch-1] - cost_history[epoch]) > min_delta:
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt > patience:
                print("Early stopping at epoch ", epoch, ", cost ", cost)
                stopping = 1

    y_pred = sess.run(tf.argmax(y_,1), feed_dict={X: test_x})
    y_true = sess.run(tf.argmax(test_y,1))
    #saving model
    save_path = saver.save(sess, model_path)
    print("Model saved at: %s" % save_path)

# Performance Evaluation
p, r, f, s = precision_recall_fscore_support(y_true, y_pred) #average='micro'
print("Precision:", p)
print("Recall:", r)
print("F-Score:", f)