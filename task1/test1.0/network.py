from __future__ import print_function

import warnings

from keras.utils import to_categorical
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight

warnings.filterwarnings("ignore")

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Bidirectional, LSTM
from keras.optimizers import Adamax

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.utils.multiclass import type_of_target

class Classifier:
    def __init__(self, data, name="", batch_size=64):
        vectors = np.stack(data.iloc[:, 1].values)
        labels = data.iloc[:, 0].values
        positive_idxs = np.where(labels == 1)[0]
        negative_idxs = np.where(labels == 0)[0]
        undersampled_negative_idxs = np.random.choice(negative_idxs, len(positive_idxs), replace=False)
        resampled_idxs = np.concatenate([positive_idxs, undersampled_negative_idxs])

        X_train, X_test, y_train, y_test = train_test_split(vectors[resampled_idxs, ], labels[resampled_idxs],
                                                            test_size=0.2, stratify=labels[resampled_idxs])
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)
        self.name = name
        self.batch_size = batch_size
        self.class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=labels)
        
        model = Sequential()
        model.add(Bidirectional(LSTM(300), input_shape=(vectors.shape[1], vectors.shape[2])))
        model.add(Activation('relu'))
        model.add(Dense(2, activation='softmax'))
        
        adamax = Adamax(lr=0.002)
        model.compile(adamax, 'categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def train(self):
        self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, epochs=2, class_weight=self.class_weight)
        self.model.save_weights(self.name + "_model.h5")

    def test(self):
        self.model.load_weights(self.name + "_model.h5")
        values = self.model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size)
        print("Accuracy:", values[1])
        predictions = (self.model.predict(self.X_test, batch_size=self.batch_size)).round()

        tn, fp, fn, tp = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(predictions, axis=1)).ravel()

        # print('False positive rate:', fp / (fp + tn))
        # print('False negative rate:', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('recall:', recall)
        precision = tp / (tp + fp)
        print('Precision:', precision)
        print('F1 score:', (2 * precision * recall) / (precision + recall))

        fpr, tpr, thresholds = metrics.roc_curve(np.argmax(self.y_test, axis=1), np.argmax(predictions, axis=1))
        roc_auc = metrics.auc(fpr, tpr)
        print('AUC:', roc_auc)
        