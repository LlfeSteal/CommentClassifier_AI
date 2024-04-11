from typing import List

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras.callbacks import EarlyStopping
from transformers import TFBertForSequenceClassification, BertTokenizer


class Classifier():
    """The Classifier: template """

    def __init__(self):
        """ initialize the classifier, in particular the neural model """
        self.labelset = None
        self.model = None
        self.epoch = 200
        self.batchsize = 16
        self.label_binarizer = LabelBinarizer()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.max_features = 8000
        self.learning_rate = 1e-3
        self.dropout_rate = 0.08

    #############################################

    def preprocess_data(self, sentences, labels=None):
        return self.tokenizer(
            sentences,
            padding="max_length",
            max_length=self.max_features,
            return_tensors="tf"
        )

    def create_model(self):
        bert_model = TFBertForSequenceClassification.from_pretrained("asi/gpt-fr-cased-base")
        input_ids = tf.keras.layers.Input(shape=(self.max_features,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.layers.Input((self.max_features,), dtype=tf.int32, name='attention_mask')
        output = bert_model([input_ids, attention_mask])[0]
        output = tf.keras.layers.Dense(3, activation='softmax')(output)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=1e-08)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.model = tf.keras.models.Model(inputs=[input_ids, attention_mask], outputs=output)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    def train(self, texts: List[str], labels: List[str], dev_texts=None, dev_labels=None):
        """Trains the classifier model on the training set stored in file trainfile"""
        Y_train = self.label_binarizer.fit_transform(labels)
        self.labelset = set(self.label_binarizer.classes_)
        print("LABELS: %s" % self.labelset)
        X_train = self.preprocess_data(texts, labels)
        self.create_model()
        my_callbacks = []
        early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=0, mode='auto', baseline=None)
        my_callbacks.append(early_stopping)

        if dev_texts is not None and dev_labels is not None:
            X_dev = self.preprocess_data(dev_texts)
            Y_dev = self.label_binarizer.transform(dev_labels)
            dev_data = (X_dev, Y_dev)
        else:
            dev_data = None

        print("Xtrain", X_train)
        print("Ytrain", Y_train)
        self.model.fit(
            X_train,
            epochs=self.epoch,
            batch_size=self.batchsize,
            callbacks=my_callbacks,
            validation_data=dev_data,
            verbose=2)

    def predict(self, texts: List[str]) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels (same order as the input texts)
        """
        X = self.preprocess_data(texts)
        # get the predicted output vectors: each vector will contain a probability for each class label
        Y = self.model.predict(X)
        # from the output probability vectors, get the labels that got the best probability scores
        return self.label_binarizer.inverse_transform(Y)
