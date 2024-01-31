import pandas as pd
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


class Classifier:
    def __init__(self):
        self.labse_model = None
        self.model_nn = None
        self.label_encoder = None
        self.label_amount = None

        self.train()

    def train(self) -> None:
        train = pd.read_csv('data/retail/train.csv')

        self.label_amount = len(train['category'].unique())

        train_texts, test_texts, train_labels, test_labels = train_test_split(
            train['text'], train['category'], test_size=0.15, random_state=42)

        self.labse_model = SentenceTransformer('LaBSE')

        train_embeddings = self.labse_model.encode(train_texts.to_list()).astype(np.float32)
        test_embeddings = self.labse_model.encode(test_texts.to_list()).astype(np.float32)

        self.label_encoder = LabelEncoder()
        train_labels = self.label_encoder.fit_transform(train_labels.tolist())
        test_labels = self.label_encoder.transform(test_labels.tolist())

        self.model_nn = Sequential()
        self.model_nn.add(Dense(64, activation='relu', input_shape=(train_embeddings.shape[1],)))
        self.model_nn.add(Dense(len(self.label_encoder.classes_), activation='softmax'))

        self.model_nn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model_nn.fit(train_embeddings, train_labels, validation_data=(test_embeddings, test_labels), epochs=30,
                          batch_size=64)

        loss, accuracy = self.model_nn.evaluate(test_embeddings, test_labels)

        print('Val Loss:', loss)
        print('Val Accuracy:', accuracy)

    def test(self):

        test = pd.read_csv('data/retail/test.csv')

        X_test, y_test = test['text'], test['category']

        test_embeddings = self.labse_model.encode(X_test.to_list()).astype(np.float32)

        yprob = self.model_nn.predict(test_embeddings)
        yclasses = yprob.argmax(axis=-1)
        y_pred = self.label_encoder.inverse_transform(yclasses)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        confusion = confusion_matrix(y_test, y_pred)

        print('Test Accuracy:', accuracy)
        print('Test precision:', precision)
        print('Test recall:', recall)
        print('Test f1:', f1)


    def predict(self, text: str):
        test_embeddings = self.labse_model.encode([text]).astype(np.float32)

        yprob = self.model_nn.predict(test_embeddings)
        yclasses = yprob.argmax(axis=-1)
        intent = self.label_encoder.inverse_transform(yclasses)

        return intent
