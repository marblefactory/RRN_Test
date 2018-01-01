from dataset import GoogleDataSource
import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf
import tensorflow.contrib.learn as learn
from tensorflow.contrib import rnn

MAX_SENTENCE_LENGTH = 10

if __name__ == "__main__":
    data_source = GoogleDataSource()

    movement_training = data_source.movement_training()

    training_sentences = pandas.DataFrame(movement_training.sentence_words)[1]
    training_targets = pandas.Series(movement_training.targets)

    print(training_sentences)
    print(training_targets)

    vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_SENTENCE_LENGTH)


    x_train = np.array(list(vocab_processor.fit_transform(x)))
    n_words = len(vocab_processor.vocabulary_)

    print(x_train)
    print(n_words)