import gspread
from oauth2client.service_account import ServiceAccountCredentials
from typing import List
from abc import abstractclassmethod
import numpy as np


class Dataset:
    """
    Represents a collection of data and its corresponding target category.
    """

    # An array containing an array of words in each sentence.
    sentence_words: List[List[str]]
    # An array of integers corresponding to each sentence target.
    targets: List[int]

    def __init__(self, sentences: List[str], targets: List[int]):
        self.targets = targets
        self.sentence_words: List[List[str]] = []

        for sentence in sentences:
            words: List[str] = []
            for word in sentence.split():
                words.append(word)

            self.sentence_words.append(words)


class DataSource:
    """
    Used to fetch raw data in the form of sentences in different categories.
    """

    @abstractclassmethod
    def movement_training(self) -> Dataset:
        raise NotImplemented

    @abstractclassmethod
    def movement_testing(self) -> Dataset:
        raise NotImplemented


class GoogleDataSource(DataSource):
    """
    Used to fetch raw data from Google Sheets in the form of sentences in different categories.
    """

    def __init__(self, credentials_file='client_secret.json'):
        scope = ['https://spreadsheets.google.com/feeds']
        credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
        self.client = gspread.authorize(credentials)

    def movement_training(self) -> Dataset:
        rows = self.client.open("MovementTrainingData").sheet1.get_all_records()

        sentences: List[str] = []
        targets: List[int] = []

        for i, row in enumerate(rows):
            for sentence in list(row.values())[1:]:
                sentences.append(sentence)
                targets.append(i)

        return Dataset(sentences, targets)

