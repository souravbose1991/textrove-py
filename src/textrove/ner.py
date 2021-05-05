
import spacy
import numpy as np
import pandas as pd
from PIL import Image
# import swifter
from bs4 import BeautifulSoup
from itertools import chain
import re
import unicodedata
from pathlib import Path
from .eda import Documents
# import contractions
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
# import importlib.resources as pkg_resources
from . import utils
import os
from tqdm.autonotebook import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings("ignore")

# Spacy
nlp = spacy.load('en_core_web_md')

global UTIL_PATH
# with pkg_resources.path('utils', '.') as p:
#     UTIL_PATH = str(p)

UTIL_PATH = os.path.abspath(os.path.dirname(utils.__file__))


class NER:
    def __init__(self, documents_object=None):
        if isinstance(documents_object, Documents):
            self.doc_obj = documents_object
            self.raw_df = documents_object.raw_df
            if documents_object.clean_status:
                self.processed_df = documents_object.processed_df
                self.text_column = documents_object.text_column
            else:
                # raise ValueError("Please run the prep_docs method on the Documents object first.")
                self.doc_obj.prep_docs()
                self.processed_df = self.doc_obj.processed_df
                self.text_column = self.doc_obj.text_column
        else:
            raise TypeError("Only an object of Documents Class can be passed.")

    ################## Text Cleaning ##################

    def __flatten(self, listOfLists):
        "Flatten one level of nesting"
        return list(chain.from_iterable(listOfLists))

    def __strip_html_tags(self, text):
        soup = BeautifulSoup(text, "html.parser")
        [s.extract() for s in soup(['iframe', 'script'])]
        stripped_text = soup.get_text()
        stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
        return stripped_text

    def __remove_accented_chars(self, text):
        text = unicodedata.normalize('NFKD', text).encode(
            'ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    # def __expand_contractions(self, text):
    #     return contractions.fix(text)

    def __remove_special_characters(self, text, remove_digits=False):
        pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
        text = re.sub(pattern, '', text)
        return text

    def __clean_text(self, document):
        # converting to text
        document = str(document)

        # strip HTML
        document = self.__strip_html_tags(document)

        # lower case
        # document = document.lower()

        # remove extra newlines (often might be present in really noisy text)
        document = document.translate(document.maketrans("\n\t\r", "   "))

        # remove accented characters
        document = self.__remove_accented_chars(document)
        document = re.sub(r"x+", "", document)
        document = re.sub(r"X+", "", document)
        document = re.sub(r"(<br/>)", "", document)
        document = re.sub(r"(<a).*(>).*(</a>)", "", document)
        document = re.sub(r"(&amp)", "", document)
        document = re.sub(r"(&gt)", "", document)
        document = re.sub(r"(&lt)", "", document)
        document = re.sub(r"(\xa0)", " ", document)

        # remove special characters and\or digits
        # insert spaces between special characters to isolate them
        special_char_pattern = re.compile(r'([{.(-)!}])')
        document = special_char_pattern.sub(" \\1 ", document)
        document = self.__remove_special_characters(document, remove_digits=False)

        # remove extra whitespace
        document = re.sub(' +', ' ', document)
        document = document.strip()

        # expand contractions
        # document = self.__expand_contractions(document)

        return document



