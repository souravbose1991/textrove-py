
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

    def __get_entity(self, x, text_column=None):
        if text_column is None:
            text_column = str(self.text_column)
        entity_text = str(text_column) + "_entity"
        doc = nlp(x[text_column])
        ent_list = [(ent.text, ent.label_) for ent in doc.ents]
        ner_txt = '--'.join(ent_list)
        ner_txt = re.sub('\n', ' ', ner_txt)
        ner_txt = ner_txt.strip()
        x[entity_text] = ner_txt
        return x

    def generate_ner(self):
        temp_df = self.processed_df
        temp_df = temp_df.progress_apply(lambda x: self.__get_entity(x), axis=1)
        self.processed_df = temp_df
        return temp_df




