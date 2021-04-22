
from gensim.summarization import summarize, keywords
from .eda import Documents
from . import utils
import os

# # Spacy
# import spacy
# nlp = spacy.load('en_core_web_md')
# from spacy.lang.en.stop_words import STOP_WORDS
# from string import punctuation
# from collections import Counter
# from heapq import nlargest

UTIL_PATH = os.path.abspath(os.path.dirname(utils.__file__))

class Summary:
    def __init__(self, documents_object=None, method=None, summary_ratio=None, keyword_ratio=None):
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

            if method in ['summary', 'all']:
                self.method = method
                if summary_ratio is None:
                    summary_ratio = 0.4
                if (summary_ratio > 0 & summary_ratio < 1.0):
                    self.summary_ratio = summary_ratio
                else:
                    raise ValueError("Summary-Ratio should be between (0, 1) non-inclusive rage")
            else:
                raise ValueError("Please choose method as either of summary / keyword / all")
            if method in ['keyword', 'all']:
                self.method = method
                if keyword_ratio is None:
                    keyword_ratio = 0.4
                if (keyword_ratio > 0 & keyword_ratio < 1.0):
                    self.keyword_ratio = keyword_ratio
                else:
                    raise ValueError("KeyWord-Ratio should be between (0, 1) non-inclusive rage")
            else:
                raise ValueError("Please choose method as either of summary / keyword / all")
        else:
            raise TypeError("Only an object of Documents Class can be passed.")

    def __get_summary(self, x):
        summary_text = str(self.text_column) + "_summary"
        x[summary_text] = summarize(x[self.text_column], ratio=self.summary_ratio)
        return x

    def __get_keyword(self, x):
        keyword_text = str(self.text_column) + "_keyword"
        x[keyword_text] = keywords(x[self.text_column], ratio=self.keyword_ratio)
        return x

    def generate_results(self):
        temp_df = self.processed_df
        if self.method == 'summary':
            temp_df = temp_df.apply(lambda x: self.__get_summary(x), axis=1)
        elif self.method == 'keyword':
            temp_df = temp_df.apply(lambda x: self.__get_keyword(x), axis=1)
        else:
            temp_df = temp_df.apply(lambda x: self.__get_summary(x), axis=1)
            temp_df = temp_df.apply(lambda x : self.__get_keyword(x), axis=1)
        self.processed_df = temp_df
        return temp_df







