
import warnings
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from .ploty_template import plot_title
import pandas as pd
import numpy as np
from .eda import Documents
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import feature_selection
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
# nltk.download('vader_lexicon')
# import importlib.resources as pkg_resources
from . import utils
import os
from tqdm.autonotebook import tqdm
tqdm.pandas()

warnings.filterwarnings("ignore")

# import swifter

global UTIL_PATH
# with pkg_resources.path('utils', '.') as p:
#     UTIL_PATH = str(p)

UTIL_PATH = os.path.abspath(os.path.dirname(utils.__file__))


class Classifier:
    def __init__(self, documents_object, target_column=None, method=None, algo=None):
        if isinstance(documents_object, Documents):
            self.doc_obj = documents_object
            self.raw_df = documents_object.raw_df
            self.stop_words = documents_object.stop_words
            if documents_object.clean_status:
                self.processed_df = documents_object.processed_df
                self.text_column = documents_object.text_column                        
            else:
                # raise ValueError("Please run the prep_docs method on the Documents object first.")
                self.doc_obj.prep_docs()
                self.processed_df = self.doc_obj.processed_df
                self.text_column = self.doc_obj.text_column
                
            if str(target_column) in self.processed_df.columns:
                self.target_column = str(target_column)
            else:
                raise ValueError("Cannot find " +str(target_column) + " in the dataframe.")
            
            if method in ['tfidf', 'fasttext']:
                self.class_method = method
                if method == 'tfidf':
                    if algo in ['automl', 'custom']:
                        self.class_algo = algo
                    else:
                        raise ValueError("Please choose 'algo' as either of 'automl' or 'custom' ")
            else:
                raise ValueError("Please choose 'method' as either of 'tfidf' or 'fasttext' ")
        else:
            raise TypeError("Only an object of Documents Class can be passed.")
        
    def __create_features(self, p_value_limit):
        clean_text = self.text_column + "_clean"
        corpus = self.processed_df[clean_text]
        y = self.processed_df[self.target_column]
        
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        vectorizer.fit(corpus)
        X_train = vectorizer.transform(corpus)
        X_names = vectorizer.get_feature_names()
        
        dtf_features = pd.DataFrame()
        for cat in np.unique(y):
            chi2, p = feature_selection.chi2(X_train, y == cat)
            dtf_features = dtf_features.append(pd.DataFrame({"feature": X_names, "score": 1-p, "y": cat}))
            dtf_features = dtf_features.sort_values(["y", "score"], ascending=[True, False])
            dtf_features = dtf_features[dtf_features["score"] >= p_value_limit]
        X_names = dtf_features["feature"].unique().tolist()
        
        vectorizer = TfidfVectorizer(vocabulary=X_names)
        vectorizer.fit(corpus)
        X_transformed = vectorizer.transform(corpus)
        self.tfidf_obj = vectorizer
        self.doc_term_matrix = pd.DataFrame(data=X_transformed.todense(), columns=vectorizer.get_feature_names())
        
        return self.doc_term_matrix
    
    
    def classify(self, p_value_limit = 0.95):
        y = self.processed_df[self.target_column]
        if len(y.unique()) > 2:
            self.model_objective = "Muti-Class"
        else:
            self.model_objective = "Binary-Class"
        if self.class_method == 'tfidf':
            doc_term_df = self.__create_features(p_value_limit)
            if self.class_algo == 'automl':
                
                if self.model_objective == "Binary-Class":
                    pass