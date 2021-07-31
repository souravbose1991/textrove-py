
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.corpus import stopwords
from wordcloud import STOPWORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
from .eda import Documents
from . import utils
import os
import re
from tqdm.autonotebook import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings("ignore")

nlp = spacy.load('en_core_web_md')

global UTIL_PATH, STOP_WORD
# with pkg_resources.path('utils', '.') as p:
#     UTIL_PATH = str(p)

UTIL_PATH = os.path.abspath(os.path.dirname(utils.__file__))

################## Stopwords list ##################
stop1 = [re.sub(r"(\|(.)+)|(\n)", "", x.lower()).strip()
         for x in open(UTIL_PATH + "/stopwords/"+"StopWords_Generic.txt", "r")]
stop2 = [re.sub(r"(\|(.)+)|(\n)", "", x.lower()).strip()
         for x in open(UTIL_PATH + "/stopwords/"+"StopWords_GenericLong.txt", "r")]
stop3 = [re.sub(r"(\|(.)+)|(\n)", "", x.lower()).strip()
         for x in open(UTIL_PATH + "/stopwords/"+"StopWords_DatesandNumbers.txt", "r")]

STOP_WORD = list(set(list(stopwords.words("english")) + list(STOPWORDS) + list(STOP_WORDS) + stop1 + stop2 + stop3))

class Summary:
    def __init__(self, documents_object=None, summary_size=None):
        if isinstance(documents_object, Documents):
            self.doc_obj = documents_object
            self.raw_df = documents_object.raw_df
            if not (isinstance(summary_size, int) or isinstance(summary_size, float)):
                raise TypeError("Summary Size has to be either an Interger or fraction less than 1")
            else:
                if summary_size is None:
                    self.summary_size = 0.3
                else: 
                    self.summary_size = summary_size
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

    def __summarize(self, text, size):
        doc = nlp(text)
        ori_sents = len(list(doc.sents))
        if size < 1.0:
            target_sents = round(ori_sents*size, 0)
        else:
            target_sents = size
        if target_sents > 0:
            keyword = []
            pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
            for token in doc:
                if(token.text in STOP_WORD or token.text in punctuation):
                    continue
                if(token.pos_ in pos_tag):
                    keyword.append(token.text)
            freq_word = Counter(keyword)
            max_freq = Counter(keyword).most_common(1)[0][1]
            for word in freq_word.keys():
                freq_word[word] = (freq_word[word]/max_freq)
            sent_strength={}
            for sent in doc.sents:
                for word in sent:
                    if word.text in freq_word.keys():
                        if sent in sent_strength.keys():
                            sent_strength[sent]+=freq_word[word.text]
                        else:
                            sent_strength[sent]=freq_word[word.text]
            summarized_sentences = nlargest(target_sents, sent_strength, key=sent_strength.get) 
            final_sentences = [w.text for w in summarized_sentences ]
            summary = ' '.join(final_sentences)
        else:
            summary = text
        return summary

    def __get_summary(self, x, text_column=None):
        if text_column is None:
            text_column = str(self.text_column)
        summary_text = str(text_column) + "_summary"
        try:
            summ = self.__summarize(text=x[text_column], size=self.summary_size)
        except:
            summ = x[text_column]
        summ = re.sub('\n', ' ', summ)
        summ = summ.strip()
        x[summary_text] = summ
        return x

    def generate_results(self):
        temp_df = self.processed_df
        temp_df = temp_df.progress_apply(lambda x: self.__get_summary(x), axis=1)
        self.processed_df = temp_df
        return temp_df







