
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ploty_template import plot_title
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import pandas as pd
from PIL import Image
# import swifter
from bs4 import BeautifulSoup
from itertools import chain
import re
import unicodedata
from pathlib import Path
# import contractions
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# Spacy
import spacy
nlp = spacy.load('en_core_web_md')

global UTIL_PATH, STOP_WORD
UTIL_PATH = str(Path("../utils").resolve())

################## Stopwords list ##################
stop1 = [re.sub(r"(\|(.)+)|(\n)", "", x.lower())
         for x in open(UTIL_PATH +"/stopwords/"+"StopWords_Generic.txt", "r")]
stop2 = [re.sub(r"(\|(.)+)|(\n)", "", x.lower())
         for x in open(UTIL_PATH + "/stopwords/"+"StopWords_GenericLong.txt", "r")]
stop3 = [re.sub(r"(\|(.)+)|(\n)", "", x.lower())
         for x in open(UTIL_PATH + "/stopwords/"+"StopWords_DatesandNumbers.txt", "r")]

STOP_WORD = list(set(list(stopwords.words("english")) + list(STOPWORDS) + stop1 + stop2 + stop3))


class Documents:
    def __init__(self, stopwrd=None, data_frame=None, text_column=None):
        if stopwrd is None:
            self.stop_words = STOP_WORD
        else:
            self.stop_words = list(set(STOP_WORD + stopwrd))
        self.clean_status = False

        if (isinstance(data_frame, pd.DataFrame) & isinstance(text_column, str)):
            self.raw_df = data_frame
            self.processed_df = self.raw_df
        else:
            raise TypeError(
                "data_frame should be a dataframe and the text_column should be string")

        if str(text_column) in self.raw_df.columns:
            self.text_column = text_column
        else:
            raise ValueError("Cannot find " +
                             str(text_column) + " in the dataframe.")

    ################## Text Cleaning ##################

    def __flatten(self, listOfLists):
        "Flatten one level of nesting"
        return list(chain.from_iterable(listOfLists))

    # Lemmatize with POS Tag
    # def __get_wordnet_pos(self, word):
    #     """Map POS tag to first character lemmatize() accepts"""
    #     tag = nltk.pos_tag([word])[0][1][0].upper()
    #     tag_dict = {"J": wordnet.ADJ,
    #                 "N": wordnet.NOUN,
    #                 "V": wordnet.VERB,
    #                 "R": wordnet.ADV}
    #     return tag_dict.get(tag, wordnet.NOUN)

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
        document = document.lower()

        # remove extra newlines (often might be present in really noisy text)
        document = document.translate(document.maketrans("\n\t\r", "   "))

        # remove accented characters
        document = self.__remove_accented_chars(document)
        document = re.sub(r"x+", "", document)
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
        document = self.__remove_special_characters(
            document, remove_digits=True)

        # remove extra whitespace
        document = re.sub(' +', ' ', document)
        document = document.strip()

        # expand contractions
        # document = self.__expand_contractions(document)

        # Split the documents into tokens.
        tokenizer = RegexpTokenizer(r'\w+')
        document = document.lower()  # Convert to lowercase.
        document = tokenizer.tokenize(document)  # Split into words.

        # Remove numbers, but not words that contain numbers.
        # Remove words that are only one character.
        # Remove stopwords
        document = [token for token in document if not token.isnumeric()]
        document = [token for token in document if len(document) > 1]
        document = [word for word in document if not word in self.stop_words]
        document = " ".join(document)

        # lemmatize
        document = [token.lemma_ for token in nlp(document) if not token.is_stop]
        document = [word for word in document if not word in self.stop_words]
        document = " ".join(document)

        return document

    def prep_docs(self, return_df=True):
        cleaned_text = str(self.text_column) + "_clean"
        self.processed_df[cleaned_text] = self.processed_df[self.text_column].apply(lambda x: self.__clean_text(x))
        self.clean_status = True
        if return_df:
            return self.processed_df

    def __get_ngrams(self, corpus, nwords=20, min_freq=0.05, ngram=1, most_frequent_first=True):
        vec = CountVectorizer(ngram_range=(ngram, ngram)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)

        min1 = int(min_freq*len(corpus))
        if ngram == 1:
            wordsets = [frozenset(document.split(" ")) for document in corpus]
        else:
            wordsets = [document for document in corpus]

        words_freq = []
        for word, idx in vec.vocabulary_.items():
            wrd_doc_cnt = sum(1 for s in wordsets if word in s)
            if wrd_doc_cnt >= min1:
                words_freq.append((word, sum_words[0, idx]))

        if most_frequent_first:
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        else:
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=False)
        temp_df = pd.DataFrame(words_freq[:nwords], columns=['Phrase', 'Count'])
        return temp_df

    def __plot_cloud(self, wordcloud):
        # Set figure size
        plt.figure(figsize=(40, 20))
        # Display image
        plt.imshow(wordcloud)
        # No axis details
        plt.axis("off")

    ##################### Exploratory Data Analysis #####################

    def explore(self, ngram_range=(1, 1), nwords=20, min_freq=0.05):
        cleaned_text = str(self.text_column) + "_clean"
        if not self.clean_status:
            self.prep_docs(return_df=False)

        if (isinstance(ngram_range, tuple)) & (ngram_range[0] > 0) & (ngram_range[1] >= ngram_range[0]):
            self.ngram_range = ngram_range
        else:
            raise TypeError("ngram_range has to be a tuple with first_term / lower-limit less-than or equal-to the second term / upper-limit")

        for ng in range(self.ngram_range[0], self.ngram_range[1]+1):
            if ng == 1:
                temp_df1 = self.__get_ngrams(self.processed_df[cleaned_text], nwords=nwords, min_freq=min_freq,
                                            ngram=ng, most_frequent_first=True)
                temp_df1 = temp_df1.groupby('Phrase').sum()['Count'].sort_values(ascending=False).reset_index()
                fig1 = go.Figure(data=[go.Bar(x=temp_df1['Phrase'], y=temp_df1['Count'])])
                fig1.update_layout(template="plotly_white", title_text=plot_title("Frequent Words"))
                fig1.show()
                temp_df2 = self.__get_ngrams(self.processed_df[cleaned_text], nwords=nwords, min_freq=min_freq,
                                            ngram=ng, most_frequent_first=False)
                temp_df2 = temp_df2.groupby('Phrase').sum()['Count'].sort_values(ascending=False).reset_index()
                fig2 = go.Figure(data=[go.Bar(x=temp_df2['Phrase'], y=temp_df2['Count'])])
                fig2.update_layout(template="plotly_white", title_text=plot_title("Rare Words"))
                fig2.show()
            else:
                temp_df1 = self.__get_ngrams(self.processed_df[cleaned_text], nwords=nwords, min_freq=min_freq,
                                            ngram=ng, most_frequent_first=True)
                temp_df1 = temp_df1.groupby('Phrase').sum()['Count'].sort_values(ascending=False).reset_index()
                fig1 = go.Figure(data=[go.Bar(x=temp_df1['Phrase'], y=temp_df1['Count'])])
                fig1.update_layout(template="plotly_white", title_text=plot_title("Frequent Phrases"))
                fig1.show()
                temp_df2 = self.__get_ngrams(self.processed_df[cleaned_text], nwords=nwords, min_freq=min_freq,
                                            ngram=ng, most_frequent_first=False)
                temp_df2 = temp_df2.groupby('Phrase').sum()['Count'].sort_values(ascending=False).reset_index()
                fig2 = go.Figure(data=[go.Bar(x=temp_df2['Phrase'], y=temp_df2['Count'])])
                fig2.update_layout(template="plotly_white",title_text=plot_title("Rare Phrases"))
                fig2.show()

    def create_wordcloud(self):
        cleaned_text = str(self.text_column) + "_clean"
        if not self.clean_status:
            self.prep_docs(return_df=False)
            
        # Import image to np.array & Generate word cloud
        mask = np.array(Image.open(UTIL_PATH + "/wordcloud_mask/" + "comment.png"))
        text = " ".join(self.processed_df[cleaned_text].tolist())
        wordcloud = WordCloud(width=400, height=200, random_state=1, background_color='white', colormap='tab10', collocations=False,
                              stopwords=self.stop_words, mask=mask).generate(text)
        self.__plot_cloud(wordcloud)









