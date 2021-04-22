
# from eda.eda import Documents
from gensim.summarization import summarize, keywords
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from template.ploty_template import plot_title
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

    def prep_docs(self, return_df=False):
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




class Sentiment:
    def __init__(self, documents_object=None, method=None, lexicon=None):
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

            if method in ['lexical', 'textblob', 'vader']:
                self.sent_method = method
                if method == 'lexical':
                    if lexicon in ['loughran', 'nrc']:
                        self.sent_lexicon = lexicon
                        self.lexi_dict = {}
                    else:
                        raise ValueError(
                            "Please choose lexicon as either of loughran / nrc")
            else:
                raise ValueError(
                    "Please choose method as either of lexical / textblob / vader")
        else:
            raise TypeError("Only an object of Documents Class can be passed.")

    def __load_loughran(self):
        #Load master dictionary
        master = pd.read_csv(UTIL_PATH + "/lexicons/loughran/" +
                             "LoughranMcDonald_MasterDictionary_2018.csv")

        # Henry's (2008) Word List
        # Henry, Elaine. “Are Investors Influenced By How Earnings Press Releases Are Written.” The Journal of Business
        # Communication (1973) 45, no. 4 (2008): 363–407.
        hdict = {'Negative': ['negative', 'negatives', 'fail', 'fails', 'failing', 'failure', 'weak', 'weakness', 'weaknesses',
                              'difficult', 'difficulty', 'hurdle', 'hurdles', 'obstacle', 'obstacles', 'slump', 'slumps',
                              'slumping', 'slumped', 'uncertain', 'uncertainty', 'unsettled', 'unfavorable', 'downturn',
                              'depressed', 'disappoint', 'disappoints', 'disappointing', 'disappointed', 'disappointment',
                              'risk', 'risks', 'risky', 'threat', 'threats', 'penalty', 'penalties', 'down', 'decrease',
                              'decreases', 'decreasing', 'decreased', 'decline', 'declines', 'declining', 'declined', 'fall',
                              'falls', 'falling', 'fell', 'fallen', 'drop', 'drops', 'dropping', 'dropped', 'deteriorate',
                              'deteriorates', 'deteriorating', 'deteriorated', 'worsen', 'worsens', 'worsening', 'weaken',
                              'weakens', 'weakening', 'weakened', 'worse', 'worst', 'low', 'lower', 'lowest', 'less', 'least',
                              'smaller', 'smallest', 'shrink', 'shrinks', 'shrinking', 'shrunk', 'below', 'under', 'challenge',
                              'challenges', 'challenging', 'challenged'],
                 'Positive': ['positive', 'positives', 'success', 'successes', 'successful', 'succeed', 'succeeds',
                              'succeeding', 'succeeded', 'accomplish', 'accomplishes', 'accomplishing', 'accomplished',
                              'accomplishment', 'accomplishments', 'strong', 'strength', 'strengths', 'certain', 'certainty',
                              'definite', 'solid', 'excellent', 'good', 'leading', 'achieve', 'achieves', 'achieved',
                              'achieving', 'achievement', 'achievements', 'progress', 'progressing', 'deliver', 'delivers',
                              'delivered', 'delivering', 'leader', 'leading', 'pleased', 'reward', 'rewards', 'rewarding',
                              'rewarded', 'opportunity', 'opportunities', 'enjoy', 'enjoys', 'enjoying', 'enjoyed',
                              'encouraged', 'encouraging', 'up', 'increase', 'increases', 'increasing', 'increased', 'rise',
                              'rises', 'rising', 'rose', 'risen', 'improve', 'improves', 'improving', 'improved', 'improvement',
                              'improvements', 'strengthen', 'strengthens', 'strengthening', 'strengthened', 'stronger',
                              'strongest', 'better', 'best', 'more', 'most', 'above', 'record', 'high', 'higher', 'highest',
                              'greater', 'greatest', 'larger', 'largest', 'grow', 'grows', 'growing', 'grew', 'grown', 'growth',
                              'expand', 'expands', 'expanding', 'expanded', 'expansion', 'exceed', 'exceeds', 'exceeded',
                              'exceeding', 'beat', 'beats', 'beating']}

        negate = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt", "ain't", "aren't", "can't",
                  "couldn't", "daren't", "didn't", "doesn't", "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt",
                  "neither", "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "neednt", "needn't",
                  "never", "none", "nope", "nor", "not", "nothing", "nowhere", "oughtnt", "shant", "shouldnt", "wasnt",
                  "werent", "oughtn't", "shan't", "shouldn't", "wasn't", "weren't", "without", "wont", "wouldnt", "won't",
                  "wouldn't", "rarely", "seldom", "despite", "no", "nobody"]

        pos_words = master[master["Positive"] > 0].Word.str.lower().tolist()
        pos_words = list(set(pos_words + hdict['Positive']))
        neg_words = master[master["Negative"] > 0].Word.str.lower().tolist()
        neg_words = list(set(neg_words + hdict['Negative']))
        unc_words = master[master["Uncertainty"] > 0].Word.str.lower().tolist()
        lit_words = master[master["Litigious"] > 0].Word.str.lower().tolist()
        con_words = master[master["Constraining"]
                           > 0].Word.str.lower().tolist()
        sup_words = master[master["Superfluous"] > 0].Word.str.lower().tolist()
        comb_list = [pos_words, neg_words, unc_words,
                     lit_words, con_words, sup_words, negate]
        return comb_list

    def __sent_loughran(self, x, loughran_list):
        cleaned_text = str(self.text_column) + "_clean"
        pos_cnt = neg_cnt = unc_cnt = lit_cnt = con_cnt = sup_cnt = 0
        wrd_list = x[cleaned_text].split()
        word_cnt = len(wrd_list)
        for i in range(0, word_cnt):
            mini_list = wrd_list[max(i-3, 0):i]
            if wrd_list[i] in loughran_list[0]:
                if any(item in loughran_list[6] for item in mini_list):
                    neg_cnt += 1
                else:
                    pos_cnt += 1
            if wrd_list[i] in loughran_list[1]:
                if any(item in loughran_list[6] for item in mini_list):
                    pos_cnt += 1
                else:
                    neg_cnt += 1
            if wrd_list[i] in loughran_list[2]:
                unc_cnt += 1
            if wrd_list[i] in loughran_list[3]:
                lit_cnt += 1
            if wrd_list[i] in loughran_list[4]:
                con_cnt += 1
            if wrd_list[i] in loughran_list[5]:
                sup_cnt += 1
        if (pos_cnt + neg_cnt + unc_cnt + lit_cnt + con_cnt + sup_cnt) > 0:
            x['pos_shr'] = 100*pos_cnt / \
                (pos_cnt + neg_cnt + unc_cnt + lit_cnt + con_cnt + sup_cnt)
            x['neg_shr'] = 100*neg_cnt / \
                (pos_cnt + neg_cnt + unc_cnt + lit_cnt + con_cnt + sup_cnt)
            x['unc_shr'] = 100*unc_cnt / \
                (pos_cnt + neg_cnt + unc_cnt + lit_cnt + con_cnt + sup_cnt)
            x['lit_shr'] = 100*lit_cnt / \
                (pos_cnt + neg_cnt + unc_cnt + lit_cnt + con_cnt + sup_cnt)
            x['con_shr'] = 100*con_cnt / \
                (pos_cnt + neg_cnt + unc_cnt + lit_cnt + con_cnt + sup_cnt)
            x['sup_shr'] = 100*sup_cnt / \
                (pos_cnt + neg_cnt + unc_cnt + lit_cnt + con_cnt + sup_cnt)
        else:
            x['pos_shr'] = x['neg_shr'] = x['unc_shr'] = x['lit_shr'] = x['con_shr'] = x['sup_shr'] = 0.0
        if (pos_cnt + neg_cnt) > 0:
            x['sent_scr'] = (pos_cnt - neg_cnt) / (pos_cnt + neg_cnt)
        else:
            x['sent_scr'] = 0.0
        return x

    def __plot_loughran(self, temp_df, X_variable=None):
        if X_variable is None:
            avg_sent = round(temp_df['sent_scr'].mean(), 2)
            avg_pos = round(temp_df['pos_shr'].mean(), 1)
            avg_neg = round(temp_df['neg_shr'].mean(), 1)
            avg_unc = round(temp_df['unc_shr'].mean(), 1)
            avg_lit = round(temp_df['lit_shr'].mean(), 1)
            avg_con = round(temp_df['con_shr'].mean(), 1)
            avg_sup = round(temp_df['sup_shr'].mean(), 1)
            labels = ['Positive', 'Negative', 'Uncertainty',
                      'Litigious', 'Constraining', 'Superfluous']
            values = [avg_pos, avg_neg, avg_unc, avg_lit, avg_con, avg_sup]
            fig = go.Figure(data=[go.Pie(
                labels=labels, values=values, hole=.4, hoverinfo="label+percent+name")])
            fig.update_layout(template="plotly_white", title_text=plot_title(
                "Overall Sentiment Score: " + str(avg_sent)))
            fig.show()
        else:
            tdf = temp_df[[X_variable, 'sent_scr', 'pos_shr', 'neg_shr', 'unc_shr',
                           'lit_shr', 'con_shr', 'sup_shr']].groupby([X_variable]).mean()
            tdf = tdf.sort_index().reset_index()
            fig = make_subplots(rows=3, cols=1, specs=[[{}], [{"rowspan": 2}], [None]],
                                shared_xaxes=True, vertical_spacing=0.00)
            fig.add_trace(go.Scatter(x=tdf[X_variable], y=tdf['sent_scr'], mode='lines+markers',
                                     line_shape='spline', name='Sentiment Score'), row=1, col=1)
            fig.add_trace(go.Bar(x=tdf[X_variable], y=tdf['pos_shr'].round(
                1), name='Positive'), row=2, col=1)
            fig.add_trace(go.Bar(x=tdf[X_variable], y=tdf['neg_shr'].round(
                1), name='Negative'), row=2, col=1)
            fig.add_trace(go.Bar(x=tdf[X_variable], y=tdf['unc_shr'].round(
                1), name='Uncertainty'), row=2, col=1)
            fig.add_trace(go.Bar(x=tdf[X_variable], y=tdf['lit_shr'].round(
                1), name='Litigious'), row=2, col=1)
            fig.add_trace(go.Bar(x=tdf[X_variable], y=tdf['con_shr'].round(
                1), name='Constraining'), row=2, col=1)
            fig.add_trace(go.Bar(x=tdf[X_variable], y=tdf['sup_shr'].round(
                1), name='Superfluous'), row=2, col=1)
            fig.update_layout(barmode='stack', yaxis_visible=False, yaxis_showticklabels=False,
                              yaxis2_visible=False, yaxis2_showticklabels=False, template="plotly_white",
                              title_text=plot_title("Sentiment Analysis"))
            fig.show()

    def __generate_sentiment(self):
        temp_df = self.processed_df
        if self.sent_method == 'lexical':
            if self.sent_lexicon == 'loughran':
                loughran_list = self.__load_loughran()
                self.lexi_dict['Positive'] = loughran_list[0]
                self.lexi_dict['Negative'] = loughran_list[1]
                temp_df = temp_df.apply(lambda x: self.__sent_loughran(
                    x, loughran_list=loughran_list), axis=1)
        self.processed_df = temp_df
        return temp_df

    def __generate_word_sentiment(self, temp_df, min_freq=0.05, lexi_dict=None):
        cleaned_text = str(self.text_column) + "_clean"
        min1 = int(min_freq*len(temp_df[cleaned_text]))
        wordsets = [frozenset(document.split(" "))
                    for document in temp_df[cleaned_text]]

        vec = CountVectorizer(ngram_range=(1, 1)).fit(temp_df[cleaned_text])
        bag_of_words = vec.transform(temp_df[cleaned_text])
        sum_words = bag_of_words.sum(axis=0)

        pos_words_freq = []
        neg_words_freq = []
        for word, idx in vec.vocabulary_.items():
            wrd_doc_cnt = sum(1 for s in wordsets if word in s)
            if wrd_doc_cnt >= min1:
                if word in lexi_dict['Positive']:
                    pos_words_freq.append((word, sum_words[0, idx]))
                if word in lexi_dict['Negative']:
                    neg_words_freq.append((word, -1*sum_words[0, idx]))

        pos_words_freq = sorted(
            pos_words_freq, key=lambda x: x[1], reverse=True)
        neg_words_freq = sorted(
            neg_words_freq, key=lambda x: x[1], reverse=True)
        pos_tdf = pd.DataFrame(pos_words_freq[0:min(
            len(pos_words_freq), 10)], columns=['Words', 'Weight'])
        neg_tdf = pd.DataFrame(neg_words_freq[max(
            len(neg_words_freq)-10, 0):len(neg_words_freq)], columns=['Words', 'Weight'])
        tdf = pos_tdf.append(neg_tdf, ignore_index=True)
        tdf = tdf.sort_values(by='Weight', ascending=True)
        return tdf

    def __get_word_sentiment(self, temp_df, min_freq=0.05, X_variable=None, lexi_dict=None):
        tdf_list = []
        cat_list = []
        if X_variable is None:
            tdf = self.__generate_word_sentiment(
                temp_df=temp_df, min_freq=min_freq, lexi_dict=lexi_dict)
            tdf_list.append(tdf)
        else:
            X_variable_uniq = temp_df[X_variable].unique().tolist()
            for cat in X_variable_uniq:
                temp_df_t = temp_df.loc[temp_df[X_variable] == cat]
                tdf = self.__generate_word_sentiment(
                    temp_df=temp_df_t, min_freq=min_freq, lexi_dict=lexi_dict)
                tdf_list.append(tdf)
                cat_list.append(cat)
        return (tdf_list, cat_list)

    def plot_sentiment(self, X_variable=None, return_df=False):
        temp_df = self.__generate_sentiment()
        if (X_variable not in temp_df.columns and X_variable is not None):
            raise ValueError("Provide proper variable name as X-Category.")
        if self.sent_method == 'lexical':
            if self.sent_lexicon == 'loughran':
                self.__plot_loughran(temp_df, X_variable=X_variable)
        if return_df:
            return temp_df

    def plot_word_sentiment(self, new_df=None, X_variable=None):
        if new_df is None:
            temp_df = self.__generate_sentiment()
        else:
            cleaned_text = str(self.text_column) + "_clean"
            if cleaned_text in new_df.columns:
                temp_df = new_df
            else:
                raise ValueError("Provide a DataFrame with " +
                                 cleaned_text + " column in it.")
        tdf_list, cat_list = self.__get_word_sentiment(temp_df=temp_df, min_freq=0.05, X_variable=X_variable,
                                                       lexi_dict=self.lexi_dict)
        for i in range(0, len(tdf_list)):
            item = tdf_list[i]
            clrs = ["darkred" if (
                x < 0) else "darkgreen" for x in item['Weight']]
            fig1 = go.Figure(go.Bar(
                x=item['Weight'], y=item['Words'], orientation='h', marker=dict(color=clrs)))
            if X_variable is None:
                fig1.update_layout(template="plotly_white", title_text=plot_title(
                    "Overall Word-Sentiment"))
            else:
                fig1.update_layout(template="plotly_white",
                                   title_text=plot_title(title="Word-Sentiment", subtitle=str(X_variable) + " = " + str(cat_list[i])))
            fig1.show()




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
                    raise ValueError(
                        "Summary-Ratio should be between (0, 1) non-inclusive rage")
            else:
                raise ValueError(
                    "Please choose method as either of summary / keyword / all")
            if method in ['keyword', 'all']:
                self.method = method
                if keyword_ratio is None:
                    keyword_ratio = 0.4
                if (keyword_ratio > 0 & keyword_ratio < 1.0):
                    self.keyword_ratio = keyword_ratio
                else:
                    raise ValueError(
                        "KeyWord-Ratio should be between (0, 1) non-inclusive rage")
            else:
                raise ValueError(
                    "Please choose method as either of summary / keyword / all")
        else:
            raise TypeError("Only an object of Documents Class can be passed.")

    def __get_summary(self, x):
        summary_text = str(self.text_column) + "_summary"
        x[summary_text] = summarize(
            x[self.text_column], ratio=self.summary_ratio)
        return x

    def __get_keyword(self, x):
        keyword_text = str(self.text_column) + "_keyword"
        x[keyword_text] = keywords(
            x[self.text_column], ratio=self.keyword_ratio)
        return x

    def generate_results(self):
        temp_df = self.processed_df
        if self.method == 'summary':
            temp_df = temp_df.apply(lambda x: self.__get_summary(x), axis=1)
        elif self.method == 'keyword':
            temp_df = temp_df.apply(lambda x: self.__get_keyword(x), axis=1)
        else:
            temp_df = temp_df.apply(lambda x: self.__get_summary(x), axis=1)
            temp_df = temp_df.apply(lambda x: self.__get_keyword(x), axis=1)
        self.processed_df = temp_df
        return temp_df

















