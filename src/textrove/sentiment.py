
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from .ploty_template import plot_title
import pandas as pd
from .eda import Documents
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
# nltk.download('vader_lexicon')
# import importlib.resources as pkg_resources
from . import utils
import os
from tqdm.autonotebook import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings("ignore")

# import swifter

global UTIL_PATH
# with pkg_resources.path('utils', '.') as p:
#     UTIL_PATH = str(p)

UTIL_PATH = os.path.abspath(os.path.dirname(utils.__file__))

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
                        raise ValueError("Please choose lexicon as either of loughran / nrc")
            else:
                raise ValueError("Please choose method as either of lexical / textblob / vader")
        else:
            raise TypeError("Only an object of Documents Class can be passed.")

    def __load_loughran(self):
        #Load master dictionary
        master = pd.read_csv(UTIL_PATH + "/lexicons/loughran/" + "LoughranMcDonald_MasterDictionary_2018.csv")

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
        con_words = master[master["Constraining"] > 0].Word.str.lower().tolist()
        sup_words = master[master["Superfluous"] > 0].Word.str.lower().tolist()
        comb_list = [pos_words, neg_words, unc_words, lit_words, con_words, sup_words, negate]
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
            x['pos_shr'] = 100*pos_cnt / (pos_cnt + neg_cnt + unc_cnt + lit_cnt + con_cnt + sup_cnt)
            x['neg_shr'] = 100*neg_cnt / (pos_cnt + neg_cnt + unc_cnt + lit_cnt + con_cnt + sup_cnt)
            x['unc_shr'] = 100*unc_cnt / (pos_cnt + neg_cnt + unc_cnt + lit_cnt + con_cnt + sup_cnt)
            x['lit_shr'] = 100*lit_cnt / (pos_cnt + neg_cnt + unc_cnt + lit_cnt + con_cnt + sup_cnt)
            x['con_shr'] = 100*con_cnt / (pos_cnt + neg_cnt + unc_cnt + lit_cnt + con_cnt + sup_cnt)
            x['sup_shr'] = 100*sup_cnt / (pos_cnt + neg_cnt + unc_cnt + lit_cnt + con_cnt + sup_cnt)
        else:
            x['pos_shr'] = x['neg_shr'] = x['unc_shr'] = x['lit_shr'] = x['con_shr'] = x['sup_shr'] = 0.0
        if (pos_cnt + neg_cnt) > 0:
            x['sent_scr'] = (pos_cnt - neg_cnt) / (pos_cnt + neg_cnt)
        else:
            x['sent_scr'] = 0.0
        return x

    def __sent_vader(self, x):
        cleaned_text = str(self.text_column) + "_clean"
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(x[cleaned_text])
        x['sent_scr'] = vs['compound']
        x['pos_shr'] = vs['pos']
        x['neu_shr'] = vs['neu']
        x['neg_shr'] = vs['neg']
        return x

    def __sent_textblob(self, x):
        cleaned_text = str(self.text_column) + "_clean"
        vs = TextBlob(x[cleaned_text])
        x['sent_scr'] = vs.sentiment.polarity
        x['subj_shr'] = vs.sentiment.subjectivity
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
            labels = ['Positive', 'Negative', 'Uncertainty', 'Litigious', 'Constraining', 'Superfluous']
            values = [avg_pos, avg_neg, avg_unc, avg_lit, avg_con, avg_sup]
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, hoverinfo="label+percent+name")])
            fig.update_layout(template="plotly_white", title_text=plot_title("Overall Sentiment Score: " + str(avg_sent)))
            fig.show()
        else:
            tdf = temp_df[[X_variable, 'sent_scr', 'pos_shr', 'neg_shr', 'unc_shr',
                           'lit_shr', 'con_shr', 'sup_shr']].groupby([X_variable]).mean()
            tdf = tdf.sort_index().reset_index()
            fig = make_subplots(rows=3, cols=1, specs=[[{}], [{"rowspan": 2}], [None]], 
                                shared_xaxes=True, vertical_spacing=0.00)
            fig.add_trace(go.Scatter(x=tdf[X_variable], y=tdf['sent_scr'], mode='lines+markers',
                                    line_shape='spline', name='Sentiment Score'), row=1, col=1)
            fig.add_trace(go.Bar(x=tdf[X_variable], y=tdf['pos_shr'].round(1), name='Positive'), row=2, col=1)
            fig.add_trace(go.Bar(x=tdf[X_variable], y=tdf['neg_shr'].round(1), name='Negative'), row=2, col=1)
            fig.add_trace(go.Bar(x=tdf[X_variable], y=tdf['unc_shr'].round(1), name='Uncertainty'), row=2, col=1)
            fig.add_trace(go.Bar(x=tdf[X_variable], y=tdf['lit_shr'].round(1), name='Litigious'), row=2, col=1)
            fig.add_trace(go.Bar(x=tdf[X_variable], y=tdf['con_shr'].round(1), name='Constraining'), row=2, col=1)
            fig.add_trace(go.Bar(x=tdf[X_variable], y=tdf['sup_shr'].round(1), name='Superfluous'), row=2, col=1)
            fig.update_layout(barmode='stack', yaxis_visible=True, yaxis_showticklabels=True, xaxis_showticklabels=False,
                              yaxis2_visible=False, yaxis2_showticklabels=False, yaxis_zeroline=True,
                              xaxis2_showticklabels=True, xaxis2_type='category', xaxis_showgrid=False,
                              template="plotly_white", title_text=plot_title("Sentiment Analysis"))
            fig.show()
    
    def __plot_vader(self, temp_df, X_variable=None):
        if X_variable is None:
            avg_sent = round(temp_df['sent_scr'].mean(), 2)
            avg_pos = round(temp_df['pos_shr'].mean(), 1)
            avg_neu = round(temp_df['neu_shr'].mean(), 1)
            avg_neg = round(temp_df['neg_shr'].mean(), 1)
            labels = ['Positive', 'Neutral', 'Negative']
            values = [avg_pos, avg_neu, avg_neg]
            fig = go.Figure(data=[go.Pie(
                labels=labels, values=values, hole=.4, hoverinfo="label+percent+name")])
            fig.update_layout(template="plotly_white", title_text=plot_title(
                "Overall Sentiment Score: " + str(avg_sent)))
            fig.show()
        else:
            tdf = temp_df[[X_variable, 'sent_scr', 'pos_shr', 'neu_shr', 'neg_shr']].groupby([X_variable]).mean()
            tdf = tdf.sort_index().reset_index()
            fig = make_subplots(rows=3, cols=1, specs=[[{}], [{"rowspan": 2}], [None]],
                                shared_xaxes=True, vertical_spacing=0.00)
            fig.add_trace(go.Scatter(x=tdf[X_variable], y=tdf['sent_scr'], mode='lines+markers',
                                     line_shape='spline', name='Sentiment Score'), row=1, col=1)
            fig.add_trace(go.Bar(x=tdf[X_variable], y=tdf['pos_shr'].round(
                1), name='Positive'), row=2, col=1)
            fig.add_trace(go.Bar(x=tdf[X_variable], y=tdf['neu_shr'].round(
                1), name='Neutral'), row=2, col=1)
            fig.add_trace(go.Bar(x=tdf[X_variable], y=tdf['neg_shr'].round(
                1), name='Negative'), row=2, col=1)
            fig.update_layout(barmode='stack', yaxis_visible=True, yaxis_showticklabels=True, xaxis_showticklabels=False,
                              yaxis2_visible=False, yaxis2_showticklabels=False, yaxis_zeroline=True,
                              xaxis2_showticklabels=True, xaxis2_type='category', xaxis_showgrid=False,
                              template="plotly_white", title_text=plot_title("Sentiment Analysis"))
            fig.show()

    def __plot_textblob(self, temp_df, X_variable):
            tdf = temp_df[[X_variable, 'sent_scr']].groupby([X_variable]).mean()
            tdf = tdf.sort_index().reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tdf[X_variable], y=tdf['sent_scr'], mode='lines+markers',
                                     line_shape='spline', name='Sentiment Score'))
            fig.update_layout(yaxis_visible=True, yaxis_showticklabels=True, xaxis_showticklabels=False, yaxis_zeroline=True,
                              xaxis_showgrid=False, template="plotly_white", title_text=plot_title("Sentiment Analysis"))
            fig.show()

    def generate_sentiment(self):
        temp_df = self.processed_df
        if self.sent_method == 'lexical':
            if self.sent_lexicon == 'loughran':
                loughran_list = self.__load_loughran()
                self.lexi_dict['Positive'] = loughran_list[0]
                self.lexi_dict['Negative'] = loughran_list[1]
                temp_df = temp_df.progress_apply(lambda x: self.__sent_loughran(x, loughran_list=loughran_list), axis=1)
        elif self.sent_method == 'vader':
            temp_df = temp_df.progress_apply(lambda x: self.__sent_vader(x), axis=1)
        elif self.sent_method == 'textblob':
            temp_df = temp_df.progress_apply(lambda x: self.__sent_textblob(x), axis=1)
        self.processed_df = temp_df
        return temp_df

    def __generate_word_sentiment(self, temp_df, min_freq=0.05, lexi_dict=None):
        cleaned_text = str(self.text_column) + "_clean"
        min1 = int(min_freq*len(temp_df[cleaned_text]))
        wordsets = [frozenset(document.split(" ")) for document in temp_df[cleaned_text]]

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

        pos_words_freq = sorted(pos_words_freq, key=lambda x: x[1], reverse=True)
        neg_words_freq = sorted(neg_words_freq, key=lambda x: x[1], reverse=True)
        pos_tdf = pd.DataFrame(pos_words_freq[0:min(len(pos_words_freq), 10)], columns=['Words', 'Weight'])
        neg_tdf = pd.DataFrame(neg_words_freq[max(len(neg_words_freq)-10, 0):len(neg_words_freq)], columns=['Words', 'Weight'])
        tdf = pos_tdf.append(neg_tdf, ignore_index=True)
        tdf = tdf.sort_values(by='Weight', ascending=True)
        return tdf

    def __get_word_sentiment(self, temp_df, min_freq=0.05, X_variable=None, lexi_dict=None):
        tdf_list = []
        cat_list = []
        if X_variable is None:
            tdf = self.__generate_word_sentiment(temp_df=temp_df, min_freq=min_freq, lexi_dict=lexi_dict)
            tdf_list.append(tdf)
        else:
            X_variable_uniq = temp_df[X_variable].unique().tolist()
            for cat in X_variable_uniq:
                temp_df_t = temp_df.loc[temp_df[X_variable] == cat]
                tdf = self.__generate_word_sentiment(temp_df=temp_df_t, min_freq=min_freq, lexi_dict=lexi_dict)
                tdf_list.append(tdf)
                cat_list.append(cat)
        return (tdf_list, cat_list)

    def plot_sentiment(self, X_variable=None, return_df=False):
        temp_df = self.generate_sentiment()
        if (X_variable not in temp_df.columns and X_variable is not None):
            raise ValueError("Provide proper variable name as X-Category.")
        if self.sent_method == 'lexical':
            if self.sent_lexicon == 'loughran':
                self.__plot_loughran(temp_df, X_variable=X_variable)
        elif self.sent_method == 'vader':
            self.__plot_vader(temp_df, X_variable=X_variable)
        elif self.sent_method == 'textblob':
            if X_variable is None:
                raise ValueError("Need a X-Variable to plot with TextBlob")
            else:
                self.__plot_textblob(temp_df, X_variable=X_variable)
        if return_df:
            return temp_df

    def plot_word_sentiment(self, new_df=None, X_variable=None):
        if self.sent_method != 'lexical':
            raise ValueError("Word Sentiment can be plotted only with Lexical-based Sentiments")
        if new_df is None:
            temp_df = self.generate_sentiment()
        else:
            cleaned_text = str(self.text_column) + "_clean"
            if cleaned_text in new_df.columns:
                temp_df = new_df
            else:
                raise ValueError("Provide a DataFrame with " + cleaned_text + " column in it.")
        tdf_list, cat_list = self.__get_word_sentiment(temp_df=temp_df, min_freq=0.05, X_variable=X_variable, 
                                                        lexi_dict=self.lexi_dict)
        for i in range(0, len(tdf_list)):
            item = tdf_list[i]
            clrs = ["darkred" if (x < 0) else "darkgreen" for x in item['Weight']]
            fig1 = go.Figure(go.Bar(x=item['Weight'], y=item['Words'], orientation='h', marker=dict(color=clrs)))
            if X_variable is None:
                fig1.update_layout(yaxis_showticklabels=True, yaxis_type='category',
                                   template="plotly_white", title_text=plot_title("Overall Word-Sentiment"))
            else:
                fig1.update_layout(template="plotly_white", yaxis_showticklabels=True, yaxis_type='category',
                title_text=plot_title(title = "Word-Sentiment", subtitle = str(X_variable) + " = " + str(cat_list[i])))
            fig1.show()
            














