# Gensim
import unicodedata
import contractions
import string
from nltk.tokenize import RegexpTokenizer
import re
import pickle
import os
import matplotlib.pyplot as plt
import pyLDAvis.gensim  # don't skip this
import pyLDAvis
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import TweetTokenizer
import nltk
from itertools import chain
from bs4 import BeautifulSoup
import swifter
import pandas as pd
import numpy as np
import gensim
import gensim.corpora as corpora
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models import Phrases
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import ldaseqmodel
from gensim.corpora import Dictionary, bleicorpus
from gensim.matutils import hellinger

from wordcloud import STOPWORDS
from nltk.corpus import stopwords

global UTIL_PATH, STOP_WORD
UTIL_PATH = str(Path("../utils").resolve())

################## Stopwords list ##################
stop1 = [re.sub(r"(\|(.)+)|(\n)", "", x.lower())
         for x in open(UTIL_PATH+"/stopwords/"+"StopWords_Generic.txt", "r")]
stop2 = [re.sub(r"(\|(.)+)|(\n)", "", x.lower())
         for x in open(UTIL_PATH+"/stopwords/"+"StopWords_GenericLong.txt", "r")]
stop3 = [re.sub(r"(\|(.)+)|(\n)", "", x.lower())
         for x in open(UTIL_PATH+"/stopwords/"+"StopWords_DatesandNumbers.txt", "r")]

STOP_WORD = list(set(list(stopwords.words("english")) + list(STOPWORDS) + stop1 + stop2 + stop3))



#Spacy
# import spacy
# spacy.load('en')
# from spacy.lang.en import English
# parser = English()

# nltk.data.path.append("/app/localstorage/u_am_coe/notebooks/nltk_data")
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


# Gensim

# Plots
# %matplotlib inline


    
################## Dynamic Topic Modelling ##################

class DynTM:
    def __init__(self, documents_object=None, num_topics=None):
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

            if num_topics is None:
                self.num_topics = 1
                self.method = 'auto'
            elif num_topics > 1:
                self.num_topics = num_topics
                self.method = 'mannual'
            else:
                raise ValueError("Please enter num_topics > 1")
        else:
            pass
            


        self.stop_words = stop_words
        self.struct_df = struct_df
        self.subset_df = subset_df
        self.uniquetimes = uniquetimes
        self.time_slices = time_slices
        self.time_df = time_df
#         self.vis_obj = vis_obj
        self.num_topics = num_topics
        self.company = company
        self.ldamodel = ldamodel
        self.dictionary = dictionary
        self.corpus = corpus
        self.texts = texts

    

    ########### Prepare texts for Topic-Model ##############

    def prepare_text(self,):
        pass


    ################## Optimal Topic Number ##################

    def __jaccard_similarity(self, topic_1, topic_2):
        """
        Derives the Jaccard similarity of two topics

        Jaccard similarity:
        - A statistic used for comparing the similarity and diversity of sample sets
        - J(A,B) = (A ∩ B)/(A ∪ B)
        - Goal is low Jaccard scores for coverage of the diverse elements
        """
        intersection = set(topic_1).intersection(set(topic_2))
        union = set(topic_1).union(set(topic_2))

        return float(len(intersection))/float(len(union))

    
    def __chooseK(self, limit=20, start=2, step=1):
        """
        Compute c_v coherence & Model Stability for various number of topics and find optimal one
        Parameters:
        ----------
        limit : Max num of topics
        start : Least num of topics
        step  : Step-size
        Returns:
        -------
        optim_model : List of LDA topic models
        optim_k : optimal number of Topics
        """
        coherence_values = []
        perplexity_values = []
        num_topics = list(range(start, limit+step+step, step))
        model_list = {}
        LDA_topics = {}
        for i in num_topics:
            model_list[i] = gensim.models.ldamulticore.LdaMulticore(corpus=self.corpus, id2word=self.dictionary,
                            num_topics=i, passes=20, alpha='asymmetric', eta='auto', random_state=42, iterations=500,
                            per_word_topics=True, eval_every=None)

            shown_topics = model_list[i].show_topics(num_topics=i, num_words=20, formatted=False)
            LDA_topics[i] = [[word[0] for word in topic[1]] for topic in shown_topics]
            perplexity_values.append(model_list[i].log_perplexity(self.corpus))
            coherence_values.append(CoherenceModel(model=model_list[i], texts=self.texts, dictionary=self.dictionary,
                                    coherence='c_v').get_coherence())
        
        LDA_stability = {}
        for i in range(0, len(num_topics)-1):
            jaccard_sims = []
            for t1, topic1 in enumerate(LDA_topics[num_topics[i]]):  # pylint: disable=unused-variable
                sims = []
                for t2, topic2 in enumerate(LDA_topics[num_topics[i+1]]):  # pylint: disable=unused-variable
                    sims.append(self.__jaccard_similarity(topic1, topic2))
                jaccard_sims.append(sims)
            LDA_stability[num_topics[i]] = jaccard_sims

        mean_stabilities = [np.array(LDA_stability[i]).mean() for i in num_topics[:-1]]

        coh_sta_diffs = [coherence_values[i] - mean_stabilities[i] for i in range(0, len(num_topics)-1)]
        coh_sta_max = max(coh_sta_diffs)
        coh_sta_max_idxs = [i for i, j in enumerate(coh_sta_diffs) if j == coh_sta_max]
        ideal_topic_num_index = coh_sta_max_idxs[0] # choose less topics in case there's more than one max
        ideal_topic_num = num_topics[ideal_topic_num_index]

        #### Plot for Optimal K ####

        # plt.figure(figsize=(20,10))
        # ax = sns.lineplot(x=num_topics[:-1], y=mean_stabilities, label='Average Topic Overlap')
        # ax = sns.lineplot(x=num_topics[:-1], y=coherences, label='Topic Coherence')

        # ax.axvline(x=ideal_topic_num, label='Ideal Number of Topics', color='black')
        # ax.axvspan(xmin=ideal_topic_num - 1, xmax=ideal_topic_num + 1, alpha=0.5, facecolor='grey')

        # y_max = max(max(mean_stabilities), max(coherences)) + (0.10 * max(max(mean_stabilities), max(coherences)))
        # ax.set_ylim([0, y_max])
        # ax.set_xlim([1, num_topics[-1]-1])
                        
        # ax.axes.set_title('Model Metrics per Number of Topics', fontsize=25)
        # ax.set_ylabel('Metric Level', fontsize=20)
        # ax.set_xlabel('Number of Topics', fontsize=20)
        # plt.legend(fontsize=20)
        # plt.show() 



    def __train_topicmodel(self):
        pass



    ################## Topic Modelling Formatted output ##################

    def format_topics_sentences(self, ldamodel=None, corpus=None, texts=None):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row_list in enumerate(ldamodel[corpus]):
            row = row_list[0] if ldamodel.per_word_topics else row_list
            # print(row)
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series(
                        [int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic',
                                  'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return sent_topics_df

    






    ################## Defining Topic models for sequences with optimal number of Topics ##################

    def compute_coherence_values(self, limit=11, start=4, step=1):
        """
        Compute c_v coherence for various number of topics
        Parameters:
        ----------
        limit : Max num of topics
        start : Least num of topics
        step  : Step-size
        Returns:
        -------
        model_list : List of LDA topic models (not now)
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics (not now)
        optim_k : optimal number of Topics
        """
        coherence_values = []
        perplexity_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = gensim.models.ldamulticore.LdaMulticore(corpus=self.corpus,
                                                            id2word=self.dictionary,
                                                            num_topics=num_topics,
                                                            passes=10,
                                                            alpha='asymmetric',
                                                            eta='auto',
                                                            random_state=42,
                                                            iterations=500,
                                                            per_word_topics=True,
                                                            eval_every=None)
            model_list.append(model)
            coherencemodel = CoherenceModel(
                model=model, texts=self.texts, dictionary=self.dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
            perplexity_values.append(model.log_perplexity(self.corpus))

        plot_val = [-1 * i / j for i,
                    j in zip(coherence_values, perplexity_values)]
        x = np.array(coherence_values)
        z = (x-min(x))/(max(x)-min(x))
        scaled_coherence = z.tolist()
        scaled_subset = [i for i in scaled_coherence if i >=
                         max(scaled_coherence)*0.7]
        scaled_subset_index = [
            scaled_coherence.index(i) for i in scaled_subset]

        try:
            scaled_subset_index.remove(4)
            if(len(scaled_subset_index) == 0):
                scaled_subset_index = [4]
        except:
            scaled_subset_index = scaled_subset_index
        finally:
            best_model = model_list[min(scaled_subset_index)]

        optim_k = best_model.get_topics().shape[0]

        x = range(start, limit, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()

        x = range(start, limit, step)
        plt.plot(x, perplexity_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Perplexity score")
        plt.legend(("perplexity_values"), loc='best')
        plt.show()

        self.num_topics = optim_k

    ################## Defining Topic models evaluation methods ##################

    def evaluate_model(self, coherence='c_v', seq=False):
        if(seq == True):
            coh_t = []
            for i in range(len(self.uniquetimes)):
                topics_dtm = self.model.dtm_coherence(time=i)
                temp_mod = CoherenceModel(
                    topics=topics_dtm, texts=self.texts, dictionary=self.dictionary, coherence='c_v')
                coh_t.append(temp_mod.get_coherence())

            avg_coherence = sum(coh_t) / len(coh_t)
            print('Average Model coherence: %.4f.' % avg_coherence)

            plt.plot(self.uniquetimes, coh_t)
            plt.xticks(rotation=70)
            plt.xlabel("Time Slices")
            plt.ylabel("Coherence score")
            plt.legend(("coherence_values"), loc='best')
            plt.show()

            fin = []
            cols = ["Topic-" + str(i+1)
                    for i in range(len(self.model.doc_topics(0)))]
            for i in range(len(texts)):
                temp1 = self.model.doc_topics(i)
                fin.append(temp1)

            top_share = pd.DataFrame(fin, columns=cols)
            abc = pd.concat([self.time_df, top_share], axis=1)
            top_share = abc.groupby('Year2').mean(
            ).sort_values('Year2', ascending=True)

            for i in range(len(self.model.doc_topics(0))):
                y = np.array(top_share[cols[i]])
                plt.plot(self.uniquetimes, y, label="Topic-" + str(i+1))

            plt.xticks(rotation=70)
            plt.xlabel("Time Slices")
            plt.ylabel("Coherence score")
            plt.legend()
            plt.show()

        else:
            coherencemodel = CoherenceModel(
                model=self.model, texts=self.texts, dictionary=self.dictionary, coherence='c_v')
            top_topics = self.model.top_topics(
                corpus=self.corpus, texts=self.texts, dictionary=self.dictionary, window_size=None, coherence='c_v', topn=20)
            # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
            avg_topic_coherence = sum(
                [t[1] for t in top_topics]) / len(top_topics)
            print('Average topic coherence: %.4f.' % avg_topic_coherence)
            print('Model coherence: %.4f.', coherencemodel.get_coherence())
            print('Perplexity: ', self.model.log_perplexity(self.corpus))

    ################## Topic Trends for DTM ##################

    def topic_trends(self, explore_topic=1, top_terms=5, normalize=False):
        result = pd.DataFrame([], columns=['keywords'])
        for i in range(len(self.uniquetimes)):
            x = pd.DataFrame(self.model.print_topic(
                explore_topic-1, time=i, top_terms=top_terms), columns=['keywords', 'weight'])
            if(normalize == True):
                x['Nmlz_Prob_' + str(self.uniquetimes[i])
                  ] = x['weight']/x['weight'].sum()
            else:
                x['Nmlz_Prob_' + str(self.uniquetimes[i])] = x['weight']
            x = x.drop(['weight'], axis=1)
            result = pd.merge(result, x, on='keywords', how='outer')

        for j in range(len(result)):
            y = np.array(result.loc[j])[1:]
            plt.plot(self.uniquetimes, y, label=str(result['keywords'][j]))

        plt.xticks(rotation=70)
        plt.xlabel("Time Slices")
        plt.ylabel("Trend Importance")
        plt.legend()
        plt.show()
        return result

    ################## Defining Topic models Training Method ##################

    def dynamic_model(self, n_top=None, timestamp=None, seq=False):
        """
        Function to define both Dynamic / Static topic models.
        Returns:
        lda_sta_<timestamp>.html = The pyLDA vis html to visualize the static LDA model
        lda_seq_snap_<timestamp>.html = The pyLDA vis html to visualize the snap at time for the sequential LDA model
        model_sta.pkl = The pickle object for static LDA model
        model_seq.pkl = The pickle object for static LDA model
        Parameters:
        n_top = (str or int) number of topics (optional) or else the default is 'optimal'
        timestamp = (str or int) To subset the data by value of 'Year2' column for static LDA and also 
                    the time-snapshot for LDA vis html in sequential model (for validation) as per values of Year2
        seq = (bool) To decide if builing a LDA Sequential Model
        """

        abc1 = self.subset_df
        if(seq == True):
            abc1 = abc1.sort_values(by='Year2', ascending=True)
            abc1.reset_index(drop=True, inplace=True)
            self.time_df = pd.DataFrame(abc1.Year2, columns=['Year2'])
            uniquetimes, time_slices = np.unique(
                abc1.Year2, return_counts=True)
            self.uniquetimes = uniquetimes
            self.time_slices = time_slices

            doc_lst = abc1.Cleaned_Text.tolist()
            doc_lst2 = []
            doc_lst = [nltk.word_tokenize(str(doc)) for doc in doc_lst]

            # Compute bigrams.
            # Add bigrams and trigrams to docs (only ones that appear 10 times or more).
            bigram = Phrases(doc_lst, min_count=2, threshold=1.0)
            for idx in range(len(doc_lst)):
                temp_bigram = []
                for token in bigram[doc_lst[idx]]:
                    if '_' in token:
                        # Token is a bigram, add to document.
                        temp_bigram.append(token)
                doc_lst.append(temp_bigram)

            # Create Corpus
            # data_lemmatized = [nltk.word_tokenize(str(sent)) for sent in doc_lst2]
            dictionary = corpora.Dictionary(doc_lst)
            dictionary.filter_extremes(no_below=2, no_above=0.8)
            corpus = [dictionary.doc2bow(text) for text in doc_lst]

            self.texts = doc_lst
            self.dictionary = dictionary
            self.corpus = corpus

            if (n_top == 'optimal'):
                self.compute_coherence_values(limit=11, start=4, step=1)

            # Build LDA Sequence model
            ldamodel = gensim.models.ldaseqmodel.LdaSeqModel(corpus=self.corpus, time_slice=self.time_slices, id2word=self.dictionary,
                                                             alphas=0.01, num_topics=self.num_topics, initialize='gensim', sstats=None,
                                                             lda_model=None, obs_variance=0.5, chain_variance=0.005,
                                                             passes=10, random_state=42, lda_inference_max_iter=25,
                                                             em_min_iter=6, em_max_iter=20, chunksize=100)

            self.ldamodel = ldamodel
            ldamodel.save(str(self.company) + "_model_seq.pkl")
            self.evaluate_model(coherence='c_v', seq=True)
            self.topic_trends(explore_topic=1, top_terms=5, normalize=False)

            # Visualize the topics
            pyLDAvis.enable_notebook()

            if (timestamp is not None):
                snap = uniquetimes.tolist().index(timestamp)
#                 print('Coherence at ' + timestamp + ': ', ldamodel.dtm_coherence(snap))
            else:
                snap = 0

            doc_topic, topic_term, doc_lengths, term_frequency, vocab = self.ldamodel.dtm_vis(
                time=snap, corpus=self.corpus)
            vis = pyLDAvis.prepare(topic_term_dists=topic_term, doc_topic_dists=doc_topic, doc_lengths=doc_lengths, vocab=vocab,
                                   term_frequency=term_frequency, sort_topics=False)
            pyLDAvis.save_html(
                vis, str(self.company) + '_lda_seq_snap_' + str(uniquetimes[snap]) + '.html')

        else:
            doc_lst = abc1[abc1.Year2 == timestamp].Cleaned_Text.tolist()
            doc_lst2 = []
            doc_lst = [nltk.word_tokenize(str(doc)) for doc in doc_lst]

            # Compute bigrams.
            # Add bigrams and trigrams to docs (only ones that appear 10 times or more).
            bigram = Phrases(doc_lst, min_count=2, threshold=1.0)
            for idx in range(len(doc_lst)):
                temp_bigram = []
                for token in bigram[doc_lst[idx]]:
                    if '_' in token:
                        # Token is a bigram, add to document.
                        temp_bigram.append(token)
                doc_lst.append(temp_bigram)

            # Create Corpus
            # data_lemmatized = [nltk.word_tokenize(str(sent)) for sent in doc_lst2]
            dictionary = corpora.Dictionary(doc_lst)
            dictionary.filter_extremes(no_below=2, no_above=0.8)
            corpus = [dictionary.doc2bow(text) for text in doc_lst]

            self.texts = doc_lst
            self.dictionary = dictionary
            self.corpus = corpus

            if (n_top == 'optimal'):
                self.compute_coherence_values(limit=11, start=4, step=1)

            # Build LDA model
            ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus=self.corpus, id2word=self.dictionary, num_topics=self.num_topics, passes=10, alpha='asymmetric', eta='auto',
                                                               random_state=42, iterations=500, per_word_topics=True, eval_every=None)

    #         ldamodel.print_topics()
            self.ldamodel = ldamodel
            ldamodel.save(str(self.company) + "_model_sta_" + \
                          str(timestamp) + ".pkl")
            self.evaluate_model(coherence='c_v', seq=False)

            # Visualize the topics
            pyLDAvis.enable_notebook()
            vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics= False)
            pyLDAvis.save_html(vis, str(self.company) + \
                               '_lda_sta_' + str(timestamp) + '.html')

        return vis




