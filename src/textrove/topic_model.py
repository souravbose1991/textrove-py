import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pyLDAvis.gensim_models  # don't skip this
import pyLDAvis
from nltk import word_tokenize
import pandas as pd
import numpy as np  
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import Phrases
from gensim.models import CoherenceModel
# from gensim.models import ldaseqmodel
from gensim.corpora import Dictionary, bleicorpus
from datetime import datetime
from .ploty_template import plot_title
from .eda import Documents
from . import models

import warnings
warnings.filterwarnings("ignore")

global MODEL_PATH
MODEL_PATH = os.path.abspath(os.path.dirname(models.__file__))


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

            if (num_topics is None or num_topics <=1):
                self.num_topics = 1
            elif num_topics > 1:
                self.num_topics = num_topics
            else:
                raise ValueError("Please enter num_topics > 1")
        else:
            raise TypeError("Only an object of Documents Class can be passed.")
            
        self.ldamodel = None
        self.model_path = None
        self.dictionary = None
        self.corpus = None  
        self.texts = None

    
    ########### Prepare texts for Topic-Model ##############
    def __prep_texts(self):
        print("--- Preparing Texts for Model ---")
        cleaned_text = str(self.text_column) + "_clean"
        doc_lst = self.processed_df[cleaned_text].tolist()
        doc_lst = [word_tokenize(str(doc)) for doc in doc_lst]

        # Compute bigrams.
        # Add bigrams to docs (as per the linked NPMI paper).
        bigram = Phrases(doc_lst, threshold=10e-5, scoring='npmi')
        for idx in range(len(doc_lst)):
            temp_bigram = []
            for token in bigram[doc_lst[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    temp_bigram.append(token)
            doc_lst.append(temp_bigram)

        # Create Corpus
        dictionary = Dictionary(doc_lst)
        dictionary.filter_extremes(no_above=0.9)
        corpus = [dictionary.doc2bow(text) for text in doc_lst]

        self.texts = doc_lst
        self.dictionary = dictionary
        self.corpus = corpus


    ################## Optimal Topic Counts ##################
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


    def __plot_chooseK(self, num_topics, mean_stabilities, coherence_values, perplexity_values, optim_k):
        miny1 = min(perplexity_values[:-1])*0.8
        maxy1 = max(perplexity_values[:-1])*1.2
        fig = make_subplots(rows=3, cols=1, specs=[[{}], [{"rowspan": 2}], [None]], 
                            shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.00)
        fig.add_trace(go.Scatter(x=num_topics[:-1], y=perplexity_values[:-1], mode='lines+markers',
                                line_shape='spline', name='Perplexity Score'), row=1, col=1)
        fig.add_trace(go.Scatter(x=num_topics[:-1], y=coherence_values[:-1], mode='lines+markers',
                                line_shape='spline', name='Coherence Score'), row=2, col=1)
        fig.add_trace(go.Scatter(x=num_topics[:-1], y=mean_stabilities, mode='lines+markers',
                                 line_shape='spline', name='Stability Score'), row=2, col=1)
        fig.add_shape(type='line', x0=optim_k, y0=0, x1=optim_k,
                      line=dict(color="MediumPurple", width=3, dash='dashdot'),  row=1, col=1)
        fig.add_shape(type='line', x0=optim_k, y0=0, x1=optim_k,
                      line=dict(color="MediumPurple", width=3, dash='dashdot'),  row=2, col=1)
        fig.update_layout(yaxis_visible=False, yaxis_showticklabels=False, xaxis2_title='Number of Topics', 
                          yaxis2_visible=False, yaxis2_showticklabels=False, template="plotly_white", 
                          yaxis_range=[miny1, maxy1], yaxis2_range=[0, 1],
                          title_text=plot_title("Optimal Topic Analysis"))
        fig.show()

    
    def __chooseK(self, limit=15, start=2, step=1):
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
        print("--- Checking for best K between [" + str(start) + ", " + str(limit) +"] --- \n")
        for i in num_topics:
            if i <= limit:
                print("--- Simulating Model with K=" + str(i) + " ---")
            model_list[i] = LdaMulticore(corpus=self.corpus, id2word=self.dictionary,
                            num_topics=i, passes=20, alpha='asymmetric', eta='auto', random_state=42, iterations=500,
                            per_word_topics=True, eval_every=None)

            shown_topics = model_list[i].show_topics(num_topics=i, num_words=20, formatted=False)
            LDA_topics[i] = [[word[0] for word in topic[1]] for topic in shown_topics]
            perplexity_values.append(model_list[i].log_perplexity(self.corpus)*-1)
            coherence_values.append(CoherenceModel(model=model_list[i], texts=self.texts, dictionary=self.dictionary,
                                    coherence='c_v').get_coherence())
        
        print("--- Calculating Stability Index ---")
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

        print("--- Identifying Optimal K ---")
        coh_sta_diffs = [coherence_values[i] - mean_stabilities[i] for i in range(0, len(num_topics)-1)]
        coh_sta_max = max(coh_sta_diffs)
        coh_sta_max_idxs = [i for i, j in enumerate(coh_sta_diffs) if j == coh_sta_max]
        ideal_topic_num_index = coh_sta_max_idxs[0] # choose less topics in case there's more than one max
        optim_k = num_topics[ideal_topic_num_index]

        #### Plot for Optimal K ####
        self.__plot_chooseK(num_topics, mean_stabilities, coherence_values, perplexity_values, optim_k)

        return optim_k


    def __train_topicmodel(self, num_topics=1, save_model=False, dir_name=None, file_name=None):
        if num_topics <= 1:
            if self.num_topics == 1:
                self.num_topics = self.__chooseK(limit=20, start=2, step=1)
        else:
            if self.num_topics == 1:
                self.num_topics = num_topics
        # Build LDA model
        ldamodel = LdaMulticore(corpus=self.corpus, id2word=self.dictionary, num_topics=self.num_topics, 
                                passes=20, alpha='asymmetric', eta='auto', random_state=42, iterations=500, 
                                per_word_topics=True, eval_every=None)
        self.ldamodel = ldamodel
        if save_model:
            self.__save_topicmodel(dir_name=dir_name, file_name=file_name)
        
    
    def __save_topicmodel(self, dir_name=None, file_name=None):
        if dir_name is None:
            dir_name = MODEL_PATH
        if file_name is None:
            file_name = "LDA_Model"
        print("--- Saving Model ---")
        self.model_path = str(dir_name + file_name.strip() + " " + str(datetime.now()))
        self.ldamodel.save(self.model_path + "/model_obj")
        self.dictionary.save(self.model_path + "/dictionary_obj")


    def load_topicmodel(self, model_path=None):
        print("--- Loading Model ---")
        self.model_path = model_path
        self.ldamodel = LdaMulticore.load(model_path + "/model_obj", mmap='r')
        self.dictionary = Dictionary.load(model_path + "/dictionary_obj", mmap='r')
        self.num_topics = self.ldamodel.num_topics


    def __eval_topicmodel(self, return_df=True, evaluation='complete'):
        if evaluation == 'complete':
            print("--- Evaluating Model Metrics ---")
            coherencemodel = CoherenceModel(model=self.ldamodel, texts=self.texts, dictionary=self.dictionary, coherence='c_v')
            top_topics = self.ldamodel.top_topics(corpus=self.corpus, texts=self.texts, dictionary=self.dictionary, 
                                                    window_size=None, coherence='c_v', topn=20)
            # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
            avg_topic_coherence = sum([t[1] for t in top_topics]) / len(top_topics)
            print("\n")
            print('Average topic coherence: %.4f.' % avg_topic_coherence)
            print('Model coherence: %.4f.' % coherencemodel.get_coherence())
            print('Perplexity: %.4f.' % self.ldamodel.log_perplexity(self.corpus)*-1)
            print("\n")
        topics = self.ldamodel.show_topics(num_topics=-1, formatted=False, num_words=30)
        topic_dict = {}
        for topicid, word_weight in topics:
            wrd_lst = [word[0] for word in word_weight]
            topic_dict['Topic-'+str(topicid+1)] = wrd_lst
            print('Topic-'+str(topicid+1)+": ", wrd_lst)
            print("\n")
        topic_df = pd.DataFrame(topic_dict)
        self.topic_map = topic_df
        if return_df:
            return topic_df


    def __visualize(self, save_vis=False):
        # Visualize the topics
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim_models.prepare(self.ldamodel, self.corpus, self.dictionary, sort_topics=False)
        if save_vis:
            pyLDAvis.save_html(vis, self.model_path + "/model_vis.html")
        return vis


    ################## Topic Modelling Formatted output ##################
    def __eval_text(self, x, cleaned_text=None):
        print("--- Preparing texts for predictions ---")
        if cleaned_text is None:
            cleaned_text = str(self.text_column) + "_clean"
        doc_lst = list(x[cleaned_text])
        doc_lst = [word_tokenize(str(doc)) for doc in doc_lst]

        # Compute bigrams.
        # Add bigrams to docs (as per the linked NPMI paper).
        bigram = Phrases(doc_lst, threshold=10e-5, scoring='npmi')
        for idx in range(len(doc_lst)):
            temp_bigram = []
            for token in bigram[doc_lst[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    temp_bigram.append(token)
            doc_lst.append(temp_bigram)
        
        corpus = [self.dictionary.doc2bow(text) for text in doc_lst]
        print("--- Predicting on the texts ---")
        all_topics = self.ldamodel.get_document_topics(corpus[0])
        for element in all_topics:
            x['Topic-'+str(element[0]+1)] = element[1]
        return x
    
    def suggest_num_topic(self, limit=15, start=2, step=1):
        self.__prep_texts()
        optim_k = self.__chooseK(limit=limit, start=start, step=step)
        return optim_k


    def fit(self, num_topics=1, save_model=False, dir_name=None, file_name=None):
        self.__prep_texts()
        self.__train_topicmodel(num_topics=num_topics, save_model=save_model, dir_name=dir_name, file_name=file_name)
        print("Model training complete.")
        

    def evaluate(self, save_vis=False):
        if self.ldamodel is not None:
            topic_df = self.__eval_topicmodel(return_df=True)
            vis = self.__visualize(save_vis=save_vis)
            return (topic_df, vis)
        else:
            raise Exception("Train/Load a LDA model first")


    def predict(self, data=None, text_column=None, return_df=True):
        if self.ldamodel is not None:
            if data is None:
                data = self.processed_df
            data = data.apply(lambda x: self.__eval_text(x, cleaned_text=text_column), axis=1)
            self.processed_df = data
            if return_df:
                return data
        else:
            raise Exception("Train/Load a LDA model first")


    def plot_topics(self, data=None, text_column=None, X_variable=None, return_df=False):
        if self.ldamodel is None:
            raise Exception("Train/Load a LDA model first")
        temp_df = self.predict(data=data, text_column=text_column, return_df=True)
        
        if (X_variable not in temp_df.columns and X_variable is not None):
            raise ValueError("Provide proper variable name as X-Category.")

        self.__eval_topicmodel(return_df=False, evaluation='partial')
        print("\n")

        topic_shr = {}
        if X_variable is None:
            for i in range(0, self.ldamodel.num_topics):
                topic_shr['Topic-'+str(i+1)] = round(100.0*temp_df['Topic-'+str(i+1)].mean(), 1)
            labels = list(topic_shr.keys())
            values = list(topic_shr.values())
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, hoverinfo="label+percent+name")])
            fig.update_layout(template="plotly_white", title_text=plot_title("Static Topic Analysis"))
            fig.show()
        else:
            tops = ['Topic-'+str(i+1) for i in range(0, self.ldamodel.num_topics)]
            tdf = temp_df[[X_variable]+tops].groupby([X_variable]).mean()
            tdf = tdf.sort_index().reset_index()
            fig = go.Figure()
            for item in tops:
                fig.add_trace(go.Bar(x=tdf[X_variable], y=100.0*tdf[item].round(1), name=item))
            fig.update_layout(barmode='stack', yaxis_visible=False, yaxis_showticklabels=False, template="plotly_white",
                              title_text=plot_title("Dynamic Topic Analysis"))
            fig.show()

        if return_df:
            return temp_df
        


















