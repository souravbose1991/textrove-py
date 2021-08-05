
import numpy as np
import pandas as pd
# import swifter
from .eda import Documents
# import contractions
# import importlib.resources as pkg_resources
from sklearn.feature_extraction.text import CountVectorizer
from . import www
import os
from pyvis.network import Network
import networkx as nx
from tqdm.autonotebook import tqdm  
tqdm.pandas()

import warnings
warnings.filterwarnings("ignore")

global WWW_PATH
# with pkg_resources.path('utils', '.') as p:
#     UTIL_PATH = str(p)

WWW_PATH = os.path.abspath(os.path.dirname(www.__file__))


class WordNetwork:
    def __init__(self, documents_object):
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

    def __get_corr_mat(self, docs):
        vec = CountVectorizer()
        X = vec.fit_transform(docs)
        dtm = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
        dtm = dtm.clip(0, 1)
        corr_mat = dtm.corr()
        np.fill_diagonal(corr_mat.values, 0.0)
        corr_mat = corr_mat.progress_apply(abs)
        return corr_mat

    def __corr_to_netx(self, corr_mat, thresh):
        corr_mat = corr_mat.reset_index()
        net_mat = corr_mat.melt(id_vars=['index'])
        net_mat = net_mat[~net_mat[['index', 'variable']].progress_apply(frozenset, 1).duplicated()]
        net_mat = net_mat.rename(columns={"index": "Source", "variable": "Target", "value": "Weight"})
        net_mat = net_mat.loc[net_mat['Weight'] >= thresh].reset_index(drop=True)
        return net_mat

    def create_network(self, new_df=None, threshold=0.6):
        cleaned_text = str(self.text_column) + "_clean"
        if new_df is None:
            temp_df = self.processed_df
        else:
            if cleaned_text in new_df.columns:
                temp_df = new_df
            else:
                raise ValueError("Provide a DataFrame with " + cleaned_text + " column in it.")
        docs = temp_df[cleaned_text].tolist()
        corr_mat = self.__get_corr_mat(docs)
        net_mat = self.__corr_to_netx(corr_mat, threshold)
        nx_graph = nx.from_pandas_edgelist(net_mat, source='Source', target='Target', edge_attr=['Weight'])
        word_netwrk = Network(height='600px', width='1100px', notebook=True)
        word_netwrk.from_nx(nx_graph)
        word_netwrk.toggle_physics(True)
        word_netwrk.toggle_stabilization(True)
        word_netwrk.show_buttons(filter_=['physics'])
        word_netwrk.show(WWW_PATH + '/word_network.html')
