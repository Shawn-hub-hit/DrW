from distutils.command.config import config
import numpy as np
import os
import torch
import pandas as pd
import csv
from collections import Counter
# from pytorch_transformers import BertTokenizer,BertModel
from collections import Iterable
from torch.utils import data
import math
import pickle
from math import log
from scipy import spatial
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import os
from pathlib import Path
from config import *

class TextCollection(object):
    def __init__(self, source):
        self._texts = source
        self._idf_cache = {}

    def tf(self, term, text):
        return text.count(term) / len(text)

    def idf(self, term):
        # idf values are cached for performance.
        idf = self._idf_cache.get(term)
        if idf is None:
            matches = len([True for text in self._texts if term in text])
            # FIXME Should this raise some kind of error instead?
            idf = (log(len(self._texts) / matches) if matches else 0.0)
            self._idf_cache[term] = idf
        return idf

    def tf_idf(self, term, text):
        return self.tf(term, text) * self.idf(term)

class Interactions(object):
    def __init__(self, parser, log_):

        self.up_data = DATA_DIR / dataname

        # self.train_relation_df = pd.read_csv(self.up_data / 'train.csv', header=0)
        self.test_relation_df = pd.read_csv(self.up_data / 'test.csv', header=0)
        self.left_text_df = pd.read_csv(self.up_data / 'query_text.csv', header=0)
        self.right_text_df = pd.read_csv(self.up_data / 'POI_text.csv', header=0)

        # self.left_text_df['location_left'] = self.left_text_df['location_left'].map(eval)
        # self.right_text_df['location_right'] = self.right_text_df['location_right'].map(eval)
        self.left_text_df['text_left'] = self.left_text_df['text_left'].map(eval)
        self.right_text_df['text_right'] = self.right_text_df['text_right'].map(eval)
        self.left_text_df.set_index('id_left', inplace=True)
        self.right_text_df.set_index('id_right', inplace=True)
        self.change_type(self.test_relation_df)
        self.data_process()

    def change_type(self, relation_df):
        relation_df['label'] = relation_df['label'].astype(float)
        relation_df['id_left'] = relation_df['id_left'].astype(int)
        relation_df['id_right'] = relation_df['id_right'].astype(int)
        relation_df['record_id'] = relation_df['record_id'].astype(int)

    def data_process(self):

        with open(self.up_data / 'embedding_matrix', 'rb') as f:
                self.embedding_matrix = pickle.load(f)


    def pack_data(self, index, relation, _stage, with_distance):
        relation['label'] = relation['label'].astype(float)
        relation['id_left'] = relation['id_left'].astype(int)
        relation['id_right'] = relation['id_right'].astype(int)
        index = self._convert_to_list_index(index, len(relation))
        relation = relation.iloc[index].reset_index(drop=True)
        # relation = relation.reindex(columns = index).reset_index(drop=True)
        relation = relation[relation['label'] == 1]
        relation_l = []
        
        poi_coodis = self.right_text_df['location_right'].tolist()
        poi_coodis = list(map(eval, poi_coodis))
        poi_coodis = np.array(poi_coodis)
        query_coordis = self.left_text_df['location_left'].tolist()
        query_coordis = list(map(eval, query_coordis))
        query_coordis = np.array(query_coordis)
        poi_num = len(self.right_text_df)
        for index, row in relation.iterrows():
            record_id = row.record_id
            id_left = row.id_left
            id_right = row.id_right
            
            location = query_coordis[int(id_left)]
            distance = row.distance
            candidates_distance = spatial.distance.cdist(np.array([location]), poi_coodis)[0]/1000
            max_score_spatial = np.max(candidates_distance)
            candidates_distance = list(1 - candidates_distance / max_score_spatial)  # convert to distance score
            # candidates_distance = list(1 / candidates_distance)

            candidate_pois = list(range(poi_num))
            # if poiid in candidate_pois:
            idx_ = candidate_pois.index(id_right)
            del candidate_pois[idx_]
            del candidates_distance[idx_]
            if _stage == 'train':
                negative_idx_samples = np.random.choice(list(range(len(candidate_pois))), size=self._parser.num_neg, replace=False)
                candidate_pois = np.array(candidate_pois)[negative_idx_samples]
                if with_distance:
                    candidates_distance = np.array(candidates_distance)[negative_idx_samples]
            # l2geohash = location_map2(location)
            
            relation_l.append([record_id, id_left, 1.0, id_right, 1-distance/max_score_spatial])
            # relation_l.append([record_id, id_left, 1.0, id_right, 1 / distance])

            # tile the negative samples
            neg_list = [[record_id, id_left, 0.0]]
            neg_list = neg_list * len(candidate_pois)
            df_neg_l = pd.DataFrame(neg_list, columns=['record_id', 'id_left', 'label'])
            df_neg_l['id_right'] = candidate_pois
            if with_distance:
                df_neg_l['distance'] = candidates_distance
            neg_list_ = df_neg_l.values.tolist()
            relation_l = relation_l + neg_list_
            # for i in range(len(candidate_pois)):
            #     relation_l.append([record_id, id_left, 0.0, l2geohash, candidate_pois[i], candidates_distance[i]])
        
        relation2 = pd.DataFrame(relation_l, columns=['record_id', 'id_left', 'label', 'id_right', 'distance'])

        left = self.left_text_df.loc[relation2['id_left'].unique()]
        left['location_left'] = left['location_left'].map(eval)
        
        left['location_left'] = left['location_left'].map(self.location_map2)
        right = self.right_text_df.loc[relation2['id_right'].unique()]
        right['location_right'] = right['location_right'].map(eval)
        right['location_right'] = right['location_right'].map(self.location_map2)

        return relation2, left, right

    def unpack(self, data_pack):
        relation, left, right = data_pack
        index = list(range(len(relation)))
        left_df = left.loc[relation['id_left'][index]].reset_index()
        right_df = right.loc[relation['id_right'][index]].reset_index()
        joined_table = left_df.join(right_df)
        for column in relation.columns:
            if column not in ['id_left', 'id_right']:
                labels = relation[column][index].to_frame()
                labels = labels.reset_index(drop=True)
                joined_table = joined_table.join(labels)

        columns = list(joined_table.columns)

        y = np.vstack(np.asarray(joined_table['label']))

        x = joined_table[columns].to_dict(orient='list')
        for key, val in x.items():
            x[key] = np.array(val)

        return x, y

    def location_map2(self, obj):
        dXMin = 366950.2449227313       #39.53912
        dXMax = 541972.8942822593        #40.96375
        dYMin = 4377750.661527712        #115.4517
        dYMax = 4534852.758361747        #117.4988

        # dXMin = 211537      #30.0016  shanghai  the max x 401729.64888385497, the max y 3543502.07441587  the min x 211537.682914985, the min y 3321378.324052256
        # dXMax = 401730      #31.9981      31.99793815612793 30.00271797180176 121.95459365844728 120.00383377075195; 31.99808 30.00162000000001 121.97192 120.001339
        # dYMin = 3321378       #120.0013
        # dYMax = 3543503      #121.9720

        m_nGridCount = 999
        m_dOriginX = dXMin
        m_dOriginY = dYMin
        dSizeX = (dXMax - dXMin) / m_nGridCount
        dSizeY = (dYMax - dYMin) / m_nGridCount
        nXCol = int((obj[0] - m_dOriginX) / dSizeX)
        nYCol = int((obj[1] - m_dOriginY) / dSizeY)

        return [nXCol, nYCol]

    def _convert_to_list_index(self, index, length):
        if isinstance(index, int):
            index = [index]
        elif isinstance(index, slice):
            index = list(range(*index.indices(length)))
        return index

class MatchingHistogram(object):

    def __init__(self, bin_size: int = 30, embedding_matrix=None,
                 normalize=True, mode: str = 'LCH'):
        """The constructor."""
        self._hist_bin_size = bin_size
        self._embedding_matrix = embedding_matrix
        if normalize:
            self._normalize_embedding()
        self._mode = mode

    def _normalize_embedding(self):
        """Normalize the embedding matrix."""
        l2_norm = np.sqrt(
            (self._embedding_matrix * self._embedding_matrix).sum(axis=1)
        )
        self._embedding_matrix = \
            self._embedding_matrix / l2_norm[:, np.newaxis]

    def transform(self, input_: list) -> list:
        """Transform the input text."""
        text_left, text_right = input_
        matching_hist = np.ones((len(text_left), self._hist_bin_size),
                                dtype=np.float32)
        embed_left = self._embedding_matrix[text_left]
        embed_right = self._embedding_matrix[text_right]
        matching_matrix = embed_left.dot(np.transpose(embed_right))
        for (i, j), value in np.ndenumerate(matching_matrix):
            bin_index = int((value + 1.) / 2. * (self._hist_bin_size - 1.))
            matching_hist[i][bin_index] += 1.0
        if self._mode == 'NH':
            matching_sum = matching_hist.sum(axis=1)
            matching_hist = matching_hist / matching_sum[:, np.newaxis]
        elif self._mode == 'LCH':
            matching_hist = np.log(matching_hist)
        return matching_hist.tolist()


class my_Datasets(data.Dataset):

    def __init__(self, relation, interaction, stage, batch_size=32, resample=False, bin_size=10, shuffle=True):
        self._orig_relation = relation.copy()
        self.interaction = interaction
        self._batch_size = batch_size
        self._batch_indices = None
        self._shuffle = shuffle
        self.reset_index()
        self._resample = resample
        self.stage = stage
        self._with_distance = True
        self._bin_size = bin_size
        self.with_histogram = False

    def reset_index(self):
        index_pool = []
        step_size = 1
        num_instances = int(len(self._orig_relation) / step_size)
        for i in range(num_instances):
            lower = i * step_size
            upper = (i+1) * step_size
            indices = list(range(lower, upper))
            if indices:
                index_pool.append(indices)
        if self._shuffle == True:
            np.random.shuffle(index_pool)
        self._batch_indices = []
        for i in range(math.ceil(num_instances / self._batch_size)):
            lower = self._batch_size * i
            upper = self._batch_size * (i + 1)
            candidates = index_pool[lower:upper]
            candidates = sum(candidates, [])
            self._batch_indices.append(candidates)

    def __getitem__(self, item):
        if isinstance(item, slice):
            indices = sum(self._batch_indices[item], [])
        elif isinstance(item, Iterable):
            indices = [self._batch_indices[i] for i in item]
        else:
            indices = self._batch_indices[item]
        data_dict = self.interaction.pack_data(indices, self._orig_relation, self.stage, with_distance=self._with_distance)
        x, y = self.interaction.unpack(data_dict)

        if self.with_histogram == True:
            self.interaction.Histogram(x, y, bin_size=self._bin_size, hist_mode='CH')
        return x, y

    def __iter__(self):
        """Create a generator that iterate over the Batches."""
        if self._resample or self._shuffle:
            self.on_epoch_end()
        for i in range(len(self)):
            yield self[i]

    def on_epoch_end(self):
        """Reorganize the index array if needed."""
        self.reset_index()

    def __len__(self):
        """Get the total number of batches."""
        return len(self._batch_indices)