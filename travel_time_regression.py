# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

data_matrix = pd.read_csv('sanfrancisco/labeled_emb/my_model/sanfrancisco_shortest_wn160_d128_ns5_ws5_time_7.embeddings', header=None, sep=' ',)
print(data_matrix.head())
X = data_matrix[3:]
print(X)