# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


def main(timed_embeddings):

    reader = pd.read_csv(timed_embeddings, header=None, sep=' ', iterator=True)
    df = reader.get_chunk(1000)
    # print(df)
    row_size, col_size = df.shape
    data_features = df[list(range(2, col_size-1))]
    data_label = df[col_size - 1]
    train_features, test_features, train_label, test_label = train_test_split(data_features, data_label, random_state=1)

    linear_regression = LinearRegression()
    linear_regression.fit(train_features, train_label)
    label_predict = linear_regression.predict(test_features)
    print("MAE:", metrics.mean_absolute_error(test_label, label_predict))


if __name__ == '__main__':
    main(timed_embeddings='sanfrancisco/labeled_emb/my_model/sf_shortest_wn160_d128_ns5_ws5_time_21.embeddings',
         )
