import numpy as np
import pandas as pd


def euclidean_distance(x1, x2):
    return np.sqrt((x1 - x2) ** 2)


def skip(data: np.array, k: int):
    data_without_nan = pd.DataFrame(data.copy())
    data_without_nan = data_without_nan.dropna(axis=0).to_numpy()
    data_is_nan = np.isnan(data)  # bool
    row, col = data.shape

    for i in range(row):  # row
        if True in data_is_nan[i]:  # are NaN values
            distance_data = euclidean_distance(
                data[i], data_without_nan)  # broadcast
            # 創建一个遮罩，將NaN值排除在外
            masked = np.ma.masked_where(np.repeat(data_is_nan[i].reshape(
                1, -1), len(distance_data), axis=0), distance_data)
            # 計算每个非NaN值的平均距離
            mean = masked.mean(axis=1)  # flatten
            # 創建一个包含距离和對應行的列表
            distance_pair_lst = [[mean[j], data_without_nan[j]]
                                 for j in range(len(mean))]
            # 照距離排序
            mean_sort = sorted(distance_pair_lst,
                               key=lambda distance: distance[0])
            # 取最近的k行，並計算它們的均值
            k_mean = np.concatenate([mean_sort[j][1].reshape(
                1, -1) for j in range(k)], axis=0).mean(axis=0)

            for j in range(len(k_mean)):
                if data_is_nan[i, j]:
                    data[i, j] = k_mean[j]

    return data


def distance_mean(data: np.array, k: int):
    data_is_nan = np.isnan(data)
    row, col = data.shape

    for i in range(row):
        if True in data_is_nan[i]:
            filter_data = data[~np.any(data_is_nan & np.repeat(
                [data_is_nan[i]], row, axis=0), axis=1)]
            # print(filter_data)
            # 計算當前行與其他行之間的距離
            distance_data = euclidean_distance(data[i], filter_data)
            # print(distance_data)
            masked_distance_data = np.ma.masked_invalid(distance_data)
            # print(masked_distance_data)
            mean = masked_distance_data.mean(axis=1)  # 距離壓扁
            # print(mean)
            distance_pair_lst = [[d, row] for d, row in zip(mean, filter_data)]
            # print(distance_pair_lst)
            # 照距离排序
            mean_sort = sorted(distance_pair_lst, key=lambda distance: (
                distance[0], id(distance[1])))
            # print(mean_sort)
            k_mean = np.concatenate([mean_sort[j][1].reshape(
                1, -1) for j in range(k)], axis=0).mean(axis=0)
            # print(k_mean)
            for j in range(len(k_mean)):
                if data_is_nan[i, j]:
                    data[i, j] = k_mean[j]

    return data


def fill_nan_by_knn(df: pd.DataFrame, k: int, method: str):
    # 1. 防呆
    assert k > 1
    assert method in ["distance_mean", "skip"]

    data = df.to_numpy()
    if method == 'skip':
        filled_data = skip(data, k)

    elif method == 'distance_mean':
        filled_data = distance_mean(data, k)
    df_without_nan = pd.DataFrame(filled_data, columns=df.columns)

    return df_without_nan
