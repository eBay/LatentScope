import numpy as np


def run_pearson(metric_value, exist_edges, batch_size=8192):
    # Running pearson
    result_edges = []

    if len(exist_edges) == 0:
        return []

    for idx in range(0, len(exist_edges), batch_size):
        X = np.stack([metric_value[u] for (u, v) in exist_edges[idx: idx + batch_size]])
        Y = np.stack([metric_value[v] for (u, v) in exist_edges[idx: idx + batch_size]])

        EX = np.mean(X, axis=1, keepdims=True)
        EY = np.mean(Y, axis=1, keepdims=True)
        DX = np.std(X, axis=1) + 1e-7
        DY = np.std(Y, axis=1) + 1e-7

        corr = np.mean((X - EX) * (Y - EY), axis=1) / (DX * DY)
        for i in range(len(corr)):
            if corr[i] > 0.4:
                result_edges.append(exist_edges[idx + i])

    return result_edges
