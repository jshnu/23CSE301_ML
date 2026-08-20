import numpy as np
import pandas as pd

def label_encoder_fit(y):
    classes = np.unique(y)
    class_to_index = {c: i for i, c in enumerate(classes)}
    state = {'classes': classes, 'class_to_index': class_to_index}
    return state

def label_encoder_transform(y, state):
    return np.array([state['class_to_index'][v] for v in y])

def label_encoder_inverse_transform(indices, state):
    return state['classes'][np.array(indices)]

def feature_encoder_fit(X, cat_enc='onehot'):
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    state = {
        'numeric_cols': num_cols,
        'categorical_cols': cat_cols,
        'categorical_encoding': cat_enc
    }

    if cat_enc == 'ordinal':
        cat_maps = {}
        for col in cat_cols:
            unique_vals = X[col].dropna().unique()
            cat_maps[col] = {v: i for i, v in enumerate(unique_vals)}
        state['category_maps'] = cat_maps
        state['output_cols'] = num_cols + cat_cols
    else:
        dummies = pd.get_dummies(X[cat_cols], prefix_sep='__')
        state['output_cols'] = num_cols + list(dummies.columns)

    return state

def feature_encoder_transform(X, state):
    X = X.copy()
    num_cols = state['numeric_cols']
    cat_cols = state['categorical_cols']
    encoding = state['categorical_encoding']

    if encoding == 'ordinal':
        for col in cat_cols:
            mapping = state['category_maps'][col]
            X[col] = X[col].map(mapping).fillna(-1).astype(int)
        return X[state['output_cols']].values
    else:
        dummies = pd.get_dummies(X[cat_cols], prefix_sep='__')
        X = X.drop(columns=cat_cols)
        X = pd.concat([X, dummies], axis=1)
        X = X.reindex(columns=state['output_cols'], fill_value=0)
        return X.values

def imputer_fit(X, num_strat='mean', cat_strat='mode'):
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    fills = {}

    for col in num_cols:
        if num_strat == 'mean':
            fills[col] = X[col].mean()
        elif num_strat == 'median':
            fills[col] = X[col].median()
        elif num_strat == 'mode':
            mode = X[col].mode()
            fills[col] = mode[0] if not mode.empty else 0
        else:
            raise ValueError(f"Unknown num_strat: {num_strat}")

    for col in cat_cols:

        mode = X[col].mode()
        fills[col] = mode[0] if not mode.empty else 'missing'

    state = {
        'numeric_cols': num_cols,
        'categorical_cols': cat_cols,
        'fill_values': fills
    }
    return state

def imputer_transform(X, state):
    X = X.copy()
    for col in state['numeric_cols'] + state['categorical_cols']:
        X[col] = X[col].fillna(state['fill_values'][col])
    return X

def compute_distance(x, y, metric='euclidean', p=2):
    x = np.asarray(x)
    y = np.asarray(y)

    if metric == 'euclidean':
        return np.sqrt(np.sum((x - y) ** 2))
    elif metric == 'manhattan':
        return np.sum(np.abs(x - y))
    elif metric == 'minkowski':
        return np.sum(np.abs(x - y) ** p) ** (1.0 / p)
    elif metric == 'chebyshev':
        return np.max(np.abs(x - y))
    elif metric == 'cosine':
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        if denom == 0:
            return 1.0
        return 1.0 - np.dot(x, y) / denom
    else:
        raise ValueError(f"Unknown metric: {metric}")

def bubble_sort_pairs(pairs):
    arr = pairs[:]
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

def merge_sort_pairs(pairs):
    if len(pairs) <= 1:
        return pairs
    mid = len(pairs) // 2
    left = merge_sort_pairs(pairs[:mid])
    right = merge_sort_pairs(pairs[mid:])
    return _merge(left, right)

def _merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def quick_sort_pairs(pairs):
    if len(pairs) <= 1:
        return pairs
    pivot = pairs[len(pairs) // 2]
    less = [x for x in pairs if x < pivot]
    equal = [x for x in pairs if x == pivot]
    greater = [x for x in pairs if x > pivot]
    return quick_sort_pairs(less) + equal + quick_sort_pairs(greater)

def sort_indices(dists, algorithm='quick'):
    pairs = list(zip(dists, range(len(dists))))

    if algorithm == 'bubble':
        sort_pairs = bubble_sort_pairs(pairs)
    elif algorithm == 'merge':
        sort_pairs = merge_sort_pairs(pairs)
    elif algorithm == 'quick':
        sort_pairs = quick_sort_pairs(pairs)
    else:
        raise ValueError(f"Unknown sorting algorithm: {algorithm}")

    return [idx for _, idx in sort_pairs]

def compute_weights(dists, scheme='uniform', epsilon=1e-10):
    dists = np.asarray(dists)

    if scheme == 'uniform':
        return np.ones_like(dists)
    elif scheme == 'distance':

        return 1.0 / (dists + epsilon)
    else:
        raise ValueError(f"Unknown weights scheme: {scheme}")

def knn_fit(X, y, params):

    imp_state = imputer_fit(
        X,
        num_strat=params.get('numeric_imputation', 'mean'),
        cat_strat=params.get('categorical_imputation', 'mode')
    )
    X_imp = imputer_transform(X, imp_state)

    enc_state = feature_encoder_fit(
        X_imp,
        cat_enc=params.get('categorical_encoding', 'onehot')
    )
    X_enc = feature_encoder_transform(X_imp, enc_state)

    y_enc_state = label_encoder_fit(y)
    y_encoded = label_encoder_transform(y, y_enc_state)

    model = {
        'Xtr': X_enc,
        'ytr': y_encoded,
        'classes': y_enc_state['classes'],
        'imputer_state': imp_state,
        'encoder_state': enc_state,
        'y_encoder_state': y_enc_state,
        'params': params
    }
    return model

def knn_kneighbors(model, X):

    X_imp = imputer_transform(X, model['imputer_state'])
    X_enc = feature_encoder_transform(X_imp, model['encoder_state'])

    k = model['params'].get('n_neighbors', 5)
    metric = model['params'].get('metric', 'euclidean')
    p = model['params'].get('p', 2)
    sort_algo = model['params'].get('sort_algorithm', 'quick')

    nbrs_list = []
    dists_list = []

    for x in X_enc:
        dists = np.zeros(len(model['Xtr']))
        for i, x_train in enumerate(model['Xtr']):
            dists[i] = compute_distance(x, x_train, metric, p)

        sort_inds = sort_indices(dists, sort_algo)
        selected = sort_inds[:k]
        sel_d = dists[selected]

        nbrs_list.append(selected)
        dists_list.append(sel_d)

    return np.array(nbrs_list), np.array(dists_list)

def knn_predict(model, X):
    neighbors, dists = knn_kneighbors(model, X)
    ytr = model['ytr']
    classes = model['classes']
    tie_break = model['params'].get('vote_tie_break', 'nearest')
    w_scheme = model['params'].get('weights', 'uniform')

    preds = []

    for i, nbrs in enumerate(neighbors):
        labels = ytr[nbrs]
        d = dists[i]

        weights = compute_weights(d, scheme=w_scheme)

        cw = np.zeros(len(classes))
        for label, w in zip(labels, weights):
            cw[label] += w

        mx = np.max(cw)
        tied = np.where(cw == mx)[0]

        if len(tied) == 1:
            pred = tied[0]
        else:
            if tie_break == 'lowest_label':
                pred = tied[0]
            elif tie_break == 'nearest':

                pred = None
                for idx in nbrs:
                    if ytr[idx] in tied:
                        pred = ytr[idx]
                        break
                if pred is None:
                    pred = tied[0]
            else:
                pred = tied[0]

        preds.append(pred)

    return label_encoder_inverse_transform(np.array(preds), model['y_encoder_state'])

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    np.random.seed(42)
    n = 200

    X = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'feature3': np.random.randn(n)
    })

    y = (X['feature1'] + X['feature2'] + X['feature3'] > 0).astype(int)
    y = y.map({0: 'negative', 1: 'positive'})

    X.iloc[:10, 0] = np.nan
    X.iloc[20:25, 1] = np.nan
    X.iloc[30:35, 2] = np.nan

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    for w_scheme in ['uniform', 'distance']:
        params = {
            'n_neighbors': 5,
            'metric': 'euclidean',
            'sort_algorithm': 'quick',
            'numeric_imputation': 'mean',
            'categorical_imputation': 'mode',
            'categorical_encoding': 'onehot',
            'vote_tie_break': 'nearest',
            'weights': w_scheme
        }

        model = knn_fit(Xtr, ytr, params)
        preds = knn_predict(model, Xte)

        acc = np.mean(preds == yte)
        print(f"Weights: {w_scheme:10s} -> Accuracy: {acc:.4f}")

    params['weights'] = 'distance'
    model_dist = knn_fit(Xtr, ytr, params)
    sample_idx = 0
    sample = Xte.iloc[[sample_idx]]
    pred = knn_predict(model_dist, sample)
    print(f"\nSample true label: {yte.iloc[sample_idx]}")
    print(f"Predicted label (distance weighting): {pred[0]}")
