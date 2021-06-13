import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def compute_gini(clazz):
    return 1 - sum([(c/clazz.shape[0]) ** 2 for c in np.bincount(clazz)])


def compute_accuracy(a, b):
    try:
        iter(a)
        a = np.array(a)
    except TypeError:
        a = np.array([a])
    try:
        iter(b)
        b = np.array(b)
    except TypeError:
        b = np.array([b])
    assert len(a.shape) == 1 and len(b.shape) == 1 and a.shape == b.shape, \
        'a and b should have the same dimension of (n, )'
    return (a == b).sum() / a.shape[0]


class Dataset:
    FTYPE_REAL = 0
    FTYPE_CATEGORICAL = 1
    FTYPE_UNUSED = 2
    FTYPE_CLASS = 3

    BezdekIris: 'Dataset' = None
    BreastCancer: 'Dataset' = None
    Glass: 'Dataset' = None
    Ionosphere: 'Dataset' = None
    Iris: 'Dataset' = None
    Wine: 'Dataset' = None

    def __init__(self, filepath, feature_types, header=None, delim=','):
        """Initialize a Dataset

        Args:
            filepath (str): path to the csv file
            feature_types (Iterable / Callable): An iterable contains feature type of each column, or a callable that gives corresponding feature type from column index
            header (Iterable, optional): The header of the csv file. Defaults to None.
            delim (str, optional): The string used to seperate columns in the csv file. Defaults to ','.
        """
        # Read csv in
        df = pd.read_csv(filepath, sep=delim, header=header)
        # Sanity check the feature types
        feature_types = Dataset._sanity_check_ftypes(
            df.shape[1], feature_types)
        # Drop unused columns
        unused_features = [x[0] for x in filter(
            lambda x: x[1] == Dataset.FTYPE_UNUSED, enumerate(feature_types))]
        df.drop(columns=unused_features, inplace=True)
        # Split feature and class label
        class_label = [x[0] for x in filter(
            lambda x: x[1] == Dataset.FTYPE_CLASS, enumerate(feature_types))]
        target_classes = df[class_label].to_numpy().reshape(-1)
        df.drop(columns=class_label, inplace=True)
        # One-hot categorical features
        cate_features = [x[0] for x in filter(
            lambda x: x[1] == Dataset.FTYPE_CATEGORICAL, enumerate(feature_types))]
        df = pd.get_dummies(df, columns=cate_features).astype('float32')
        # Encode the class labels
        self._class_dict = {}
        self._r_class_dict = []
        _class = []
        for clazz in target_classes:
            if clazz not in self._class_dict:
                self._class_dict[clazz] = len(self._r_class_dict)
                self._r_class_dict.append(clazz)
            _class.append(self._class_dict[clazz])
        # Set final results
        self._class = np.array(_class)
        self._data = df.to_numpy()

    def holdout(self, train_test_ratio=0.7, shuffle=True):
        """Holdout validation

        Args:
            train_test_ratio (float, optional): The ratio of train data to split the dataset. Defaults to 0.7.
            shuffle (bool, optional): Should the data be shuffled. Defaults to True.

        Returns:
            tuple: (train_data, train_labels, test_data, test_labels)
        """
        assert 0 <= train_test_ratio <= 1, 'Train-Test ratio should be in [0, 1]'
        data, class_ = self._shuffle() \
            if shuffle else (self._data.copy(), self._class.copy())
        sep = int(self._class.shape[0] * train_test_ratio)
        return data[:sep, :], class_[:sep], data[sep:, :], class_[sep:]

    def kfold(self, k=3, shuffle=True):
        """K-Fold validation

        Args:
            k (int, optional): The 'K' in kfold. Defaults to 3.
            shuffle (bool, optional): Should the data be shuffled. Defaults to True.

        Yields:
            tuple: (train_data, train_labels, test_data, test_labels)
        """
        data, class_ = self._shuffle() \
            if shuffle else (self._data.copy(), self._class.copy())
        res = []
        for train, test in KFold(k).split(class_):
            res.append((data[train, :], class_[train],
                       data[test, :], class_[test]))
        return res

    def convert_label(self, x):
        try:
            iter(x)
            # Handle x as a list
            return np.array([self._r_class_dict[_] for _ in x])
        except TypeError:
            # Handle x as a single value
            return self._r_class_dict[x]

    @property
    def data(self):
        return self._data.copy()

    @property
    def clazz(self):
        return self._class.copy()

    def _shuffle(self):
        tmp = np.concatenate((self._class.reshape(-1, 1), self._data), axis=1)
        np.random.shuffle(tmp)
        return tmp[:, 1:], tmp[:, 0].astype(np.int32)

    def _sanity_check_ftypes(n_features, ftypes):
        if hasattr(ftypes, '__call__'):
            ftypes = [ftypes(x) for x in range(n_features)]
        else:
            assert len(ftypes) == n_features, \
                'The feature_types length does not match with the input data. Expecting %d, got %d' \
                % (n_features, len(ftypes))
            ftypes = list(ftypes)
        for idx, x in enumerate(ftypes):
            if not 0 <= x <= 3:
                ftypes[idx] = Dataset.FTYPE_REAL
                print('Detect an undefined feature type: %d, this value will be changed to FTYPE_REAL. Check Dataset.FTYPE_* for proper usage.' % x)
        return tuple(ftypes)

    def __len__(self):
        return self._class.shape[0]

    def __getitem__(self, idx):
        return self._data[idx].reshape(1, -1), self._class[idx]


Dataset.BezdekIris = Dataset('./data/iris/bezdekiris.data',
                             lambda x: Dataset.FTYPE_REAL if x != 4 else Dataset.FTYPE_CLASS)
Dataset.BreastCancer = Dataset('./data/breast-cancer/breast-cancer.data',
                               [Dataset.FTYPE_CATEGORICAL] * 6 + [Dataset.FTYPE_REAL] + [Dataset.FTYPE_CATEGORICAL] * 2 + [Dataset.FTYPE_CLASS])
Dataset.Glass = Dataset('./data/glass/glass.data',
                        [Dataset.FTYPE_UNUSED] + [Dataset.FTYPE_REAL] * 9 + [Dataset.FTYPE_CLASS])
Dataset.Ionosphere = Dataset('./data/ionosphere/ionosphere.data',
                             lambda x: Dataset.FTYPE_CLASS if x == 34 else Dataset.FTYPE_REAL)
Dataset.Iris = Dataset('./data/iris/iris.data',
                       [Dataset.FTYPE_REAL, Dataset.FTYPE_REAL, Dataset.FTYPE_REAL, Dataset.FTYPE_REAL, Dataset.FTYPE_CLASS])
Dataset.Wine = Dataset('./data/wine/wine.data',
                       lambda x: Dataset.FTYPE_CLASS if x == 0 else Dataset.FTYPE_REAL)
