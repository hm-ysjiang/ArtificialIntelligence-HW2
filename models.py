import numpy as np

from utils import compute_gini, kfold_indices


class DecisionTree:
    class Node:
        def __init__(self, depth=0):
            self.depth = depth
            self.feature = None
            self.threshold = None
            self.label = None
            self.child1 = None
            self.child2 = None

        def split(self, data, label, feature_bagging, depth_lim, min_samples):
            # Set node label if only one label presents
            if np.unique(label).shape[0] == 1:
                self.label = label[0]
                return
            # Set node label if limit reached
            if self.depth >= depth_lim or label.shape[0] <= min_samples:
                self.label = np.argmax(np.bincount(label))
                return
            n_samples, n_features = data.shape
            # Consider only sqrt(n_features) features
            n_consider = DecisionTree._F_bagging_policy(n_features)
            # Find splitable features
            splitable = list(filter(lambda feature: np.unique(data[:, feature]).shape[0] > 1,
                                    list(range(n_features))))
            n_splitable = len(splitable)
            if n_splitable > 0:
                min_gini = None
                # Iterate through the features chosen
                for feature in (np.random.choice(splitable, min(n_consider, n_splitable), replace=False) if feature_bagging else range(n_features)):
                    # Sorted unique elements
                    values = np.unique(data[:, feature])
                    for idx in range(len(values) - 1):
                        # Try midpoints between each two unique values
                        threshold = (values[idx] + values[idx+1]) / 2

                        # Compute total Gini impurity
                        g1 = data[:, feature] <= threshold
                        g2 = np.invert(g1)
                        n1 = g1.sum()
                        n2 = n_samples - n1
                        gini = n1 * compute_gini(label[g1]) \
                            + n2 * compute_gini(label[g2])

                        # Updates
                        if self.feature is None or min_gini is None or gini < min_gini:
                            min_gini = gini
                            self.feature = feature
                            self.threshold = threshold
            else:   # If no splitable feature
                self.label = np.argmax(np.bincount(label))
                return

            # Split the data into two groups and continue the split of children
            g1 = data[:, self.feature] <= self.threshold
            g2 = np.invert(g1)
            self.child1 = DecisionTree.Node(self.depth+1)
            self.child1.split(data[g1], label[g1],
                              feature_bagging, depth_lim, min_samples)
            self.child2 = DecisionTree.Node(self.depth+1)
            self.child2.split(data[g2], label[g2],
                              feature_bagging, depth_lim, min_samples)

        def __call__(self, data):
            if self.label is not None:
                return self
            else:
                return self.child1 if data[self.feature] <= self.threshold else self.child2

    def _F_bagging_policy(x):
        return max(1, int(round(np.sqrt(x))))

    def __init__(self, feature_bagging=True, depth_lim=8, min_samples=0):
        """Initialize a Decision Tree Classifier

        Args:
            feature_bagging (bool, optional): Enable Feature-bagging or not. Defaults to True.
            depth_lim (int, optional): The depth limit of the tree. Defaults to 8.
            min_samples (int, optional): The minimum amount of samples in each node. Defaults to 0.
        """
        assert depth_lim > 0
        assert min_samples >= 0

        self.feature_bagging = feature_bagging
        self.depth_lim = depth_lim
        self.min_samples = min_samples
        self.root = DecisionTree.Node()
        self.trained = False

    def train(self, data, label):
        """
        Args:
            data (Iterable): An iterable contains training data, the dimension should be (samples, features)
            label (Iterable): An iterable contains training data, the dimension should be (samples, )
        """
        assert not self.trained, 'This tree has already been trained!'
        self.root.split(data, label, self.feature_bagging,
                        self.depth_lim, self.min_samples)
        self.trained = True

    def predict(self, data):
        """
        Args:
            data (Iterable): An iterable contains testing data, the dimension should be (samples, features)

        Returns:
            numpy.ndarray: An array of dim (samples, ), contains the predictions of each input
        """
        assert self.trained, 'This tree has not been trained yet!'

        res = [self.root] * data.shape[0]
        for _ in range(self.depth_lim):
            res = [node(data[idx]) for idx, node in enumerate(res)]
        return np.array([node.label for node in res])


class RandomForest:
    def __init__(self, n_tree=5, tree_bagging=True, feature_bagging=True, depth_lim=8, min_samples=0):
        """Initialize a Random Forest Classfier

        Args:
            n_tree (int, optional): The number of trees in this forest. Defaults to 5.
            tree_bagging (bool, optional): Enable Tree-bagging or not. Defaults to True.
            feature_bagging (bool, optional): Enable Feature-bagging or not. Defaults to True.
            depth_lim (int, optional): The depth limit of each tree. Defaults to 8.
            min_samples (int, optional): The minimum amount of samples in each tree's node. Defaults to 0.
        """
        assert n_tree >= 1
        assert tree_bagging or feature_bagging

        self.n_tree = n_tree
        self.tree_bagging = tree_bagging
        self.trees = [DecisionTree(feature_bagging, depth_lim, min_samples)
                      for _ in range(n_tree)]
        self.trained = False

    def train(self, data, label):
        """
        Args:
            data (Iterable): An iterable contains training data, the dimension should be (samples, features)
            label (Iterable): An iterable contains training data, the dimension should be (samples, )
        """
        assert not self.trained, 'This forest has already been trained!'
        if not type(data) is np.ndarray:
            data = np.array(data)
        if not type(label) is np.ndarray:
            label = np.array(label)

        if self.tree_bagging:
            for idx, fold_idx in enumerate(kfold_indices(self.n_tree, label.shape[0])):
                self.trees[idx].train(data[fold_idx[0]], (label[fold_idx[0]]))
        else:
            [tree.train(data, label) for tree in self.trees]
        self.trained = True

    def predict(self, data):
        """
        Args:
            data (Iterable): An iterable contains testing data, the dimension should be (samples, features)

        Returns:
            numpy.ndarray: An array of dim (samples, ), contains the predictions of each input
        """
        assert self.trained, 'This forest has not been trained yet!'
        if not type(data) is np.ndarray:
            data = np.array(data)
        return np.array([np.argmax(np.bincount(votes)) for votes in np.array([tree.predict(data) for tree in self.trees]).T])
