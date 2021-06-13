import numpy as np
import tqdm

from models import DecisionTree, RandomForest
from utils import Dataset, compute_accuracy


def model_provider():
    n_trees = 5
    tree_bagging = True
    feature_bagging = True
    depth_lim = 8
    min_samples = 0

    # return DecisionTree(feature_bagging, depth_lim, min_samples)
    return RandomForest(n_trees, tree_bagging, feature_bagging, depth_lim, min_samples)


def run(dataset, use_holdout=True, holdout_ratio=0.8, kfold=8, episodes=5):
    assert use_holdout or kfold >= 1

    accu_scores_train = []
    accu_scores = []
    with tqdm.tqdm(total=episodes * (1 if use_holdout else kfold)) as pg:
        for _ in range(episodes):
            if use_holdout:
                train_data, train_label, test_data, test_label = dataset.holdout(
                    holdout_ratio)
                model = model_provider()
                model.train(train_data, train_label)
                accu_scores_train.append(
                    compute_accuracy(train_label, model.predict(train_data)))
                accu_scores.append(
                    compute_accuracy(test_label, model.predict(test_data)))
                pg.update(1)
            else:
                for train_data, train_label, test_data, test_label in dataset.kfold(kfold):
                    model = model_provider()
                    model.train(train_data, train_label)
                    accu_scores_train.append(
                        compute_accuracy(train_label, model.predict(train_data)))
                    accu_scores.append(
                        compute_accuracy(test_label, model.predict(test_data)))
                    pg.update(1)
    print('Training data accuracy: %.2f%%' %
          (100 * np.average(accu_scores_train)))
    print('Testing data accuracy: %.2f%%' %
          (100 * np.average(accu_scores)))


if __name__ == '__main__':
    run(Dataset.Glass, use_holdout=False)
