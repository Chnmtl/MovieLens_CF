import argparse
import statistics
import math

from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from tabulate import tabulate

CLI = argparse.ArgumentParser()
CLI.add_argument("ratings", help="Absolute path of the MovieLens 100K data file")
CLI.add_argument("model", choices=["user", "item"], help="Collaborative filtering approach")
CLI.add_argument("kfold", choices=[5, 10], type=int, help="k-fold cross validation")
CLI.add_argument("knn", choices=[10, 20, 30, 40, 50, 60, 70, 80], type=int, help="k-nearest neighbors")


def read_dataset():
    dataset = []
    with open("ml-100k/u.data", "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            user_id = int(parts[0])
            movie_id = int(parts[1])
            rating = int(parts[2])

            dataset.append((user_id, movie_id, rating))
    return dataset


# print(read_dataset())


def pearson_correlation(u1, u2):
    mean_u1 = statistics.mean(u1.values())
    mean_u2 = statistics.mean(u2.values())

    commons = set(u1.keys()).intersection(set(u2.keys()))

    dividend = sum([(u1[c] - mean_u1) * (u2[c] - mean_u2) for c in commons])

    dvr1 = math.sqrt(sum([(u1[c] - mean_u1) ** 2 for c in commons]))
    dvr2 = math.sqrt(sum([(u2[c] - mean_u2) ** 2 for c in commons]))

    divisor = dvr1 * dvr2

    try:
        return dividend / divisor
    except ZeroDivisionError:
        return 0


def user_prediction(user, movie, ratings, neighbors):
    divident = sum([n[1] * (ratings[n[0]][movie] - statistics.mean(ratings[n[0]].values())) for n in neighbors])
    divisor = sum([n[1] for n in neighbors])

    try:
        return statistics.mean(ratings[user].values()) + (divident / divisor)
    except ZeroDivisionError:
        return statistics.mean(ratings[user].values())


def user_based(train, test, knn):
    ratings, similarities = defaultdict(lambda: dict()), defaultdict(lambda: dict())
    truth, predictions = [], []

    for user_id, movie_id, rating in train:
        ratings[user_id][movie_id] = rating

    for user_id, movie_id, rating in test:
        truth.append(rating)

        others = [k for k in rating.keys() if k != user_id]
        for o in others:
            if o not in similarities[user_id]:
                similarities[user_id][o] = pearson_correlation(u1=ratings[user_id], u2=ratings[o])

        relative = [i for i in similarities[user_id].items() if movie_id in ratings[i[0]]]
        nearest = sorted(relative, key=lambda temp: temp[1], reverse=True)[:knn]

        p = user_prediction(user=user_id, movie=movie_id, ratings=ratings, neighbors=nearest)

        predictions.append(p)
    return mean_absolute_error(truth, predictions)


def present(model, knn, results):
    rows = []
    for i in range(len(results)):
        rows.append([model, knn, i + 1, results[i]])
    rows.append([model, knn, "Average", statistics.mean(results)])

    print(tabulate(rows, headers=["Model", "KNN", "Fold", "MAE"]))


if __name__ == '__main__':
    args = CLI.parse_args()

    ML100K = read_dataset()
    kf = KFold(n_splits=args.kfold)

    maes = []
    for train_index, test_index in kf.split(ML100K):
        train_data = [ML100K[i] for i in train_index]
        test_data = [ML100K[i] for i in test_index]

        if args.model == "user":
            mae = user_based(train_data, test_data, args.knn)
            maes.append(mae)

    present(results=maes, knn=args.knn, model=args.model)
