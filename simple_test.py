from sklearn.semi_supervised import LabelPropagation
from experiment_functions import *
from OVA_TSVM import *
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from self_learning import *
import time
import warnings
warnings.filterwarnings("ignore")


def read_dna():
    df1 = load_svmlight_file("data/dna.scale")
    x1 = df1[0].todense()
    y1 = df1[1]
    df2 = load_svmlight_file("data/dna.scale.t")
    x2 = df2[0].todense()
    y2 = df2[1]
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    # label transform to 0..K-1
    y -= 1
    return x, y


def simple_test():
    # read and split data
    x, y = read_dna()
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.99, random_state=40)
    print("shape of labeled part:")
    print(xTrain.shape, yTrain.shape)
    print("shape of unlabeled part:")
    print(xTest.shape, yTest.shape)
    print("class distribution of labeled examples:")
    print([np.sum(yTrain == i) for i in range(len(np.unique(y)))])
    print("class distribution of unlabeled examples:")
    print([np.sum(yTest == i) for i in range(len(np.unique(y)))])

    # purely supervised classification
    print("random forest:")
    t0 = time.time()
    model = RandomForestClassifier(n_estimators=200, oob_score=True, n_jobs=-1)
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    print("accuracy:", accuracy_score(yTest, yPred))
    print("f1-score:", f1_score(yTest, yPred, average="weighted"))
    t1 = time.time()
    print("random forest is done")
    print("time:", t1-t0, "seconds")

    # multi-class self-learning algorithm
    print("msla:")
    t0 = time.time()
    H, thetas = msla(xTrain, yTrain, xTest)
    yPred = H.predict(xTest)
    print("optimal theta at each step:")
    print(thetas)
    print("accuracy:", accuracy_score(yTest, yPred))
    print("f1-score:", f1_score(yTest, yPred, average="weighted"))
    t1 = time.time()
    print("msla is done!")
    print("time:", t1-t0, "seconds")

    # multi-class self-learning algorithm with fixed theta
    theta = 0.7
    max_iter = 10
    print("fsla with theta={}:".format(theta))
    t0 = time.time()
    H = fsla(xTrain, yTrain, xTest, theta, max_iter)
    yPred = H.predict(xTest)
    print("accuracy:", accuracy_score(yTest, yPred))
    print("f1-score:", f1_score(yTest, yPred, average="weighted"))
    t1 = time.time()
    print("fsla is done!")
    print("time:", t1-t0, "seconds")


#
# X_new, y_new, yTestShuffled = Make_SSL_Train_Set(xTrain, yTrain, xTest, yTest, binary=False)
# #
# label_prop_model = LabelSpreading(kernel='rbf', alpha=0.5, gamma=0.01, n_jobs=-1, tol=1e-3)
# # Fitting the model
# label_prop_model.fit(X_new, y_new)
# yPred = label_prop_model.predict(X_new[y_new == -1, :])
# print("Label Spreading with gamma = 0.1:")
# print("Accuracy:", accuracy_score(yTestShuffled, yPred))
# print("F1-score:", f1_score(yTestShuffled, yPred))

# tracemalloc.start()
# snapshot = tracemalloc.take_snapshot()
# display_top(snapshot)
# label_prop_model = LabelSpreading(kernel='rbf', alpha=0.5, gamma=0.5, n_jobs=-1, tol=1e-3)
# # Fitting the model
# label_prop_model.fit(X_new, y_new)
# yPred = label_prop_model.predict(X_new[y_new == -1, :])
# print("Label Spreading with gamma = 0.5:")
# print("Accuracy:", accuracy_score(yTestShuffled, yPred))
# print("F1-score:", f1_score(yTestShuffled, yPred, average="weighted"))
#
# label_prop_model = LabelSpreading(kernel='rbf', alpha=0.5, gamma=1.5, n_jobs=-1, tol=1e-3)
# # Fitting the model
# label_prop_model.fit(X_new, y_new)
# yPred = label_prop_model.predict(X_new[y_new == -1, :])
# print("Label Spreading with gamma = 1.5:")
# print("Accuracy:", accuracy_score(yTestShuffled, yPred))
# print("F1-score:", f1_score(yTestShuffled, yPred, average="weighted"))


# t0 = time.time()
# label_prop_model = LabelSpreading(kernel='rbf', alpha=0.5, gamma=10, n_jobs=-1, tol=1e-3)
# # Fitting the model
# label_prop_model.fit(X_new, y_new)
# yPred = label_prop_model.predict(X_new[y_new == -1, :])
# print("Label Spreading with gamma = 10:")
# print("Accuracy:", accuracy_score(yTestShuffled, yPred))
# print("F1-score:", f1_score(yTestShuffled, yPred, average="weighted"))
# t1 = time.time()
# print("Computational time:", t1-t0)

# t0 = time.time()
# print("TSVM:")
# yTestShuffled, yPred = one_vs_all_tsvm(xTrain, yTrain, xTest, yTest, timeout=None)
# print("Accuracy:", accuracy_score(yTestShuffled, yPred))
# print("F1-score:", f1_score(yTestShuffled, yPred, average="weighted"))
# print("TSVM is done!")
# t1 = time.time()
# print("It took ", t1-t0, "seconds")




# d = []
# K = len(np.unique(yTrain))
# p = xTrain.shape[1]
#from pomegranate import GeneralMixtureModel, BayesClassifier, NormalDistribution, NaiveBayes, MultivariateGaussianDistribution
# for c in range(K):
#     d.append(NaiveBayes.from_samples(NormalDistribution, X_new[y_new == c,:]))
# model = BayesClassifier(d)
# model.fit(X_new, y_new)
# model = NaiveBayes.from_samples(NormalDistribution, X_new, y_new, verbose=True, n_jobs=-1)
# yPred = model.predict(X_new[y_new==-1,:])
# print("Semi-supervised Naive Bayes:")
# print("Accuracy:", accuracy_score(yTestShuffled, yPred))
# print("F1-score:", f1_score(yTestShuffled, yPred, average="weighted"))



# SaveBinaryExpreminetSeries(x, y, "Wisconsin", 0.98, n_estimators=200, base="RF")
# SaveExpreminetSeries(x, y, "MNIST", 0.995, n_estimators=200, base="RF")
# SaveExpreminetSeries(x, y, "Pendigits", 0.99, n_estimators=200, base="RF")
# SaveExpreminetSeries(x, y, "Acoustic", 0.9995, n_estimators=200, base="RF")
# SaveExpreminetSeries(x, y, "Vehicle", 0.9995, n_estimators=200, base="RF")

# read_data = ReadDataset()
# x, y = read_data.read('vowel')
# SaveExpreminetSeries(x, y, "Vowel", 0.9, n_estimators=200, base="RF")
#
# x, y = read_data.read('dna')
# SaveExpreminetSeries(x, y, "DNA", 0.99, n_estimators=200, base="RF")
#
# x, y = read_data.read('mnist')
# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# # SaveExpreminetSeries(x, y, "MNIST", 0.995, n_estimators=200, base="RF")
# SaveExpreminetSeries(x, y, "MNIST", 0.9975, n_estimators=200, base="RF")
#
# x, y = read_data.read('pendigits')
# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# SaveExpreminetSeries(x, y, "Pendigits", 0.99, n_estimators=200, base="RF")

# x, y = read_data.read('vehicle')
# SaveExpreminetSeries(x, y, "SensIT", 0.9995, n_estimators=200, base="RF")
#
# x, y = read_data.read('fashion')
# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# SaveExpreminetSeries(x, y, "Fashion", 0.9975, n_estimators=200, base="RF")
#
# x, y = read_data.read('har')
# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# SaveExpreminetSeries(x, y, "HAR", 0.99, n_estimators=200, base="RF")
#
# x, y = read_data.read('isolet')
# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# SaveExpreminetSeries(x, y, "Isolet", 0.95, n_estimators=200, base="RF")
#
# x, y = read_data.read('letter recognition')
# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# SaveExpreminetSeries(x, y, "Letter", 0.98, n_estimators=200, base="RF")
#
# x, y = read_data.read('mice protein')
# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# SaveExpreminetSeries(x, y, "Protein", 0.88, n_estimators=200, base="RF")
#
# x, y = read_data.read('page blocks')
# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# SaveExpreminetSeries(x, y, "Page Blocks", 0.8, n_estimators=200, base="RF")

# x, y = read_data.read('diabetes')
# SaveBinaryExpreminetSeries(x, y, "Diabetes", 0.95, n_estimators=200, base="RF")
#
# x, y = read_data.read('adult')
# SaveBinaryExpreminetSeries(x, y, "Adult", 0.999, n_estimators=200, base="RF")

# SaveBinaryExpreminetSeries(x, y, "SVM Guide 1", 0.995, n_estimators=200, base="RF")
# SaveBinaryExpreminetSeries(x, y, "Phishing", 0.99, n_estimators=200, base="RF")
# partitions = [0.5, 0.7, 0.9, 0.95, 0.97]
# dbName = "SmallMNIST"
# ExperimentWithDifferentPartition(x, y, dbName, partitions, n_estimators=200, base="RF")











#
# totalTime = t1-t0
# print("Computational time:", totalTime)


if __name__ == '__main__':
    simple_test()



