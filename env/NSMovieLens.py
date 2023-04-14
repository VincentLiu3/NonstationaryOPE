import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import NMF
from sklearn.multioutput import MultiOutputClassifier
from env.NSClassification import NSBaseEnv


class NSMovieLens(NSBaseEnv):
    def __init__(self):
        super(NSMovieLens, self).__init__()
        self.dir_name = 'ml-25m'
        self.num_round = 25
        self.num_day_per_round = 60
        self.max_r = 5.0
        self.min_r = 0.0
        self.feature_dim = 32

    def read_data(self):
        print('read data')

        self.all_Ys = np.load('data/{}/ns_data_Y_{}_{}.npy'.format(self.dir_name, self.num_round, self.num_day_per_round), allow_pickle=True)
        self.target_prob = np.load('data/{}/target_prob.npy'.format(self.dir_name), allow_pickle=True)
        self.all_X = np.load('data/{}/population_X.npy'.format(self.dir_name), allow_pickle=True)

    def generate_data(self, num_round, temp):
        self.all_Ys = np.load('data/{}/ns_data_Y_{}_{}.npy'.format(self.dir_name, self.num_round, self.num_day_per_round), allow_pickle=True)

        average_Y = np.zeros(self.all_Ys[0].shape)
        for Ys in self.all_Ys:
            average_Y += Ys
        average_Y /= len(self.all_Ys)

        nmf = NMF(n_components=self.feature_dim, max_iter=1000)
        X = nmf.fit_transform(average_Y)
        # H = nmf.components_
        # nR = np.dot(X, H)
        Y = (average_Y >= 2.5).astype(np.float32)

        # train a policy on training set and compute the value of the policy on testing set
        clf = SVC(kernel='linear', C=1.0, probability=True)
        multi_clf = MultiOutputClassifier(clf, n_jobs=None)
        multi_clf.fit(X, np.asarray(Y))

        # compute the value of the policy on testing set
        output_prob = multi_clf.predict_proba(X)
        output_prob = np.stack([output_prob[i][:, 1] for i in range(len(output_prob))], axis=1)
        self.target_prob = np.exp(temp * output_prob) / np.sum(np.exp(temp * output_prob), axis=1, keepdims=True)
        self.all_X = X
        np.save('data/{}/target_prob'.format(self.dir_name), self.target_prob)
        np.save('data/{}/population_X'.format(self.dir_name), self.all_X)
