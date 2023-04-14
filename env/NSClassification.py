import numpy as np
import scipy.sparse
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.decomposition import PCA
from src.read_data import read_data
import os


class NSBaseEnv():
    def __init__(self):
        self.all_Ys = None
        self.target_prob = None
        self.all_X = None
        self.num_round = None
        self.dir_name = None

    def read_data(self):
        print('read data')
        self.all_Ys = np.load('data/{}/ns_Y.npy'.format(self.dir_name), allow_pickle=True)
        self.target_prob = np.load('data/{}/target_prob.npy'.format(self.dir_name), allow_pickle=True)
        self.all_X = self.get_auxiliary()

    def sample_data(self, sample_size, data_name, fix_seed=1567):
        if self.all_Ys is None or self.target_prob is None or self.all_X is None:
            self.read_data()

        num_round = len(self.all_Ys)
        target_prob = self.target_prob

        data_list = []
        for round_id in range(num_round):
            print('sample data round {}'.format(round_id))
            np.random.seed(fix_seed + round_id)
            new_Y = self.all_Ys[round_id]
            vpi = np.multiply(new_Y, target_prob).sum(axis=1)

            # sample a dataset
            # sample_size = new_Y.shape[0]
            num_action = new_Y.shape[1]
            sample_context_id = np.random.choice(new_Y.shape[0], size=sample_size)
            # sample_context_id = np.arange(new_Y.shape[0])
            sample_action = []
            sample_pb = []
            sample_feedbacks = []
            for c_id in sample_context_id:
                a, prob = self.random_policy(num_action)
                sample_action.append(a)
                sample_pb.append(prob)
                sample_feedbacks.append(new_Y[c_id, a])

            sample_action = np.array(sample_action)
            sample_pb = np.array(sample_pb)
            sample_feedbacks = np.array(sample_feedbacks)

            # sample a dataset without replace for context
            # sample_context_id = np.arange(new_Y.shape[0])
            # sample_prob = np.ones(new_Y.shape) / new_Y.shape[1]
            # c = sample_prob.cumsum(axis=1)
            # u = np.random.rand(len(c), 1)
            # sample_action = (u < c).argmax(axis=1)
            # sample_feedbacks = np.array([new_Y[i, sample_action[i]] for i in range(len(c))])
            # sample_pb = np.take_along_axis(sample_prob, np.expand_dims(sample_action, axis=1), axis=1).flatten()

            data_dict = {
                'round_id': round_id,
                'target_prob': target_prob,
                'vpi': vpi,
                'context_idx': sample_context_id,
                'actions': sample_action,
                'feedbacks': sample_feedbacks,
                'pb': sample_pb,
                'num_action': num_action,
                'n': sample_size
            }
            data_list.append(data_dict)

        # print('saving data/yeast/ns_data{}'.format(round_id))
        np.save(data_name, data_list)

    def num_round(self):
        return len(self.all_Ys)

    def get_auxiliary(self):
        if self.all_X is None:
            all_X = np.load('data/{}/population_X.npy'.format(self.dir_name), allow_pickle=True)
            if all_X.dtype == np.dtype('O'):
                self.all_X = all_X.item()
            else:
                self.all_X = all_X
            return self.all_X
        else:
            return self.all_X

    @staticmethod
    def random_policy(num_action):
        a = np.random.choice(num_action)
        prob = 1 / num_action
        return a, prob


class NSClassification(NSBaseEnv):
    def __init__(self, dir_name):
        super(NSClassification, self).__init__()
        self.dir_name = dir_name
        self.max_r = 1.0
        self.min_r = 0.0
        self.feature_dim = 32

    def generate_data(self, num_round, temp):
        if self.dir_name == 'yeast':
            trn_X, trn_Y, num_samples, num_feat, num_labels = read_data('data/yeast/yeast_train.svm')
            tst_X, tst_Y, _, _, _ = read_data('data/yeast/yeast_test.svm')
            all_X = scipy.sparse.vstack([trn_X, tst_X])
            all_X = all_X[:, 1:]  # remove the first zero column
            all_Y = scipy.sparse.vstack([trn_Y, tst_Y])
            num_trn = all_X.shape[0]//10
        else:
            all_X, all_Y, num_samples, num_feat, num_labels = read_data('data/youtube/youtube_node2vec.svm')
            all_X = all_X[:, 1:]  # remove the first zero column
            all_Y = all_Y[:, 1:]
            num_trn = all_X.shape[0] // 100

        trn_id = np.random.choice(all_X.shape[0], num_trn, replace=False)

        pre_Y = all_Y.toarray()
        all_Ys = []
        # all_Ps = []

        # train a policy on training set and compute the value of the policy on testing set
        clf = SVC(kernel='linear', C=1.0, probability=True)
        multi_clf = MultiOutputClassifier(clf, n_jobs=None)
        multi_clf.fit(all_X[trn_id], pre_Y[trn_id])
        output_prob = multi_clf.predict_proba(all_X)
        output_prob = np.stack([output_prob[i][:, 1] for i in range(len(output_prob))], axis=1)
        target_prob = np.exp(temp * output_prob) / np.sum(np.exp(temp * output_prob), axis=1, keepdims=True)

        # dimension reduction
        all_X = all_X.toarray()
        pca = PCA(n_components=self.feature_dim)
        pca.fit(all_X)
        all_X = pca.transform(all_X)

        # generate a sequence of reward functions
        num_pos = all_Y.data.shape[0]
        flip_prob = 0.05
        speed = 0.25
        mean_line = 0.5
        amplitude = np.random.uniform(0.1, 0.5, num_pos)
        frequency = np.random.uniform(0.1, 1.0, num_pos) * speed
        stds = np.random.rand(num_pos) * 0.01

        for round_id in range(num_round):
            np.random.seed(1687 + round_id)
            print('generate data round {}'.format(round_id))
            ns_data = mean_line + amplitude * np.sin((round_id + 1) * frequency) + np.random.randn() * stds
            print(ns_data)
            ns_data = ns_data.clip(min=0.0, max=1.0)
            pos_Y = scipy.sparse.csr_matrix((ns_data, all_Y.indices, all_Y.indptr)).toarray()
            additive_Y = np.random.uniform(0.0, 1.0, pos_Y.shape) * np.random.choice([1.0, 0.0], pos_Y.shape, p=[flip_prob, 1-flip_prob])

            # flip with 10% probability
            # binary_pre_Y = 2 * pre_Y - 1
            # flip_mask = np.random.choice([1.0, -1.0], pre_Y.shape, p=[1-flip_prob, flip_prob])
            # new_Y = (binary_pre_Y * flip_mask + 1) / 2

            # keep half of true positives
            # pos_var = all_Y.data.copy()
            # uniform_mask = np.random.uniform(0, 1, pos_var.shape[0]) >= (1-keep_prob)
            # uniform_mask = uniform_mask.astype(float)
            # new_pos_var = pos_var * uniform_mask
            # new_additive_Y = scipy.sparse.csr_matrix((new_pos_var, all_Y.indices, all_Y.indptr)).toarray()
            # new_Y = (new_Y + new_additive_Y).clip(min=0.0, max=1.0)

            new_Y = (pos_Y + additive_Y).clip(min=0.0, max=1.0)
            all_Ys.append(new_Y)

            # train a policy on training set and compute the value of the policy on testing set
            # clf = SVC(kernel='linear', C=1.0, probability=True)
            # multi_clf = MultiOutputClassifier(clf, n_jobs=None)
            # multi_clf.fit(trn_X, np.asarray(trn_Y[:num_trn].todense()))
            # multi_clf.fit(all_X[:num_trn], np.sign(pre_Y[:num_trn]))
            # multi_clf.fit(all_X[trn_id], np.sign(pre_Y[trn_id]))

            # compute the value of the policy on testing set
            # output_prob = multi_clf.predict_proba(all_X)
            # output_prob = np.stack([output_prob[i][:, 1] for i in range(len(output_prob))], axis=1)
            # target_prob = np.exp(temp * output_prob) / np.sum(np.exp(temp * output_prob), axis=1, keepdims=True)
            print(new_Y.mean(), (new_Y * target_prob).sum() / new_Y.shape[0])
            # all_Ps.append(target_prob)

            # pre_Y = new_Y

        np.save('data/{}/ns_Y'.format(self.dir_name), all_Ys)
        np.save('data/{}/target_prob'.format(self.dir_name), target_prob)
        np.save('data/{}/population_X'.format(self.dir_name), all_X)
        self.all_Ys = all_Ys
        self.target_prob = target_prob
        self.all_X = all_X


class SClassification():
    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.max_r = 1.0
        self.min_r = 0.0
        self.feature_dim = 32

    def generate_stationary_data(self, temp):
        cond = (os.path.isfile('data/{}/stationary_X.npy'.format(self.dir_name)) is False)
        cond = cond or (os.path.isfile('data/{}/stationary_Y.npy'.format(self.dir_name)) is False)
        cond = cond or (os.path.isfile('data/{}/stationary_target_prob.npy'.format(self.dir_name)) is False)
        cond = cond or (os.path.isfile('data/{}/train_X.npy'.format(self.dir_name)) is False)
        cond = cond or (os.path.isfile('data/{}/train_Y.npy'.format(self.dir_name)) is False)

        if cond:
            if self.dir_name == 'yeast':
                trn_X, trn_Y, num_samples, num_feat, num_labels = read_data('data/yeast/yeast_train.svm')
                tst_X, tst_Y, _, _, _ = read_data('data/yeast/yeast_test.svm')
                all_X = scipy.sparse.vstack([trn_X, tst_X])
                all_X = all_X[:, 1:]  # remove the first zero column
                all_Y = scipy.sparse.vstack([trn_Y, tst_Y])
                num_trn = all_X.shape[0] // 2
            else:
                all_X, all_Y, num_samples, num_feat, num_labels = read_data('data/youtube/youtube_node2vec.svm')

                random_index = np.arange(np.shape(all_X)[0])
                np.random.shuffle(random_index)

                all_X = all_X[random_index, 1:]  # remove the first zero column
                all_Y = all_Y[random_index, 1:]
                num_trn = all_X.shape[0] // 10

            trn_X, trn_Y = all_X[:num_trn], all_Y[:num_trn]
            tst_X, tst_Y = all_X[num_trn:], all_Y[num_trn:]
            # val_X, val_Y = all_X[2*num_trn:], all_Y[2*num_trn:]

            # train a policy on training set and compute the value of the policy on testing set
            clf = SVC(kernel='linear', C=1.0, probability=True)
            multi_clf = MultiOutputClassifier(clf, n_jobs=None)
            multi_clf.fit(trn_X, np.asarray(trn_Y.todense()))

            # keep 10% data for learning reward prediction
            output_prob = multi_clf.predict_proba(tst_X)
            output_prob = np.stack([output_prob[i][:, 1] for i in range(len(output_prob))], axis=1)
            target_prob = np.exp(temp * output_prob) / np.sum(np.exp(temp * output_prob), axis=1, keepdims=True)

            # dimension reduction
            X = tst_X
            pca = PCA(n_components=self.feature_dim)
            pca.fit(X)
            X = pca.transform(X)

            tst_X = X[:num_trn]
            val_X = X[num_trn:]

            np.save('data/{}/stationary_X'.format(self.dir_name), tst_X)
            np.save('data/{}/stationary_Y'.format(self.dir_name), tst_Y)
            np.save('data/{}/stationary_target_prob'.format(self.dir_name), target_prob)
            # np.save('data/{}/train_X'.format(self.dir_name), tst_X)
            # np.save('data/{}/train_Y'.format(self.dir_name), tst_Y)

    def sample_stationary_data(self, num_run, sample_size, fix_seed=1567):
        val_Y = np.load('data/{}/stationary_Y.npy'.format(self.dir_name), allow_pickle=True).item()
        target_prob = np.load('data/{}/stationary_target_prob.npy'.format(self.dir_name), allow_pickle=True)

        # sample a dataset
        all_data_dict = []
        for run_id in range(num_run):
            np.random.seed(fix_seed + run_id)

            num_action = val_Y.shape[1]
            sample_context_id = np.random.choice(val_Y.shape[0], size=sample_size)
            sample_action = []
            sample_pb = []
            sample_feedbacks = []
            for c_id in sample_context_id:
                a, prob = self.random_policy(num_action)
                sample_action.append(a)
                sample_pb.append(prob)
                sample_feedbacks.append(val_Y[c_id, a])

            sample_action = np.array(sample_action)
            sample_pb = np.array(sample_pb)
            sample_feedbacks = np.array(sample_feedbacks)

            vpi = np.multiply(val_Y.toarray(), target_prob).sum(axis=1)

            data_dict = {
                'run_id': run_id,
                'target_prob': target_prob,
                'vpi': vpi,
                'context_idx': sample_context_id,
                'actions': sample_action,
                'feedbacks': sample_feedbacks,
                'pb': sample_pb,
                'num_action': num_action,
                'n': sample_size
            }
            all_data_dict.append(data_dict)

        print('saving data/{}/stationary_data'.format(self.dir_name))
        np.save('data/{}/stationary_data'.format(self.dir_name), all_data_dict)

    def get_auxiliary(self):
        return np.load('data/{}/stationary_X.npy'.format(self.dir_name), allow_pickle=True)

    def get_train(self):
        train_X = np.load('data/{}/train_X.npy'.format(self.dir_name), allow_pickle=True)
        train_Y = np.load('data/{}/train_Y.npy'.format(self.dir_name), allow_pickle=True).item()
        return train_X, train_Y

    @staticmethod
    def random_policy(num_action):
        a = np.random.choice(num_action)
        prob = 1 / num_action
        return a, prob