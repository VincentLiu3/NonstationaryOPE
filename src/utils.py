import numpy as np


class BasesValueFunction:
    def __init__(self, order, type, max_feature_value):
        """
        order: # of bases, each function also has one more constant parameter (called bias in machine learning)
        type: polynomial bases or Fourier bases
        """
        POLYNOMIAL_BASES = 0
        FOURIER_BASES = 1

        self.order = order
        self.weights = np.zeros(order + 1)
        self.max_feature_value = max_feature_value

        # set up bases function
        self.bases = []
        if type == POLYNOMIAL_BASES:
            for i in range(0, order):
                self.bases.append(lambda s, i=i: pow(s, i))
        elif type == FOURIER_BASES:
            for i in range(0, order):
                self.bases.append(lambda s, i=i: np.cos(i * np.pi * s))

    # get the value of @state
    def phi(self, state):
        # map the state space into [0, 1]
        state /= self.max_feature_value
        # get the feature vector
        feature = np.asarray([func(state) for func in self.bases])
        return feature


def prediction_subroutine(df, num_round):
    """
    Predicting future using BasesValueFunction
    """
    # add two variables in df
    df['future_pred'] = 0
    df['future'] = 0

    num_run = int(df['run'].max() + 1)
    num_feat = 5
    start_round = num_feat  # max(num_round//5, 5)
    for run_id in range(num_run):
        single_run_df = df[df['run'] == run_id]
        for max_h in range(start_round, num_round - 1):
            for est in list(np.unique(single_run_df['estimator'])):
                prediction_df = single_run_df[(single_run_df['estimator'] == est) & (single_run_df['round'] <= max_h) & (single_run_df['round'] > 0)].sort_values('round')

                Basis = BasesValueFunction(num_feat, 1, max_h + 1)
                X = np.zeros([max_h, num_feat])
                for h in list(prediction_df['round']):
                    X[h - 1] = Basis.phi(h)
                Y = np.asarray(prediction_df['estimation'])

                hat_w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), Y)

                df.loc[(df['run'] == run_id) & (df['estimator'] == est) & (df['round'] == max_h), 'future_pred'] = np.dot(Basis.phi(max_h + 1), hat_w)
                df.loc[(df['run'] == run_id) & (df['estimator'] == est) & (df['round'] == max_h), 'future'] = \
                    float(df.loc[(df['run'] == run_id) & (df['estimator'] == 'vpi') & (df['round'] == max_h + 1), 'estimation'])
                    # float(single_run_df.loc[(single_run_df['estimator'] == 'vpi') & (single_run_df['round'] == max_h + 1), 'estimation'])

    df['pred_error'] = (df['future_pred'] - df['future']) ** 2
    new_df = df[(df['round'] > start_round) & (df['round'] < num_round - 1)]
    print(new_df[['estimator', 'pred_error']].groupby('estimator').mean())
    return df