import os.path
import numpy as np
import scipy.sparse
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
import pandas as pd
from src.parser import get_parser
from env.NSClassification import NSClassification
from env.NSMovieLens import NSMovieLens
from env.NSChain import NSRLEnv
from src.utils import prediction_subroutine
np.set_printoptions(suppress=True)


def regression(sample_feature, sample_feedback, sample_ratio, feature_mat, prob_mat):
    reg = RidgeCV(alphas=[0.01, 0.1, 0.5, 1.0, 10.0], fit_intercept=True).fit(sample_feature, sample_feedback, sample_ratio)
    proxy = np.dot(sample_feature, reg.coef_) + reg.intercept_
    proxy_mat = np.dot(feature_mat, reg.coef_) + reg.intercept_

    total_proxy = proxy_mat * prob_mat / prob_mat.sum()
    is_diff = np.mean(sample_ratio * (sample_feedback - proxy))
    mean_est = total_proxy.sum() + is_diff
    var_est = (np.square(sample_ratio * (sample_feedback - proxy)).sum() - n * is_diff ** 2) / (n * (n - 1))

    reg_error = np.sqrt(np.power(proxy - yi, 2).mean())
    return mean_est, var_est, reg_error


if __name__ == '__main__':
    """
    These are example scripts for running the experiment.
    To generate data: 
    python main.py --env_type movie --window_size 1 --sample_percentage 0.1 --num_run 5 --generate_data True --sample_data True
    To run experiment without re-generating data: 
    python main.py --env_type movie --window_size 1 --sample_percentage 0.1 --num_run 5 --generate_data False --sample_data False
    """
    parser = get_parser()
    args = parser.parse_args()

    np.random.seed(1687)
    num_round = args.num_round
    window_size = args.window_size

    if args.env_type == 'yeast':
        num_round = 50
        temp = 4.0
        env = NSClassification(args.env_type)
    elif args.env_type == 'youtube':
        num_round = 50
        temp = 4.0
        env = NSClassification(args.env_type)
    elif args.env_type == 'movie':
        num_round = 25
        temp = 4.0
        env = NSMovieLens()
    elif args.env_type == 'chain':
        num_round = 100
        temp = 2.0
        env = NSRLEnv(num_round)
    else:
        raise ValueError

    if args.generate_data:
        print('generate data.')
        env.generate_data(num_round, temp)

    if args.env_type in ['yeast', 'movie', 'youtube']:
        all_X = env.get_auxiliary()
        n = int(all_X.shape[0] * args.sample_percentage)  # max = 1995
        exp = 'cb'
    else:
        all_X = None
        n = args.n
        args.sample_percentage = n
        exp = 'rl'

    all_rows = []
    for run_id in range(args.num_run):
        print('Run {}. #round {}. # sample {}'.format(run_id, num_round, n))
        data_name = 'data/{}/ns_sample_data_{}_r{}.npy'.format(env.dir_name, args.sample_percentage, run_id)

        if args.generate_data or args.sample_data or not os.path.isfile(data_name):
            print('sample data.')
            env.sample_data(n, data_name, fix_seed=1567*(run_id+1))

        if exp == 'cb':
            # start evaluation
            context_ind_history = []
            context_history = []
            action_history = []
            feedback_history = []
            pbi_history = []
            pti_history = []

            data_list = np.load(data_name, allow_pickle=True)
            weights_all_round = {}

            for round_id in range(num_round):
                data_dict = data_list[round_id]
                num_data = data_dict['n']
                num_context = all_X.shape[0]
                num_action = data_dict['num_action']
                true_vpi = data_dict['vpi'].mean()

                print('round {}: Vpi={:.4f}'.format(round_id, true_vpi))
                new_row = {'run': run_id, 'round': round_id, 'estimator': 'vpi', 'estimation': true_vpi, 'std': 0, 'error': 0}
                all_rows.append(new_row)

                # build the reward prediction for sliding window
                context_idx = data_dict['context_idx']
                contexts = all_X[context_idx]
                actions = data_dict['actions']
                yi = data_dict['feedbacks']
                pbi = data_dict['pb']
                pti = np.array([data_dict['target_prob'][context_idx[i], actions[i]] for i in range(num_data)])
                target_prob = data_dict['target_prob']

                context_ind_history.append(context_idx)
                context_history.append(contexts)
                action_history.append(actions)
                feedback_history.append(yi)
                pbi_history.append(pbi)
                pti_history.append(pti)

                min_window_id = max(round_id-window_size, 0)
                max_window_id = round_id+1
                if scipy.sparse.issparse(contexts):
                    window_contexts = scipy.sparse.vstack(context_history[min_window_id:max_window_id])
                else:
                    window_contexts = np.vstack(context_history[min_window_id:max_window_id])

                window_context_ind = np.concatenate(context_ind_history[min_window_id:max_window_id])
                window_actions = np.concatenate(action_history[min_window_id:max_window_id])
                window_feedback = np.concatenate(feedback_history[min_window_id:max_window_id])
                window_pbi = np.concatenate(pbi_history[min_window_id:max_window_id])
                window_pti = np.concatenate(pti_history[min_window_id:max_window_id])

                # train weights for each action separately (for one prediction)
                weights = {}
                empirical_cov = {}
                empirical_error = {}
                for a in range(num_action):
                    a_index = np.where(window_actions == a)[0]
                    if len(a_index) > 0:
                        X = window_contexts[a_index]
                        Y = window_feedback[a_index]
                        reg = LinearRegression().fit(X, Y)
                        weights[a] = {'coef_': reg.coef_, 'intercept_': reg.intercept_}
                        if round_id > 0:
                            # for variance estimation of DM
                            # sklearn LinearRegression use min norm solution when d > n, which is the same as using pinv.
                            # see https://stackoverflow.com/questions/23714519/how-does-sklearn-do-linear-regression-when-p-n
                            if scipy.sparse.issparse(X):
                                empirical_cov[a] = scipy.sparse.linalg.pinv(X.T.dot(X))
                            else:
                                empirical_cov[a] = np.linalg.pinv(X.T.dot(X))

                            pred_Y = reg.predict(X)
                            empirical_error[a] = np.sum((pred_Y - Y) ** 2)
                    else:
                        weights[a] = {'coef_': np.zeros(contexts.shape[1]), 'intercept_': 0.0}

                # train weights for each action separately (for K prediction)
                single_run_weight = {}
                for a in range(num_action):
                    a_index = np.where(actions == a)[0]
                    if len(a_index) > 0:
                        X = contexts[a_index]
                        Y = yi[a_index]
                        reg = LinearRegression().fit(X, Y)
                        single_run_weight[a] = {'coef_': reg.coef_, 'intercept_': reg.intercept_}
                    else:
                        single_run_weight[a] = {'coef_': np.zeros(contexts.shape[1]), 'intercept_': 0.0}
                weights_all_round[round_id] = single_run_weight

                DM_Y = np.zeros((num_context, num_action))
                for a in range(num_action):
                    if a in weights.keys():
                        DM_Y[:, a] = all_X.dot(weights[a]['coef_']) + weights[a]['intercept_']
                DM_Y = DM_Y.clip(min=env.min_r, max=env.max_r)  # clip prediction to [min_r, max_r]

                if round_id == 0:
                    continue

                # Direct Method
                mb_vpi = (DM_Y * target_prob).sum() / DM_Y.shape[0]
                mb_var = 0
                sigma_hat = 0
                for a in range(num_action):
                    if a in empirical_cov.keys():
                        if scipy.sparse.issparse(all_X):
                            tx = target_prob[:, a] * all_X / all_X.shape[0]
                        else:
                            tx = np.matmul(np.expand_dims(target_prob[:, a], axis=0), all_X).flatten() / all_X.shape[0]

                        mb_var += np.dot(tx, empirical_cov[a].dot(tx.T))
                        sigma_hat += empirical_error[a]

                p = all_X.shape[1]
                sigma_hat = sigma_hat / (num_data - p)
                mb_var = mb_var * sigma_hat
                if mb_var < 0:
                    mb_var = 0
                new_row = {'run': run_id, 'round': round_id, 'estimator': 'DM', 'estimation': mb_vpi, 'std': np.sqrt(mb_var), 'error': mb_vpi - true_vpi}
                all_rows.append(new_row)

                # IS
                hat_vpi = np.mean(yi * pti / pbi)
                n = yi.shape[0]
                hat_var = (np.square(yi * pti / pbi).sum() - n * hat_vpi ** 2) / (n * (n - 1))
                new_row = {'run': run_id, 'round': round_id, 'estimator': 'IS', 'estimation': hat_vpi, 'std': np.sqrt(hat_var), 'error': hat_vpi - true_vpi}
                all_rows.append(new_row)

                # sliding window IS
                hat_vpi = np.mean(window_feedback * window_pti / window_pbi)
                n = window_feedback.shape[0]
                hat_var = (np.square(window_feedback * window_pti / window_pbi).sum() - n * hat_vpi ** 2) / (n * (n - 1))
                new_row = {'run': run_id, 'round': round_id, 'estimator': 'SW-IS', 'estimation': hat_vpi, 'std': np.sqrt(hat_var), 'error': hat_vpi - true_vpi}
                all_rows.append(new_row)

                # sliding window WIS
                sum_ratio = sum(window_pti / window_pbi)
                t_ratio = np.sum(window_feedback * window_pti / window_pbi / sum_ratio)
                t_tu = (window_feedback - t_ratio) * window_pti / window_pbi
                n = window_feedback.shape[0]
                v_ratio = (np.square(t_tu).sum() - t_tu.sum() ** 2 / n) / (n * (n - 1))
                new_row = {'run': run_id, 'round': round_id, 'estimator': 'SW-WIS', 'estimation': t_ratio, 'std': np.sqrt(v_ratio), 'error': t_ratio-true_vpi}
                all_rows.append(new_row)

                if window_size > 0:
                    # building the reward prediction without using the current data
                    min_window_id = max(round_id - window_size, 0)
                    max_window_id = round_id
                    if scipy.sparse.issparse(contexts):
                        window_contexts = scipy.sparse.vstack(context_history[min_window_id:max_window_id])
                    else:
                        window_contexts = np.vstack(context_history[min_window_id:max_window_id])

                    window_context_ind = np.concatenate(context_ind_history[min_window_id:max_window_id])
                    window_actions = np.concatenate(action_history[min_window_id:max_window_id])
                    window_feedback = np.concatenate(feedback_history[min_window_id:max_window_id])

                    # train weights for each action separately
                    weights = {}
                    for a in range(num_action):
                        a_index = np.where(window_actions == a)[0]
                        if len(a_index) > 0:
                            X = window_contexts[a_index]
                            Y = window_feedback[a_index]
                            reg = LinearRegression().fit(X, Y)
                            weights[a] = {'coef_': reg.coef_, 'intercept_': reg.intercept_}
                        else:
                            weights[a] = {'coef_': np.zeros(contexts.shape[1]), 'intercept_': 0.0}

                    proxy_Y = np.zeros((num_context, num_action))
                    DR_Y = np.zeros((contexts.shape[0], num_action))
                    partial_proxy_Y = np.zeros((len(window_context_ind), num_action))
                    for a in range(num_action):
                        if a in weights.keys():
                            proxy_Y[:, a] = all_X.dot(weights[a]['coef_']) + weights[a]['intercept_']
                            DR_Y[:, a] = contexts.dot(weights[a]['coef_']) + weights[a]['intercept_']
                            partial_proxy_Y[:, a] = all_X[window_context_ind].dot(weights[a]['coef_']) + weights[a]['intercept_']

                    proxy_Y = proxy_Y.clip(min=env.min_r, max=env.max_r)  # clip prediction to [min_r, max_r]
                    DR_Y = DR_Y.clip(min=env.min_r, max=env.max_r)
                    partial_proxy_Y = partial_proxy_Y.clip(min=env.min_r, max=env.max_r)

                    # Diff
                    proxy = np.array([proxy_Y[context_idx[i], actions[i]] for i in range(num_data)])  # data_dict['proxy']
                    # total_proxy = proxy_Y * data_dict['target_prob'] / data_dict['target_prob'].sum()
                    total_proxy = (proxy_Y * target_prob).sum() / proxy_Y.shape[0]
                    hat_diff = np.mean(pti / pbi * (yi - proxy))
                    diff_vpi = total_proxy + hat_diff
                    n = yi.shape[0]
                    diff_var = (np.square(pti / pbi * (yi - proxy)).sum() - n * hat_diff ** 2) / (n * (n - 1))
                    new_row = {'run': run_id, 'round': round_id, 'estimator': 'Diff', 'estimation': diff_vpi, 'std': np.sqrt(diff_var), 'error': diff_vpi-true_vpi}
                    all_rows.append(new_row)

                    # Reg
                    sample_weight = pti / pbi
                    reg = LinearRegression(fit_intercept=True).fit(np.expand_dims(proxy, axis=1), yi, sample_weight)
                    reg_proxy = reg.coef_ * proxy + reg.intercept_
                    reg_proxy_Y = reg.coef_ * proxy_Y + reg.intercept_
                    
                    total_reg_proxy = reg_proxy_Y * target_prob / target_prob.sum()
                    hat_reg = np.mean(pti / pbi * (yi - reg_proxy))
                    reg_vpi = total_reg_proxy.sum() + hat_reg

                    # Variance estimation for Reg
                    n = proxy.shape[0]
                    S_size = proxy_Y.shape[0]
                    tx = np.array([1, (proxy_Y * data_dict['target_prob']).sum()/S_size])
                    hat_tx = np.array([(sample_weight * np.ones(proxy.shape)).mean(), (sample_weight * proxy).mean()])

                    feature = np.stack([np.ones(proxy.shape), proxy])
                    weighted_feature = (np.sqrt(sample_weight) * feature).T
                    T1 = np.matmul(weighted_feature.T, weighted_feature) / n
                    T1_inv = np.linalg.inv(T1)
                    T2 = (sample_weight * yi * feature).mean(axis=1)
                    # beta_hat = np.matmul(T1_inv, T2)

                    g_sa = 1 + np.matmul(np.expand_dims(tx - hat_tx, axis=0), np.matmul(T1_inv, feature)).flatten()
                    hat_te = np.mean(pti / pbi * g_sa * (yi - reg_proxy))
                    reg_var = (np.square(pti / pbi * g_sa * (yi - reg_proxy)).sum() - n * hat_te ** 2) / (n * (n - 1))
                    # print(reg_vpi, np.sqrt(reg_var))
                    new_row = {'run': run_id, 'round': round_id, 'estimator': 'Reg', 'estimation': reg_vpi, 'std': np.sqrt(reg_var), 'error': reg_vpi-true_vpi}
                    all_rows.append(new_row)

                    # normal variance estimation
                    hat_te = np.mean(pti / pbi * (yi - reg_proxy))
                    reg_var = (np.square(pti / pbi * (yi - reg_proxy)).sum() - n * hat_te ** 2) / (n * (n - 1))
                    new_row = {'run': run_id, 'round': round_id, 'estimator': 'Reg-non-weighted', 'estimation': reg_vpi, 'std': np.sqrt(reg_var), 'error': reg_vpi-true_vpi}
                    all_rows.append(new_row)

                    # RegDR2: use the past data to estimate the first term
                    DR_Y = reg.coef_ * DR_Y + reg.intercept_
                    partial_proxy_Y = reg.coef_ * partial_proxy_Y + reg.intercept_

                    total_proxy = (partial_proxy_Y * target_prob[window_context_ind]).sum() / partial_proxy_Y.shape[0]
                    diff_vpi = total_proxy + hat_reg
                    new_row = {'run': run_id, 'round': round_id, 'estimator': 'RegDR2', 'estimation': diff_vpi,
                               'std': 0, 'error': diff_vpi - true_vpi}
                    all_rows.append(new_row)

                    # RegDR: use the same data to estimate the first term
                    total_proxy = (DR_Y * target_prob[context_idx]).sum() / DR_Y.shape[0]
                    diff_vpi = total_proxy + hat_reg
                    new_row = {'run': run_id, 'round': round_id, 'estimator': 'RegDR', 'estimation': diff_vpi, 'std': 0,
                               'error': diff_vpi - true_vpi}
                    all_rows.append(new_row)

                    # AR prediction
                    multiple_proxy_Y = np.zeros((num_context, num_action, max_window_id-min_window_id))
                    count = 0
                    for w_id in range(min_window_id, max_window_id):
                        for a in weights_all_round[w_id].keys():
                            multiple_proxy_Y[:, a, count] = all_X.dot(weights_all_round[w_id][a]['coef_']) + weights_all_round[w_id][a]['intercept_']
                        count += 1
                    multiple_proxy_Y = multiple_proxy_Y.clip(min=env.min_r, max=env.max_r)  # clip prediction to [min_r, max_r]
                    multiple_proxy = np.array([multiple_proxy_Y[context_idx[i], actions[i]] for i in range(num_data)])

                    sample_ratio = pti / pbi
                    reg_vpi, reg_var, ar_error = regression(sample_feature=multiple_proxy, sample_feedback=yi,
                                                            sample_ratio=sample_ratio, feature_mat=multiple_proxy_Y, prob_mat=target_prob)
                    new_row = {'run': run_id, 'round': round_id, 'estimator': 'Reg_AR', 'estimation': reg_vpi,
                               'std': np.sqrt(reg_var), 'error': reg_vpi - true_vpi}
                    all_rows.append(new_row)

                    # state-action feature only
                    context_Y = np.repeat(np.expand_dims(all_X, axis=1), num_action, axis=1)

                    reg_vpi, reg_var, ar_error3 = regression(sample_feature=contexts, sample_feedback=yi,
                                                             sample_ratio=sample_ratio, feature_mat=context_Y,
                                                             prob_mat=target_prob)
                    new_row = {'run': run_id, 'round': round_id, 'estimator': 'Reg_FeatureOnly', 'estimation': reg_vpi,
                               'std': np.sqrt(reg_var), 'error': reg_vpi - true_vpi}
                    all_rows.append(new_row)

                    # state-action feature with AR prediction
                    context_proxy = np.concatenate([multiple_proxy, contexts], axis=1)
                    context_proxy_Y = np.concatenate([multiple_proxy_Y, context_Y], axis=2)

                    reg_vpi, reg_var, ar_error4 = regression(sample_feature=context_proxy, sample_feedback=yi,
                                                             sample_ratio=sample_ratio, feature_mat=context_proxy_Y,
                                                             prob_mat=target_prob)
                    new_row = {'run': run_id, 'round': round_id, 'estimator': 'Reg_FeatureAR', 'estimation': reg_vpi,
                               'std': np.sqrt(reg_var), 'error': reg_vpi - true_vpi}
                    all_rows.append(new_row)

                    # state-action feature with one prediction
                    context_proxy = np.concatenate([np.expand_dims(proxy, axis=1), contexts], axis=1)
                    context_proxy_Y = np.concatenate([np.expand_dims(proxy_Y, axis=2), context_Y], axis=2)

                    reg_vpi, reg_var, ar_error5 = regression(sample_feature=context_proxy, sample_feedback=yi,
                                                             sample_ratio=sample_ratio, feature_mat=context_proxy_Y,
                                                             prob_mat=target_prob)
                    new_row = {'run': run_id, 'round': round_id, 'estimator': 'Reg_Feature', 'estimation': reg_vpi,
                               'std': np.sqrt(reg_var), 'error': reg_vpi - true_vpi}
                    all_rows.append(new_row)

        else:
            trajectory_history = []
            feedback_history = []
            pbi_history = []
            pti_history = []

            data_list = np.load(data_name, allow_pickle=True)
            num_round = len(data_list)
            for round_id in range(num_round):
                data_dict = data_list[round_id]
                target_policy = data_dict['target_policy']
                num_state = data_dict['num_state']
                num_action = data_dict['num_action']
                n = data_dict['n']
                true_vpi = data_dict['vpi']

                print('round {}: Vpi={:.4f}'.format(round_id, true_vpi))
                new_row = {'run': run_id, 'round': round_id, 'estimator': 'vpi', 'estimation': true_vpi, 'std': 0, 'error': 0}
                all_rows.append(new_row)

                # build the proxy values
                trajectories = data_dict['trajectory']
                yi = data_dict['feedbacks']
                pbi = data_dict['pb']
                pti = data_dict['pt']

                trajectory_history.extend(trajectories)
                feedback_history.append(yi)
                pbi_history.append(pbi)
                pti_history.append(pti)

                min_window_id = max(round_id - window_size, 0)
                max_window_id = round_id + 1
                window_feedback = np.concatenate(feedback_history[min_window_id:max_window_id])
                window_pbi = np.concatenate(pbi_history[min_window_id:max_window_id])
                window_pi = np.concatenate(pti_history[min_window_id:max_window_id])

                min_trajectory_id = max(round_id - window_size, 0) * n
                max_trajectory_id = n * (round_id + 1)
                print(min_trajectory_id, max_trajectory_id)

                window_trajectory = trajectory_history[min_trajectory_id:max_trajectory_id]
                window_transition = np.concatenate(window_trajectory)
                print(len(window_trajectory), window_transition.shape)

                # train with tabular FQE
                alpha = 0.1
                Q_values = np.zeros([num_state, num_action])
                # counts = np.zeros([num_state, num_action])
                for _ in range(100):
                    random_index = np.arange(len(window_trajectory))
                    np.random.shuffle(random_index)
                    for i in random_index:
                        tau = window_trajectory[i]
                        next_s = None
                        for h in range(tau.shape[0] - 1, 0 - 1, -1):
                            s = int(tau[h, 0])
                            a = int(tau[h, 1])
                            if next_s is None:
                                target = tau[h, 4]  # reward
                            else:
                                target = tau[h, 4] + (target_policy[next_s] * Q_values[next_s, :]).sum()
                            # counts[s, a] += 1
                            # Q_values[s, a] += (counts[s, a] - 1) / counts[s, a] * Q_values[s, a] + 1 / counts[s, a] * target
                            Q_values[s, a] += alpha * (target - Q_values[s, a])
                            next_s = s
                    alpha *= 0.99

                mb_values = (target_policy[0] * Q_values[0]).sum()

                if round_id == 0:
                    old_mb_values = mb_values
                    continue

                # Direct Method
                mb_vpi = mb_values
                new_row = {'run': run_id, 'round': round_id, 'estimator': 'DM', 'estimation': mb_vpi, 'std': 0,
                           'error': mb_vpi - true_vpi}
                all_rows.append(new_row)

                # IS
                hat_vpi = np.mean(yi * pti / pbi)
                n = yi.shape[0]
                hat_var = (np.square(yi * pti / pbi).sum() - n * hat_vpi ** 2) / (n * (n - 1))
                new_row = {'run': run_id, 'round': round_id, 'estimator': 'IS', 'estimation': hat_vpi, 'std': np.sqrt(hat_var),
                           'error': hat_vpi - true_vpi}
                all_rows.append(new_row)

                # sliding window IS
                hat_vpi = np.mean(window_feedback * window_pi / window_pbi)
                n = window_feedback.shape[0]
                hat_var = (np.square(window_feedback * window_pi / window_pbi).sum() - n * hat_vpi ** 2) / (n * (n - 1))
                new_row = {'run': run_id, 'round': round_id, 'estimator': 'SW-IS', 'estimation': hat_vpi, 'std': np.sqrt(hat_var),
                           'error': hat_vpi - true_vpi}
                all_rows.append(new_row)

                # TODO: sliding window WIS
                sum_ratio = sum(window_pi / window_pbi)
                t_ratio = np.sum(window_feedback * window_pi / window_pbi / sum_ratio)
                t_tu = (window_feedback - t_ratio) * window_pi / window_pbi
                n = window_feedback.shape[0]
                v_ratio = (np.square(t_tu).sum() - t_tu.sum() ** 2 / n) / (n * (n - 1))
                new_row = {'run': run_id, 'round': round_id, 'estimator': 'SW-WIS', 'estimation': t_ratio, 'std': np.sqrt(v_ratio), 'error': t_ratio-true_vpi}
                all_rows.append(new_row)

                # PDIS
                hat_vpi = 0
                for i in range(len(trajectories)):
                    tau = trajectories[i]
                    IS_weight = 1.0
                    for h in range(tau.shape[0]):
                        IS_weight *= tau[h, 3] / tau[h, 2]
                        hat_vpi += IS_weight * tau[h, 4]
                hat_vpi /= len(trajectories)
                hat_var = 0.0
                new_row = {'run': run_id, 'round': round_id, 'estimator': 'PDIS', 'estimation': hat_vpi,
                           'std': np.sqrt(hat_var),
                           'error': hat_vpi - true_vpi}
                all_rows.append(new_row)

                if window_size > 0:
                    # Diff
                    # total_proxy = proxy_Y * data_dict['target_prob'] / data_dict['target_prob'].sum()
                    proxy = old_mb_values
                    hat_diff = np.mean(pti / pbi * (yi - old_mb_values))
                    diff_vpi = old_mb_values + hat_diff
                    n = yi.shape[0]
                    diff_var = (np.square(pti / pbi * (yi - proxy)).sum() - n * hat_diff ** 2) / (n * (n - 1))
                    new_row = {'run': run_id, 'round': round_id, 'estimator': 'Diff', 'estimation': diff_vpi, 'std': np.sqrt(diff_var),
                               'error': diff_vpi - true_vpi}
                    all_rows.append(new_row)

                    # Reg
                    sample_weight = pti / pbi
                    proxy = np.repeat(proxy, sample_weight.shape[0])
                    reg = LinearRegression(fit_intercept=True).fit(np.expand_dims(proxy, axis=1), yi, sample_weight)
                    new_proxy = reg.coef_ * proxy + reg.intercept_
                    total_new_proxy = new_proxy[0]
                    hat_reg = np.mean(pti / pbi * (yi - new_proxy))
                    reg_vpi = total_new_proxy + hat_reg
                    
                    hat_te = np.mean(pti / pbi * (yi - new_proxy))
                    var_reg = (np.square(pti / pbi * (yi - new_proxy)).sum() - n * hat_te ** 2) / (n * (n - 1))

                    new_row = {'run': run_id, 'round': round_id, 'estimator': 'Reg', 'estimation': reg_vpi, 'std': np.sqrt(var_reg), 'error': reg_vpi-true_vpi}
                    all_rows.append(new_row)

                    # TODO: PDIS wit Reg

                    old_mb_values = mb_values

    df = pd.DataFrame(all_rows)
    df['sq_error'] = df['error'] ** 2
    df['RMSE'] = np.sqrt(df['sq_error'])
    df['covered'] = 1.96 * np.abs(df['std']) >= np.abs(df['error'])
    df['B'] = window_size

    df = prediction_subroutine(df, num_round)
    df.to_csv('results/{}_df_n{}_B{}.csv'.format(args.env_type, args.sample_percentage, window_size), index=False)
    print('save results to results/{}_df_n{}_B{}.csv'.format(args.env_type, args.sample_percentage, window_size))
