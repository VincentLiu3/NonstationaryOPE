## Asymptotically Unbiased Off-Policy Policy Evaluation when Reusing Old Data in Nonstationary Environments

Code for "Asymptotically Unbiased Off-Policy Policy Evaluation when Reusing Old Data in Nonstationary Environments" (https://arxiv.org/abs/2302.11725).

The Youtube dataset can be download from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html, and the MovieLens dataset can be downloaded from https://grouplens.org/datasets/movielens/25m/. The script `read_movie.py` is used to build the non-stationary rating function from the original MovieLens dataset.

Example for running the experiments:  
`
python main.py --env_type youtube --sample_percentage 0.1 --window_size 1
`  
where `env_type` can be youtube, movie or chain (for RL experiment). 

