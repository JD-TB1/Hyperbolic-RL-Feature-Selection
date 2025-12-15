from data_processing import build_training_data
from rl_trainer import dqn_train, FSConfig, FeatureSelectionEnv
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import os
import numpy as np


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
DATA_DIR = "./data"
BINARY_PATH = os.path.join(DATA_DIR, "binary_csv.gz")
PROTEIN_PATHS = [
    os.path.join(DATA_DIR, x)
    for x in ["xaa.gz","xab.gz","xac.gz","xad.gz","xae.gz",
              "xaf.gz","xag.gz","xah.gz","xai.gz","xaj.gz","xaa_2.gz"]
]
TARGET_CODE = "I10"  
RESULT_DIR = "./results/rl_feature_selector_I10_single"

TRAINING_MAX_STEP = 1_000_000
EPISODE_MAX_STEP = 50
os.makedirs(RESULT_DIR, exist_ok=True)

acc_reward_cache = {}

def acc_reward_fn(selected_features, features, labels):
    if not selected_features:
        return -1.0
    
    key = tuple(sorted(selected_features))
    if key in acc_reward_cache:
        return acc_reward_cache[key]
    
    X_train = features[:15_000, selected_features]
    y_train = labels[:15_000]
    
    X_test = features[15_000:, selected_features]
    y_test = labels[15_000:]
    
    clf = SVC(kernel='rbf', C=1.0)

    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    acc_reward_cache[key] = acc
    return acc  # Reward: accuracy score

if __name__ == "__main__":
    # Load Data
    # training_data = build_training_data(BINARY_PATH, PROTEIN_PATHS, TARGET_CODE, sample_n=20_000)
    # X, y, feature_names = training_data
    checkpoint_name = "Standard_SVC_NoF_50_Training_15000_Test_5000_SelectFeatures_1000_Exploration_0.1_BatchSize_2048"
    print("Saving results to:", checkpoint_name)
    
    X = np.load("./I10_data_20000_instances/Training_X_selected_I10_1000_20000_Patients.npy")
    y = np.load("./I10_data_20000_instances/Training_y.npy")
    n_feat = X.shape[1]
    
    # Initialize Environment and Config
    init_reward_fn = lambda selected: acc_reward_fn(selected, X, y)
    cfg = FSConfig(n_feat=n_feat, max_steps=TRAINING_MAX_STEP)
    cfg.lr = 1e-4
    cfg.eps_end = 0.1 # 0.2 0.3 0.05
    cfg.eps_decay = 10000
    cfg.batch_size = 2048
    cfg.buffer_size = 500_000
    cfg.target_tau = 0.005 # 0.005| 0.01 0.05 0.1 0.001
    env = FeatureSelectionEnv(n_feat=n_feat, max_steps=EPISODE_MAX_STEP, reward_fn=init_reward_fn, lam=0.0)
    model = dqn_train(env, cfg, max_total_steps=TRAINING_MAX_STEP, seed=42, other_checkpoint_info=checkpoint_name)
    
"""

@dataclass
class FSConfig:
    n_feat: int
    max_steps: int
    lam: float = 0.0 # feature-count penalty in terminal reward
    gamma: float = 1.0 # episodic tasks typically use gamma=1.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 50_000 # steps for epsilon exponential decay
    lr: float = 1e-3
    batch_size: int = 128
    buffer_size: int = 50_000
    target_tau: float = 0.005 # hard update when 1.0; else soft update (Polyak)
    target_update_every: int = 1000 # steps between hard updates if tau==1
    train_start: int = 5_000 # fill buffer before training
    train_freq: int = 1
    hidden: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_save_freq: int = 1000  # episodes between saving model checkpoints
    evaluate_freq: int = 10  # episodes between evaluations
    

"""