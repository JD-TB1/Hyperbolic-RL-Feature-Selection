from data_processing import build_training_data
from rl_trainer import dqn_train, FSConfig, FeatureSelectionEnv
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import os


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

TRAINING_MAX_STEP = 10_000_000
EPISODE_MAX_STEP = 20
os.makedirs(RESULT_DIR, exist_ok=True)

def acc_reward_fn(selected_features, features, labels):
    if not selected_features:
        return -1.0
    X = features[:, selected_features]
    y = labels
    
    clf = SVC(kernel='rbf', C=1.0)

    clf.fit(X, y)

    preds = clf.predict(X)
    
    acc = accuracy_score(y, preds)

    return acc  # Reward: accuracy score

if __name__ == "__main__":
    # Load Data
    training_data = build_training_data(BINARY_PATH, PROTEIN_PATHS, TARGET_CODE, sample_n=3000)
    X, y, feature_names = training_data
    n_feat = len(feature_names) 
    
    # Initialize Environment and Config
    init_reward_fn = lambda selected: acc_reward_fn(selected, X, y)
    cfg = FSConfig(n_feat=n_feat, max_steps=TRAINING_MAX_STEP)
    env = FeatureSelectionEnv(n_feat=n_feat, max_steps=EPISODE_MAX_STEP, reward_fn=init_reward_fn, lam=0.0)
    model = dqn_train(env, cfg, max_total_steps=TRAINING_MAX_STEP, seed=42)