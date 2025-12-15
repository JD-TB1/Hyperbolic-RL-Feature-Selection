from data_processing import build_training_data
from rl_trainer import QNetwork, FeatureSelectionEnv
import torch
import numpy as np
import os
import random

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
DATA_DIR = "./data"
BINARY_PATH = os.path.join(DATA_DIR, "binary_csv.gz")
PROTEIN_PATHS = [
    os.path.join(DATA_DIR, x)
    for x in ["xaa.gz","xab.gz","xac.gz","xad.gz","xae.gz",
              "xaf.gz","xag.gz","xah.gz","xai.gz","xaj.gz","xaa_2.gz"]
]
TARGET_CODE = "I10"
def DummyReward(selected_features):
    return 0.0

def get_model_selected_features(model_file_path, env):
    dkpt = torch.load(model_file_path, map_location=torch.device('cpu'), weights_only=False)
    obs_dim = env.n_feat + 1
    n_actions = env.n_actions
    q_network = QNetwork(obs_dim=obs_dim, n_actions=n_actions, hidden=512)
    q_network.load_state_dict(dkpt["model_state_dict"])
    
    s = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            qs = q_network(torch.as_tensor(s, device=torch.device("cpu")).unsqueeze(0))[0]
            # Mask illegal actions by setting them to -inf
            legal_mask = env.legal_actions_mask()
            q_values = qs.clone()
            illegal = torch.as_tensor(~legal_mask, device=torch.device("cpu"))
            q_values[illegal] = -1e9
            a = int(torch.argmax(q_values).item())
            # a = torch.softmax(q_values*0.8, dim=0).multinomial(num_samples=1).item()
            
            
        s2, reward, done, _ = env.step(a)
        s = s2
    return env.selected

# def train_scipy_model(X, y):
#     clf = SVC(kernel='rbf', probability=True)

#     clf.fit(X, y)

#     return clf

# def get_roc_data(y_true, y_scores):
#     from sklearn.metrics import roc_curve, auc
#     fpr, tpr, thresholds = roc_curve(y_true, y_scores)
#     roc_auc = auc(fpr, tpr)
#     return fpr, tpr, roc_auc

if __name__ == "__main__":
    X_training = np.load("./I10_DATA_20000_Instance/Training_X_selected_I10_1000_20000_Patients.npy")
    y_training = np.load("./I10_DATA_20000_Instance/Training_y.npy")
    X_test = np.load("./I10_DATA_20000_Instance/Test_X_selected_I10_1000_20000_Patients.npy")
    y_test = np.load("./I10_DATA_20000_Instance/Test_y.npy")
    
    num_feat = X_training.shape[1]
    env = FeatureSelectionEnv(num_feat, max_steps=100, reward_fn=DummyReward, lam=0.0)
    # selected_features = get_model_selected_features("./best_model/1000_best_model.pth", env)
    
    selected_features = random.sample(range(num_feat), 100)
    
    X_training = X_training[:, selected_features]
    X_test = X_test[:, selected_features]
    
    
    clf = SVC(kernel='rbf', probability=True)  # or use decision_function
    clf.fit(X_training, y_training)

    # 5. Get predicted scores
    # X_test = np.load("test.npy")
    # y_test = np.load("test_target.npy")
    y_score = clf.predict_proba(X_test)[:, 1]  # use clf.decision_function(X_test) if not using probability=True
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    # 6. Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # 7. Plot ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - SVM")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()