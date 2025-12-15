import statsmodels.api as sm
import os
from data_processing import build_training_data
import pandas as pd
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
if __name__ == "__main__":
    # Load Data
    training_data = build_training_data(BINARY_PATH, PROTEIN_PATHS, TARGET_CODE, sample_n=40000)
    # X[39257, 2941] y[39257, ] 
    X, y, feature_names = training_data
    Training_X, Test_X = X[:20000, :], X[20000:, :]
    Training_y, Test_y = y[:20000], y[20000:]
    
    # Calculate p-values and store results
    results = []
    for i in range(X.shape[1]):
        feature = Training_X[:, i]
        feature = sm.add_constant(feature)
        logit_model = sm.Logit(Training_y, feature)
        result = logit_model.fit(disp=0)
        p_value = result.pvalues[1]  # p-value for the feature (not the constant)
        results.append((i, feature_names[i], p_value))
        with open("P_values_I10_20000_Instances.txt", "a") as f:
            f.write(f"{i}\t{feature_names[i]}\t{p_value}\n")
    
    # Sort by p-value and save to a separate file
    results_sorted = sorted(results, key=lambda x: x[2])
    with open("P_values_I10_20000_Instances_sorted.txt", "w") as f:
        for i, feature_name, p_value in results_sorted:
            f.write(f"{i}\t{feature_name}\t{p_value}\n")

    
    # If your file is named "proteins.txt"
    df = pd.read_csv("P_values_I10_20000_Instances_sorted.txt", 
                    sep="\t", 
                    header=None, 
                    names=["index", "protein_info", "value"])
    
    np.save("Training_y.npy", Training_y)
    np.save("Test_y.npy", Test_y)
    
    idx_list = [200, 500, 1000]
    for idx in idx_list:
        selected_features = df["index"].iloc[:idx].to_numpy()
        Training_X_selected = Training_X[:, selected_features]
        Test_X_selected = Test_X[:, selected_features]
        np.save(f"Training_X_selected_I10_{idx}_20000_Patients.npy", Training_X_selected)
        np.save(f"Test_X_selected_I10_{idx}_20000_Patients.npy", Test_X_selected)