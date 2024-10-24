import joblib
import numpy as np

X_classify2 = joblib.load("X_classify_input_noisy_fixed1.pkl")
X_classify3 = joblib.load("X_classify_input_noisy_varied1.pkl")
X_classify_input = np.concatenate((X_classify2,X_classify3), axis=0)

joblib.dump(X_classify_input, "X_classify_input.pkl")
