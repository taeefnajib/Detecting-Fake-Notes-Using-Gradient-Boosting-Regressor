import joblib
import bentoml

with open("/userRepoData/taeefnajib/Detecting-Fake-Notes-Using-Gradient-Boosting-Regressor/sidetrek/models/68cab49fa93f23e154b27ac7da89fec5.joblib", "rb") as f:
    model = joblib.load(f)
    saved_model = bentoml.sklearn.save_model("example_model", model)
    print(saved_model) # This is required!

