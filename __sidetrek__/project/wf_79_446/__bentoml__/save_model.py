import joblib
import bentoml
with open("/userRepoData/taeefnajib/Detecting-Fake-Notes-Using-Gradient-Boosting-Regressor/__sidetrek__/models/cbe9da81743a36eb39b78e08e22b9679.joblib", 'rb') as f:
    model = joblib.load(f)
    saved_model = bentoml.sklearn.save_model(
        "fake_note_3",
        model,
    )
    print(saved_model) # This is required!
