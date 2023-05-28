import joblib
import bentoml

with open("/userRepoData/taeefnajib/Detecting-Fake-Notes-Using-Gradient-Boosting-Regressor/__sidetrek__/models/16bbf171dafdf9874abe0dd338e42854.joblib", 'rb') as f:
    
    model = joblib.load(f)
    saved_model = bentoml.sklearn.save_model("fake_notes_model", model)
    print(saved_model) # This is required!