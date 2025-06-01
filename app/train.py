import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from app.utils import preprocess_data
import yaml

# Load config
with open("app/config.yaml") as f:
    config = yaml.safe_load(f)

df = pd.read_csv(config["data_path"])
X, y = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(**config["model_params"])
model.fit(X_train, y_train)

print("Training accuracy:", model.score(X_train, y_train))
joblib.dump(model, config["model_output_path"])
