import pandas as pd
import numpy as np
import os
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# 1. Load the dataset
data_path = "long_data.csv"  
df = pd.read_csv(data_path)



# 2. Define features and target

feature_columns = ['ticker', 'sentiment', 'impact_num', 'text_length',
                   'avg_tfidf', 'embedding_mean']
target_column = 'target_return'

            

X = df[feature_columns]
y = df[target_column]

# 3. Preprocessing: 
# - For categorical columns ('ticker'), apply OneHotEncoder.
# - For numerical columns, apply StandardScaler.
categorical_features = ['ticker']
numerical_features = [col for col in feature_columns if col not in categorical_features]

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# 4. Build the modeling pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
])

# 5. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Define a parameter grid for tuning the XGBoost regressor (optional)
param_grid = {
    'regressor__n_estimators': [50, 100, 150],
    'regressor__max_depth': [3, 5, 7],
    'regressor__learning_rate': [0.05, 0.1, 0.2],
    'regressor__subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("Best parameters found:")
print(grid_search.best_params_)
print("Best cross-validation R^2: {:.3f}".format(grid_search.best_score_))

# 7. Evaluate on the test set
best_model = grid_search.best_estimator_
test_r2 = best_model.score(X_test, y_test)
print("Test set R^2: {:.3f}".format(test_r2))

# 8. Save the optimized model pipeline
model_save_path = "../../models/model_xgb_int.pkl"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
with open(model_save_path, 'wb') as f:
    pickle.dump(best_model, f)

# 9. Optional: Plot feature importances from XGBoost
# To plot feature importances, we need to get the underlying regressor
xgb_model = best_model.named_steps['regressor']
importances = xgb_model.feature_importances_

# Extract the feature names from the ColumnTransformer:
cat_encoder = best_model.named_steps['preprocessor'].named_transformers_['cat']
if cat_encoder is not None:
    cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
else:
    cat_feature_names = []
feature_names = cat_feature_names + numerical_features

plt.figure(figsize=(12, 6))
plt.bar(feature_names, importances, color='steelblue')
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("XGBoost Feature Importances")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
