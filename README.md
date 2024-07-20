# Brain Stroke Prediction
A stroke, also known as brain attack, is a medical emergency where blood supply to a part of the brain is blocked or or when a blood vessel in the brain bursts which can lead to brain death. Timely diagnosis of brain strokes are crucial for effective treatment. 
The problem revolves around classifying stroke cases accurately which allows for effective treatment decisions. Traditional methods for stroke classification rely heavily on expert interpretation of symptoms and neuroimaging results which may be prone to error. Hence, robust and efficient systems based on machine learning can help in accurate diagnosis. 

# Dataset details
The dataset can be found on [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

The dataset contains the following features:
<ol>
  <li>id</li>
  <li>gender</li>
  <li>age</li>
  <li>hypertension</li>
  <li>heart_disease</li>
  <li>ever_married</li>
  <li>work_type</li>
  <li>Residence_type</li>
  <li>avg_glucose_level</li>
  <li>bmi</li>
  <li>smoking_status</li>
  <li>stroke</li>
</ol>

# Data Cleaning
The dataset consists of missing values in the ‘bmi’ column. The missing values were filled with the mean of ‘bmi’ column values. Furthermore, the ‘id’ column was dropped from the dataframe as it possesses no value for our study.
<br>
*Remove irrelevant columns for further processing*
<br>
`df = df.drop(columns=['id'])`
<br>
*Check missing values*
<br>
`print(df.isna().any())`
<br>
*Fill missing values with mean*
<br>
`df['bmi'] = df['bmi'].fillna(df['bmi'].mean())`

# Data Integration
For our study, we only utilize the one dataset mentioned in the above sections as there was no need for integration of datasets. Furthermore, quality datasets that could be used for integration were not publicly available.

# Data Transformation
The dataset being used has columns with categorical values which need to be transformed to be more digestible for machine learning algorithms. Additionally, scaling of features may help in smoother computations by machine learning models. Furthermore, the dataset in consideration is imbalanced (i.e. the class label to be predicted has an uneven distribution). Therefore, the dataset needs to be balanced for appropriate classification. For this task, we used the SMOTE technique to balance the dataset by oversampling minority classes on the basis of the K-Nearest Neighbours algorithm.

```
# Label Encoding - Convert categorical values to integers for computation
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoder = LabelEncoder()
for col in categorical_cols:
   df[col] = label_encoder.fit_transform(df[col])


Data Normalization
scaler1 = StandardScaler() # Z-Score Normalization
X_train_scaled_1 = scaler1.fit_transform(X_train_new)
X_test_scaled_1 = scaler1.transform(X_test_new)


Balance dataset using SMOTE (Synthetic Minority Oversampling Technique) approach
print("Before (label '1'): {}".format(sum(y_train == 1)))
print("Before (label '0'): {} \n".format(sum(y_train == 0)))
smo = SMOTE()
X_res, y_res = smo.fit_resample(X_train, y_train.ravel())
print("After (label '1'):", sum(y_res == 1))
print("After (label '0'):", sum(y_res == 0))
```

# Data Mining
In our study, we have implemented the Ensemble Learning technique of bagging and boosting classifiers to reduce the variance and bias respectively. The algorithms used for bagging are Random Forest and Bagging Classifier (with Decision Trees as base estimator), while the algorithms used for boosting are AdaBoost and XGBoost. Furthermore, we also implemented a simpler machine learning model known as K-Nearest Neighbours algorithm for comparison with Ensemble Learning.

```
# Random Forest
RF = RandomForestClassifier()
RF.fit(X_train_scaled_1, y_train_new)
y_pred_RF = RF.predict(X_test_scaled_1)


# BaggingClassifier
BC = BaggingClassifier()
BC.fit(X_train_scaled_1, y_train_new)
y_pred_BC = BC.predict(X_test_scaled_1)


# Adaboost
ADA = AdaBoostClassifier()
ADA.fit(X_train_scaled_1, y_train_new)
y_pred_ADA = ADA.predict(X_test_scaled_1)


# XGBoost
XGB = xgb.XGBClassifier()
XGB.fit(X_train, y_train)
y_pred_XGB = XGB.predict(X_test_scaled_1)


# K-Nearest Neighbors
KNN = KNeighborsClassifier()
KNN.fit(X_train_scaled_1, y_train_new)
y_pred_KNN = KNN.predict(X_test_scaled_1)
```

# Optimization of Models
The models mentioned above were not initialized with a set of hyper-parameters. Instead, we applied Grid Search to find the most optimal combination of hyper-parameters for each model.

```
def perform_grid_search(model, param_grid, X_train, y_train, X_test, y_test):
 GS = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
 GS.fit(X_train, y_train)
 print("Best Parameters:", GS.best_params_)
 best_model = GS.best_estimator_
 evaluate_model(best_model, X_test, y_test)

rf_param_grid = {
   'n_estimators': [50, 100, 200],
   'max_depth': [None, 10, 20],
   'min_samples_split': [2, 5, 10],
   'min_samples_leaf': [1, 2, 4],
   'max_features': ['auto', 'sqrt']
}

bagging_param_grid = {
   'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)],
   'n_estimators': [50, 100, 200],
   'max_samples': [0.5, 0.8, 1.0],
   'max_features': [0.5, 0.8, 1.0],
   'bootstrap': [True, False],
   'bootstrap_features': [True, False]
}

ada_param_grid = {
   'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)],
   'n_estimators': [50, 100, 200],
   'learning_rate': [0.1, 0.5, 1.0]
}

xgb_param_grid = {
   'learning_rate': [0.01, 0.1, 0.2],
   'max_depth': [3, 4, 5],
   'n_estimators': [100, 200, 300]
}

knn_param_grid = {
   'n_neighbors': [3, 5, 10],
   'weights': ['uniform', 'distance'],
   'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}
```


# Accuracy Score of each model
<ul>
  <li>Random Forest: 0.95</li>
  <li>Bagging Classifier: 0.80</li>
  <li>AdaBoost: 0.95</li>
  <li>XGBoost: 0.94</li>
  <li>KNN: 0.89</li>
</ul>

