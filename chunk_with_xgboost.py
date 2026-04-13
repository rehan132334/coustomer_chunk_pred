import pandas as pd
import plotly.express as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def engineer_features(frame):
    frame = frame.copy()
    frame['charges_per_tenure'] = frame['TotalCharges'] / (frame['tenure'] + 1)
    frame['monthly_to_total_ratio'] = frame['MonthlyCharges'] / (frame['TotalCharges'] + 1)

    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']
    frame['service_count'] = frame[services].apply(lambda row: (row == 'Yes').sum(), axis=1)

    frame['is_month_to_month'] = (frame['Contract'] == 'Month-to-month').astype(int)
    frame['is_auto_pay'] = frame['PaymentMethod'].isin(
        ['Bank transfer (automatic)', 'Credit card (automatic)']).astype(int)

    bins = [0, 12, 24, 48, 72]
    labels = [0, 1, 2, 3]
    frame['tenure_group'] = pd.cut(frame['tenure'], bins=bins, labels=labels).astype(int)
    return frame


#reading the coustomer chunk file
df=pd.read_csv(r"C:\Users\Rehan Ahmed\Downloads\train.csv")
#drop the null values at the chunk
df=df.dropna(subset=['Churn'])
df=df.set_index('id')
#----------------------------------------feature engineering---------------------------------------------------------------------------------    
df = engineer_features(df)



#convert churn to numeric so it can be included in correlation
df['Churn_num'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype('int8')

#build encoded frame only for correlation visualization
cate=['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod']
numeric_source_cols = df.select_dtypes(include=[np.number]).drop(columns=['Churn_num']).columns.tolist()
neumerical=df[numeric_source_cols].reset_index(drop=True)
one=OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

standar=StandardScaler()
neumerical=standar.fit_transform(neumerical)
encoded_cate=one.fit_transform(df[cate])

df_numeric=pd.DataFrame(neumerical, columns=[f'num_{col}' for col in numeric_source_cols])
df_encoded_cate=pd.DataFrame(encoded_cate, columns=one.get_feature_names_out(cate))
df_mixed=pd.concat([df_numeric, df_encoded_cate.reset_index(drop=True)], axis=1)

#---------------------------------- train test split---------------------------------------------------------------------
X = df_mixed # drop original and numeric churn columns
y = df['Churn_num']
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


print("\nRunning Grid Search for XGBoost...")
#2. XGBoost
xg=XGBClassifier(random_state=42, n_estimators=100, max_depth=5, min_samples_split=5, min_samples_leaf=2, max_features='log2', learning_rate=0.9)
print("\nFitting XGBoost with best parameters...")
xg.fit(X_train, y_train)
y_pred_xg = xg.predict(X_test)
y_pred_xg_train = xg.predict(X_train)
print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xg))
accuracy=xg.score(X_test, y_test)
print(f"XGBoost Accuracy: {accuracy:.4f}")

accuracy_2=xg.score(X_train, y_train)
print(f"XGBoost Train Accuracy: {accuracy_2:.4f}")

#---------------------------------------loading test datra---------------------------------------------------------

test_data=pd.read_csv(r"C:\Users\Rehan Ahmed\Downloads\test (1).csv")
test_data = engineer_features(test_data)

numeric_test_values = standar.transform(test_data[numeric_source_cols])
encoded_test_cate=one.transform(test_data[cate])
df_numeric_test=pd.DataFrame(numeric_test_values, columns=[f'num_{col}' for col in numeric_source_cols])
df_encoded_test_cate=pd.DataFrame(encoded_test_cate, columns=one.get_feature_names_out(cate))
df_mixed_test=pd.concat([df_numeric_test.reset_index(drop=True), df_encoded_test_cate.reset_index(drop=True)], axis=1)

test_pridict=xg.predict(df_mixed_test)
test_pridict_proba=xg.predict_proba(df_mixed_test)[:,1]
#-------------------------------creating submission file---------------------------------------------------------
submission=pd.read_csv(r"C:\Users\Rehan Ahmed\Downloads\sample_submission.csv")
submission['Churn'] = test_pridict_proba
submission.to_csv(r"C:\Users\Rehan Ahmed\Downloads\submission.csv", index=False)
