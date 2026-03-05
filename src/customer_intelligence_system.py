import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
data=pd.read_csv("telco_datset.csv")
df=pd.DataFrame(data)
print(df.columns)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.drop(columns=['customerID']).reset_index(drop=True)
target_col='Churn'
df = df.dropna(subset=[target_col]).reset_index(drop=True)
x = df.drop(columns=[target_col])
y = df[target_col]
numericaltransformer=Pipeline(steps=[
    ("simpleimputer",SimpleImputer(strategy="mean")),
    ("scaler",StandardScaler())
])
categorical=Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
columntransformer=ColumnTransformer(transformers=[
    ("scaler",numericaltransformer,['SeniorCitizen','tenure', 'MonthlyCharges','TotalCharges']),
    ("encoder",categorical,['gender','Partner','Dependents','PhoneService','InternetService','Contract','MultipleLines','OnlineSecurity'
                            ,'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','PaymentMethod'])

])
pipeline=Pipeline(steps=[
    ("preprocessor",columntransformer),
    ("model",LogisticRegression())
])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
pipeline.fit(x_train,y_train)
df_unsup = df.drop(columns=['Churn']).reset_index(drop=True)
X_scaled = columntransformer.fit_transform(df_unsup)
a=KMeans(n_clusters=5,random_state=42)                # n_clusters=5 chosen to balance interpretability and customer diversity
df_unsup["cluster"]=a.fit_predict(X_scaled)
print(df_unsup)
print(df_unsup["cluster"].value_counts())
df_analysis = df_unsup.copy()
df_analysis['Churn'] = y.values
churn_by_cluster = (
    df_analysis
    .groupby('cluster')['Churn']
    .value_counts(normalize=True)
    .rename('churn_rate')
    .reset_index()
)
churn_by_cluster = churn_by_cluster[churn_by_cluster['Churn'] == 'Yes']
print(churn_by_cluster)
revenue_by_cluster = (
    df_analysis
    .groupby('cluster')[['MonthlyCharges', 'TotalCharges']]
    .mean()
    .reset_index()
)
print(revenue_by_cluster)
final_insights = churn_by_cluster.merge(
    revenue_by_cluster,
    on='cluster',
    how='inner'
)
print(final_insights)
def recommend(row):
    if row['churn_rate'] > 0.3 and row['TotalCharges'] > 3000:
        return "URGENT retention: discounts + loyalty rewards"
    elif row['churn_rate'] > 0.3:
        return "Targeted offers & contract changes"
    elif row['TotalCharges'] > 4000:
        return "Protect: loyalty & premium services"
    else:
        return "Maintain & monitor"
final_insights['Recommendation'] = final_insights.apply(recommend, axis=1)
print(final_insights)
sns.barplot(
    data=final_insights,
    x='cluster',
    y='churn_rate'
)
plt.title("Churn Rate by Customer Segment")
plt.show()
print(final_insights[['cluster','churn_rate','TotalCharges','Recommendation']])
import joblib

joblib.dump(pipeline, "customer_model.pkl")
