# %% [markdown]
# # COMP 352 Final Project - Spotify Predictions

# %% [markdown]
# ### By Cameron McNamara, Bilal Adam, Maximo Babun

# %% [markdown]
# Requirements: 
#   - There are four sections of the final project. You are expected to perform the following tasks within each section to fulfill the project requirements. Remember data science is cyclical in nature and requires multiple attempts and iterations. It is okay if your code moves between sections as you try different approaches, but at the end please try and organize your code into these sections for grading purposes.
# - Data Importing and Pre-processing (100 Points)
#   - Import dataset and describe characteristics such as dimensions, data types, file types, and import methods used
#   - Clean, wrangle, and handle missing data, duplicate data, etc.
#   - Encode any categorical variables
#   - Perform feature engineering on the dataset
#   - Transform data appropriately using techniques such as aggregation, normalization, and feature construction
#   - Reduce redundant data and perform need based discretization
# - Data Analysis and Visualization (100 Points)
#   - Identify categorical, ordinal, and numerical variables within data
#   - Provide measures of centrality and distribution with visualizations
#   - Diagnose for correlations between variables and determine independent and dependent variables
#   - Perform exploratory analysis in combination with visualization techniques to discover patterns and features of interest
#   - Create visualizations that allow for the discovery of insights in the data
# 
# - Data Analytics (100 Points)
#   - Determine the need for a supervised or unsupervised learning method and identify dependent and independent variables
#   - Choose and provide reasoning for the selected metric or metrics employed to assess your model.
#   - Train, test, cross validate, and provide performance metrics for model results
#   - Try multiple different types of algorithms to determine the best model for your dataset
#   - Analyze your model performance
# 

# %% [markdown]
# First we must setup our environment to make sure we have all appropriate modules installed. To do this, I have provided 2 methods. The 1st, is to install all modules using a ```.yaml``` file via ```conda```. 
# 
# To do this, run:
# ```bash
# conda env create -f env_setup/data_environment.yml
# ```
# Then activate the environment by:
# ```bash
# conda activate data_env
# ```

# %% [markdown]
# ## 1. Data Importing and Pre-processing <a class="anchor" id="data-importing"></a>

# %%
# import libraries needed
import pandas as pd

pd.set_option("display.max_columns", None)
import warnings

import branca
import folium
import geopandas as gpd
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgboost as xgb
from branca.element import Figure
from folium import Marker
from folium.plugins import HeatMap
from scipy.special import boxcox1p
from scipy.stats import norm, probplot, skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from utils.model_utils import (
    time_series_split_regression,
    StackedEnsembleCVRegressor,
)
from utils.metrics_utils import (
    compute_rmse_std,
    print_rmse_and_dates,
)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.*")
%matplotlib inline

# %%
## Import Data
spotify = pd.read_csv("SpotifyFeatures.csv")

# %% [markdown]
# #### Explore Dataset Dimensions

# %%
print(spotify.shape)
print("Total Observations:", spotify.shape[0])

# %%
cat_count = 0
for dtype in spotify.dtypes:
    if dtype == "object":
        cat_count = cat_count + 1

# %%
print("# of categorical variables:", cat_count)

numeric_vars = spotify.shape[1] - cat_count - 1
print("# of continous variables:", numeric_vars)

# %% [markdown]
# #### Remove Unecessary Columns

# %%
spotify.head()

# %%
# Remove Artist_name and track_name - high cardinality providing no predictive power.
spotify = spotify.drop(columns=["artist_name", "track_name"])

# %% [markdown]
# #### Check for missing values

# %%
total = spotify.isnull().sum().sort_values(ascending=False)
percent = (spotify.isnull().sum() / spotify.isnull().count()).sort_values(
    ascending=False
)
missing_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
missing_data.head(20)

# %%
# No Missing Values - Moving onto one-hot encoding of categorical variables

# %% [markdown]
# #### Check for Duplicate Song IDs

# %%
spotify["track_id"].duplicated().sum()

# %%
dup_ids = spotify[spotify["track_id"].duplicated(keep=False)]
dup_ids.sort_values("track_id").head(10)


# %%
genre_dummies = pd.get_dummies(spotify["genre"], prefix="genre")
spotify_with_dummies = pd.concat([spotify, genre_dummies], axis=1)

# Group by track_id, taking max for genre columns (so all applicable genres = 1)
genre_cols = [col for col in spotify_with_dummies.columns if col.startswith("genre_")]

spotify_clean = spotify_with_dummies.groupby("track_id", as_index=False).agg(
    {
        **{
            col: "first"
            for col in spotify.columns
            if col not in ["track_id", "genre", "popularity"]
        },
        **{col: "max" for col in genre_cols},  # Max ensures all genres are captured
        "popularity": "mean",  # or 'max', your choice
    }
)

spotify=spotify_clean

# %%
spotify.head()
spotify.shape

# %% [markdown]
# There are 232,000 different records, however, certain songs appear more than once under different genres. In order to get around these duplicates, we only kept one observation per song id. In order to do this we created a one-hot encoded variables for genre, and aggregated them based on song_id. THis means taht if a song applies to more than one genre, they will have more than one TRUE for the binary encoded genre columns. In this aggregation we took the mean of the popularity of the two columns, if it was differing.

# %% [markdown]
# #### One-Hot Encode reamining cateogrical variables

# %%
cat_cols = ["key", "time_signature", "mode"]
spotify = pd.get_dummies(
    spotify,
    columns=cat_cols,
    prefix=cat_cols,
    drop_first=True
)

# %%
spotify.shape

# %%
spotify.head()

# %% [markdown]
# #### Feature Engineering

# %%
spotify["Hit"] = (spotify["popularity"] >= 70).astype(int)

# %% [markdown]
# Create Binary "Hit" column, classifying all songs with popularity >= 70 as a hit

# %%
spotify["very_loud"] = (spotify["loudness"] > spotify["loudness"].quantile(0.9)).astype(int)
spotify["very_quiet"] = (spotify["loudness"] < spotify["loudness"].quantile(0.1)).astype(int)
spotify["tempo_fast"] = (spotify["tempo"] >= 120).astype(int)
spotify["tempo_slow"] = (spotify["tempo"] <= 90).astype(int)

# %% [markdown]
# Create banded very loud, very quiet, fast tempo, adn slow tempo, binary columns to help with hit classifcation. These are often penalized (i.e. very loud songs are not very popular, same with very quiet songs).

# %%
spotify["energy_dance"] = spotify["energy"] * spotify["danceability"]

# %% [markdown]
# Create an energy * dance feature to represent how danceable and energetic a song is, as many hits are both danceable and energetic

# %% [markdown]
# ## 2. Data analysis and Visualization
# 

# %% [markdown]
# 2.1: Identify categorical, ordinal, and numerical variables

# %%
numerical_cols = spotify.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = spotify.select_dtypes(include=["object", "bool"]).columns

len(numerical_cols), len(categorical_cols)

# %%
numerical_cols

# %%
categorical_cols

# %% [markdown]
# #### Variable Types
# 
# - **Numerical variables** include audio features such as popularity, tempo, loudness, energy, danceability, and duration.
# - **Categorical variables** include identifiers and encoded genre and musical attributes.
# - No ordinal variables are explicitly present in this dataset.

# %% [markdown]
# ### 2.2 Provide measures of centrality and distribution with visualizations

# %%
spotify[numerical_cols].describe()

# %%
import matplotlib.pyplot as plt

plt.hist(
    spotify["popularity"],
    bins=30,
    color="#AEC6CF",
    edgecolor="black"
)

plt.xlabel("Popularity")
plt.ylabel("Count")
plt.title("Distribution of Song Popularity")
plt.grid(axis="y", alpha=0.4)

plt.show()

# %%
plt.boxplot(spotify["popularity"], vert=False)
plt.xlabel("Popularity")
plt.title("Popularity Boxplot")
plt.show()

# %% [markdown]
# The popularity variable is right-skewed, with most songs having lower popularity values.
# A small number of songs appear as outliers with very high popularity.

# %% [markdown]
# ### 2.3: Correlations + Independent Dependent Variable

# %%
corr_matrix = spotify[numerical_cols].corr()
corr_matrix

# %%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# %% [markdown]
# Several audio features, including loudness, energy, and danceability, show noticeable positive relationships with popularity. Louder and more energetic songs tend to achieve higher popularity scores, while acousticness shows a weaker or negative relationship. These patterns suggest that production intensity and rhythmic qualities may play an important role in song popularity.

# %% [markdown]
# ### Independent and Dependent Variables
# 
# Dependent variable:
#  - Regression Tasks: Popularity, which represents the overall success of a song on Spotify.
#  - Classification (XGboost) Tasks: Genre
# 
# Independent variables:
# Audio characteristics such as loudness, energy, danceability, tempo, genre (For regression only), scale, acousticness, etc. which describe the musical and production qualities of each track and may influence its popularity.

# %% [markdown]
# ### 2.4: EDA/Visualizations
# 

# %% [markdown]
# Exploratory analysis shows that hit songs tend to cluster around higher values of
# danceability and energy, while extremely low popularity songs dominate the dataset.

# %%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=spotify,
    x="loudness",
    y="popularity",
    alpha=0.3
)

plt.title("Loudness vs Popularity")
plt.xlabel("Loudness (dB)")
plt.ylabel("Popularity")
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=spotify,
    x="acousticness",
    y="popularity",
    alpha=0.3
)

plt.title("Acousticness vs Popularity")
plt.xlabel("Acousticness")
plt.ylabel("Popularity")
plt.tight_layout()
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.boxplot(
    x="Hit",
    y="danceability",
    data=spotify
)

plt.title("Danceability by Hit Status")
plt.xlabel("Hit (0 = No, 1 = Yes)")
plt.ylabel("Danceability")
plt.tight_layout()
plt.show()

# %% [markdown]
# Hit songs tend to have higher energy and danceability than non hit songs, suggesting that these features are strongly associated with a song’s popularity.

# %% [markdown]
# ## 3. Modeling

# %% [markdown]
# ### (Supervised) Logistic Regression Hit Prediction

# %% [markdown]
# ##### We made two logistic regression models. The first uses a train set that has a 50/50 split in hit/non-hit data (~3k ea). The second uses 5 Fold CV with Class weights 

# %% [markdown]
# #### Prepare Data 

# %%
# Drop columns that shouldn't be used for prediction
# Keep all the genre_* columns since they're predictive
X = spotify.drop(columns=['Hit', 'popularity', 'track_id'])  
y = spotify['Hit']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# %% [markdown]
# #### Scaling the Data

# %%
from sklearn.preprocessing import StandardScaler

continuous_cols = [
    'acousticness',      
    'danceability',      
    'duration_ms',       
    'energy',            
    'instrumentalness',  
    'liveness',          
    'loudness',          
    'speechiness',       
    'tempo',             
    'valence',           
    'energy_dance'       
]


# %%
scalar = StandardScaler()
X_scaled_cont = scalar.fit_transform(X[continuous_cols])
X_scaled_df = pd.DataFrame(X_scaled_cont, columns=continuous_cols, index=X.index)

binary_cols = [col for col in X.columns if col not in continuous_cols]

X_scaled = pd.concat([X_scaled_df, X[binary_cols]], axis=1)

# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# %%
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# %% [markdown]
# #### Logistic Model 1 - 50/50 Split

# %%
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)  # 1.0 = 50/50

X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)

print("Balanced training set:")
print(f"Hits: {y_train_bal.sum()}")
print(f"Non-hits: {len(y_train_bal) - y_train_bal.sum()}")

# %%
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=2000, random_state=42)

log_reg.fit(X_train_bal, y_train_bal)

# %%
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Hit", "Hit"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix – Logistic Regression (50/50 Training)")
plt.show()

# %% [markdown]
# #### Stratified K-Fold CV Logistic Model

# %%
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = []
precision_scores = []
recall_scores = []
auc_scores = []
f1_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    log_reg_cw = LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=42
    )

    log_reg_cw.fit(X_train, y_train)

    y_val_prob = log_reg_cw.predict_proba(X_val)[:, 1]
    y_val_pred = log_reg_cw.predict(X_val)

    accuracy_scores.append(accuracy_score(y_val, y_val_pred))
    precision_scores.append(precision_score(y_val, y_val_pred))
    recall_scores.append(recall_score(y_val, y_val_pred))
    auc_scores.append(roc_auc_score(y_val, y_val_prob))
    f1_scores.append(f1_score(y_val, y_val_pred))

# %%
print("Class-Weighted Logistic Regression (CV)")
print(f"Mean accuracy_score: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
print(f"Mean precision_score: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
print(f"Mean recall_score: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
print(f"Mean ROC-AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
print(f"Mean F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

# %%
import numpy as np
# Feature Importance from Logistic Regression Coefficients
coefficients = log_reg.coef_[0] 
feature_names = X.columns

# Create DataFrame for feature importance format
data = pd.DataFrame(data=coefficients, index=feature_names, columns=["score"]).sort_values(
    by="score", ascending=False
)

# Plot top 20 features
data[:20].plot(kind="barh", figsize=(20, 10)).invert_yaxis()
plt.xlabel("Feature Importance (|Coefficient|)", fontsize=20)
plt.ylabel("Feature Name", fontsize=20)
plt.title("Logistic Regression Feature Importance Plot", fontsize=20)
plt.show()

# %% [markdown]
# ### (Unsupervised) K-means clustering

# %%
from sklearn.preprocessing import StandardScaler

continuous_cols = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "speechiness",
    "valence",
    "loudness",
    "tempo",
    "duration_ms",
    "energy_dance"
]
scaler = StandardScaler()
spotify[continuous_cols] = scaler.fit_transform(spotify[continuous_cols])

# %%
spotify[continuous_cols].describe()

# %%
X_cluster = spotify[continuous_cols]

# %%
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertias = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    inertias.append(kmeans.inertia_)

# %%
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means")
plt.show()

# %%
from sklearn.metrics import silhouette_score

X_sil = X_cluster.sample(n=10000, random_state=42)

k_values = range(2, 11)
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_sil)
    score = silhouette_score(X_sil, labels)
    silhouette_scores.append(score)
    print(f"k={k}, silhouette={score:.3f}")

# Plot
plt.figure()
plt.plot(k_values, silhouette_scores, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs Number of Clusters")
plt.show()

# %%
X_sil = X_cluster.sample(n=10000, random_state=42)

k_values = range(2, 11)
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_sil)
    score = silhouette_score(X_sil, labels)
    silhouette_scores.append(score)
    print(f"k={k}, silhouette={score:.3f}")

# Plot
plt.figure()
plt.plot(k_values, silhouette_scores, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs Number of Clusters")
plt.xticks(k_values)
plt.grid(True)
plt.show()

# %% [markdown]
# Given this elbow plot with intertia, we were looking at choosing k=5. There was no sharp elbow or inflection point, but after k=5, the returns seemed to dimish slightly. We ran a silouhette score comparison across all k values as well, to double check our choice. We found that the highest silouhette score was at 0.342 at k = 3. Given these results we decided to run two models with differing K values to see if we could get a well performing 3 bucket seperation, and a lower performing but higher interpretability 5 bucket seperation.

# %% [markdown]
# #### K = 3 K Means Model

# %%
k_opt = 3
kmeans_3 = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
spotify["cluster_3"] = kmeans_3.fit_predict(X_cluster)

# %%
spotify.groupby("cluster_3")[continuous_cols].mean()
spotify.groupby("cluster_3")[genre_cols].mean()

# %%
spotify.groupby("cluster_3")[continuous_cols].mean()

# %% [markdown]
# Cluster	Description
# - 0	Instrumental, classical, and cinematic music
# - 1 Spoken / Comedy / Non-musical audio
# - 2	Mainstream, genre-blending popular music

# %% [markdown]
# ##### K-Means k=3 evaluation

# %%
labels = kmeans_3.fit_predict(X_cluster)
score = silhouette_score(X_cluster, labels)
print("Kmeans_3 Intertia:", kmeans_3.inertia_)
print("Kmeans_3 Sillouttte Score:", score)

# %% [markdown]
# Kmeans_3 Intertia: 1147732.3986595399
# Kmeans_3 Sillouttte Score: 0.3485772304273481

# %%

# wanted to visualize features on 2 dimensional graph so used PCA to get principal features adn visualzied 3 clusters
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cluster)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=spotify["cluster_3"], cmap="tab10", alpha=0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clusters (PCA Projection)")
plt.legend()
plt.show()

# %% [markdown]
# #### K = 5 K means model

# %%
k_opt = 5
kmeans_5 = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
spotify["cluster_5"] = kmeans_5.fit_predict(X_cluster)

# %%
spotify.groupby("cluster_5")[continuous_cols].mean()
spotify.groupby("cluster_5")[genre_cols].mean()

# %%
spotify.groupby("cluster_5")[continuous_cols].mean()

# %%
labels_5 = kmeans_5.fit_predict(X_cluster)
score = silhouette_score(X_cluster, labels_5)
print("Kmeans_3 Intertia:", kmeans_3.inertia_)
print("Kmeans_3 Sillouttte Score:", score)

# %% [markdown]
# Kmeans_3 Intertia: 1147732.3986595399
# Kmeans_3 Sillouttte Score: 0.1875342916530692

# %%
# wanted to visualize features on 2 dimensional graph so used PCA to get principal features and visualzied 5 clusters
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cluster)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=spotify["cluster_5"], cmap="tab10", alpha=0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clusters (PCA Projection)")
plt.colorbar(label="Cluster")
plt.show()

# %% [markdown]
# ### (Supervised) XGBoost Genre predictions

# %% [markdown]
#  * Each song can belong to multiple genres

# %%
genre_cols = [c for c in spotify.columns if c.startswith("genre_")]
y = spotify[genre_cols]

# %%
X = spotify.drop(columns=genre_cols + ["track_id", "popularity", "Hit", "cluster_3", "cluster_5"])

# %%
# test train split 80/20
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
# cross validation - used a multi-label stratifcation package from iterstrat in order to keep
# consistent representation of all genres across folds
from sklearn.model_selection import KFold
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# %%
# define model
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier

xgb_base = OneVsRestClassifier(
    XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
)


# %%
#define tuning grid and tune
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

param_grid = {
    "estimator__max_depth": [4, 6],
    "estimator__n_estimators": [150, 250],
    "estimator__learning_rate": [0.05, 0.1],
}


def micro_f1_from_proba(estimator, X, y_true):
    y_proba = estimator.predict_proba(X)
    y_pred = (y_proba >= 0.5).astype(int)
    return f1_score(y_true, y_pred, average="micro")

#use 5 cross fold to tune
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)


grid = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring=micro_f1_from_proba,
    cv=mskf,
    verbose=1,
    n_jobs=-1,
)

# %%
grid.fit(X_train, y_train)

# %%
print("Best params:", grid.best_params_)
print("Best CV Micro F1:", grid.best_score_)

# %% [markdown]
# Best params: {'estimator__learning_rate': 0.1, 'estimator__max_depth': 6, 'estimator__n_estimators': 250}
# Best CV Micro F1: 0.34316938597394303

# %%
best_xgb = grid.best_estimator_

best_xgb.fit(X_train, y_train)

y_test_proba = best_xgb.predict_proba(X_test)
y_test_pred = (y_test_proba >= 0.4).astype(int)

from sklearn.metrics import f1_score, hamming_loss

print("Final Test Performance (XGBoost)")
print("Micro F1:", f1_score(y_test, y_test_pred, average="micro"))
print("Macro F1:", f1_score(y_test, y_test_pred, average="macro"))
print("Hamming:", hamming_loss(y_test, y_test_pred))

# %%
import matplotlib.pyplot as plt
import pandas as pd

# Get feature importance from first estimator
booster = best_xgb.estimators_[0]
importances = booster.feature_importances_

feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)

plt.figure()
feat_imp.head(15).plot(kind="bar")
plt.title("Top Audio Features Used for Genre Prediction")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# %%
true_counts = y_test.sum().sort_values(ascending=False)
pred_counts = pd.DataFrame(y_test_pred, columns=y_test.columns).sum()

df_counts = pd.DataFrame({
    "Actual": true_counts,
    "Predicted": pred_counts
}).head(10)

df_counts.plot(kind="bar")
plt.title("Actual vs Predicted Genre Frequency (Top Genres)")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# %%
!pip install shap

# %%
import shap

explainer = shap.TreeExplainer(best_xgb.estimators_[0])
shap_values = explainer.shap_values(X_train)


cont_idx = [X_train.columns.get_loc(c) for c in continuous_cols]

shap_cont = shap_values[:, cont_idx]
X_cont = X_train[continuous_cols]
shap.summary_plot(shap_values, X_train)


