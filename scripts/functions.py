### Import packages ###

import numpy as np
import pandas as pd
from typing import Literal

### Machine Learning ###

# Pre-processing
from sklearn.preprocessing import LabelEncoder

# Modelisation
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    GridSearchCV,
)

from scipy import stats
from scipy.stats import pearsonr, spearmanr

# Metrics
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    auc,
    roc_curve,
)
import shap

# Optimisation
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns

### Pre-processing functions ###


def replacing(df : pd.DataFrame):  # Function that will group 'Null's and 'NaN's as 'NaN's

    df.replace("None", np.nan, inplace=True)
    df.replace("Null", np.nan, inplace=True)

    return df


def nan_colormap(df : pd.DataFrame, title : str):  # Function using seaborn's colormap to locate NaNs, use after 'replacing'

    sns.heatmap(df.isna(), cbar=False)
    plt.title(title, fontdict={"fontsize": 18}, pad=12)


def spot_columns(df : pd.DataFrame):  # Function that will return a print with the columns filled with NaNs and their numbers

    nan_by_columns = df.isna().sum()
    columns = nan_by_columns[nan_by_columns > 0]

    for column, nb_nan in nan_by_columns.items():
        print(f"Column '{column}' : {nb_nan} NaN values")
    print('\n\n')

def spot_deviants_z1(df : pd.DataFrame): # Function that spots the deviants values with the Z1-method

    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    df_numeric.dropna(axis=1, how='all'
                      )
    z_scores = np.abs(stats.zscore(df))  
    outliers = np.where(z_scores > 3)

    print("Indices des valeurs aberrantes (Z-score):")
    print(outliers)


def nan_identer(df : pd.DataFrame):  # Replaces qualitatives by 'no_data' str and quantitatives by the column's mean

    df_copy = df.copy()  # Must work on a copy or I get errors after

    for col in df_copy.columns:
        if df_copy[col].dtype in ["float64", "int64"]:
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
        else:
            df_copy[col] = df_copy[col].fillna("no_data")

    return df_copy


def frequencie_pie_chart(df : pd.DataFrame, col : str, title : str):  # Frequencie pie chart on a specific column
    freq = pd.crosstab(df[col], columns="count")
    freq = freq * 100 / freq.sum()

    color_map = sns.color_palette("BrBG")

    plt.figure(figsize=(10, 6))
    plt.pie(
        freq["count"],
        labels=freq.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=color_map,
        wedgeprops={"linewidth": 2, "edgecolor": "black"},
    )
    plt.title(title, fontdict={"fontsize": 18}, pad=12)
    plt.show()


def correlation_matrix(df: pd.DataFrame, method: Literal['pearson', 'kendall', 'spearman'], title: str):  # Returns the correlation matrix colormap of a DataFrame

    # Calculer la matrice de corrélation
    tab = df.corr(method=method)

    # Création de la figure et de l'axe
    plt.figure(figsize=(10, 8))
    sns.heatmap(tab, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, cbar_kws={"shrink": .8})

    # Rotation des étiquettes des axes x pour une meilleure lisibilité
    plt.xticks(rotation=90)

    # Ajouter un titre
    plt.title(title, fontdict={"fontsize": 18}, pad=12)

    # Afficher la figure
    plt.show()

def univariate_analysis(df):
    
    # Séparer les variables numériques et catégorielles
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=[object, 'category']).columns
    
    # Statistiques descriptives pour les variables numériques
    print("\nStatistiques descriptives pour les variables numériques :")
    print(df[numeric_cols].describe())
    
    # Visualisation des distributions des variables numériques sans palette
    for col in numeric_cols:
        plt.figure(figsize=(12, 6))
        
        # Histogramme sans palette
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Histogramme de {col}')
        
        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot de {col}')
        
        plt.tight_layout()
        plt.show()
    
    # Visualisation des fréquences des variables catégorielles
    for col in categorical_cols:
        plt.figure(figsize=(12, 6))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f'Diagramme en barres de {col}')
        plt.show()

def bivariate_analysis(df):
    # Séparer les variables numériques et catégorielles
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=[object, 'category']).columns
    
    # Analyse des relations entre variables numériques
    for col1 in numeric_cols:
        for col2 in numeric_cols:
            if col1 != col2:
                plt.figure(figsize=(12, 6))
                
                # Nuage de points avec ligne de régression
                plt.subplot(1, 2, 1)
                sns.regplot(x=df[col1], y=df[col2], scatter_kws={'s':10})
                plt.title(f'{col1} vs {col2}')
                
                # Afficher le coefficient de corrélation
                corr_pearson, _ = pearsonr(df[col1], df[col2])
                corr_spearman, _ = spearmanr(df[col1], df[col2])
                plt.subplot(1, 2, 2)
                sns.heatmap([[corr_pearson, corr_spearman]], annot=True, cmap='coolwarm', cbar=False,
                            xticklabels=['Pearson', 'Spearman'], yticklabels=['Correlation'])
                plt.title(f'Corrélation entre {col1} et {col2}')
                
                plt.tight_layout()
                plt.show()
    
    # Analyse des relations entre variables catégorielles
    for col1 in categorical_cols:
        for col2 in categorical_cols:
            if col1 != col2:
                plt.figure(figsize=(12, 6))
                sns.countplot(x=col1, hue=col2, data=df)
                plt.title(f'{col1} vs {col2}')
                plt.show()
    
    # Analyse des relations entre variables numériques et catégorielles
    for col1 in categorical_cols:
        for col2 in numeric_cols:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=col1, y=col2, data=df)
            plt.title(f'{col1} vs {col2}')
            plt.show()



### Modélisation ###


def cross_validation(model, x_train : pd.DataFrame, y_train : pd.DataFrame, n_fold : int, scoring_method : str):  # Cross-validation

    kf = KFold(n_splits=n_fold, shuffle=True)
    score = cross_val_score(model, x_train, y_train, cv=kf, scoring=scoring_method)

    print(
        "##### CROSS-VALIDATION SCORES #####",
        "\n\n",
        scoring_method,
        " scores from cross-validation ",
        score,
        "\n",
        scoring_method,
        " mean from cross-validation ",
        score.mean(),
        "\n",
        "Standard deviation ",
        score.std(),
        "\n\n",
        "##################################",
    )


def metrics(model, x_test : pd.DataFrame, y_test : pd.DataFrame, title : str, type : Literal['classifier', 'regressor']):

    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]  # ?

    if type == "classifier":
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.2)
        sns.heatmap(cm, annot=True, cmap="coolwarm", cbar=True)
        plt.xlabel("Predicted")
        plt.ylabel("Originals")
        plt.title(title, fontdict={"fontsize": 18}, pad=12)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
    gini = 2 * auc - 1

    print(
        "##### SCORES #####",
        "\n\n",
        "Accuracy = ",
        accuracy,
        "\n",
        "Precision = ",
        precision,
        "\n",
        "Recall = ",
        recall,
        "\n",
        "F1 - Score = ",
        f1,
        "\n",
        "GINI = ",
        gini,
        "\n\n",
        "##################",
    )

def hyperopt(model, space : dict, x_train : pd.DataFrame, y_train : pd.DataFrame, x_test : pd.DataFrame, y_test : pd.DataFrame) : 

    def objective(params) :
        model_trained = model(**params, objective = 'binary:logistic')
        model_trained.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        return {'loss' : -accuracy, 'status' : STATUS_OK}
    
    trials = Trials()

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials
    )

    print('Meilleurs paramètres trouvés :')