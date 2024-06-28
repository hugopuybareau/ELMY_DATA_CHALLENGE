### Import packages ###

import numpy as np
import pandas as pd

### Machine Learning ###

# Pre-processing
from sklearn.preprocessing import LabelEncoder 

# Modelisation
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV

# Metrics 
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc, roc_curve
import shap

# Graphics
import matplotlib.pyplot as plt 
import seaborn as sns

# Models
import xgboost
from sklearn.ensemble import RandomForestClassifier
import catboost as cb


### Pre-processing functions ###

def replacing(df) : # Function that will group 'Null's and 'NaN's as 'NaN's

    df.replace('None', np.nan, inplace=True)
    df.replace('Null', np.nan, inplace=True)
    
    return df

def nan_colormap(df) : # Function using seaborn's colormap to locate NaNs, use after 'replacing'

    sns.heatmap(df.isna(), cbar=False)

def spot_columns(df) : # Function that will return a print with the columns filled with NaNs and their numbers
 
    nan_by_columns = df.isna().sum()
    columns = nan_by_columns[nan_by_columns>0]
    
    for column, nb_nan in nan_by_columns.items() :
        print(f"Column '{column}' : {nb_nan} NaN values")

def nan_identer(df) : # Replaces qualitatives by 'no_data' str and quantitatives by the column's mean

    df_copy = df.copy() # Must work on a copy or I get errors after

    for col in df_copy.columns : 
        if col.dtype() in ['float64', 'int64'] : 
            df_col = df_col.fillna(col.mean)
        else :
            df_col = df_col.fillna('no_data')

    return df_copy

def frequencie_pie_chart(df, col, title) : # Frequencie pie chart on a specific column
    freq = pd.crosstab(df[col], columns='count')
    freq = freq*100/freq.sum()

    color_map = sns.color_palette('BrBG')

    plt.figure(figsize=(10,6))
    plt.pie(
        freq['count'],
        labels=freq.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=color_map,
        wedgeprops={
            'linewidth' : 2,
            'edgecolor' : 'black'
            }
        )
    plt.title(title, fontdict={'fontsize' : 18}, pad=12)
    plt.show()

def correlation_matrix(df, method, title) : # Returns the correlation matrix colormap of a DataFrame

    tab = df.corr(method=method)

    fig, ax = plt.subplots()
    im = ax.pcolor(tab, cmap='BrBG')
    row_labels = tab.columns
    column_labels = tab.index
    ax.set_xticks(np.arange(tab.shape[1])+0.5, minor=False)
    plt.xticks(rotation=90)
    ax.set_yticks(np.arange(tab.shape[0])+0.5, minor=False)
    ax.set_xtickslabels(row_labels, minor=False)
    ax.set_ytickslabels(column_labels, minor=False)
    plt.title(title, fontdict={'fontsize' : 18}, pad=12)
    fig.colorbar(im)
    plt.show()

### Modélisation ### 

def cross_validation(model, x_train, y_train, n_fold, scoring_method) : # Cross-validation 
    kf = KFold(n_splits=n_fold, shuffle=True)
    score = cross_val_score(model, x_train, y_train, cv=kf, scoring=scoring_method)

    print(
        '##### CROSS-VALIDATION SCORES #####',
        '\n\n',
        scoring_method, ' scores from cross-validation ', score,
        '\n',
        scoring_method, ' mean from cross-validation ', score.mean(),
        '\n',
        'Standard deviation ', score.std(),
        '\n\n',
        '###################################'
    )

def metrics(model, x_test, y_test, title, type) :

    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1] # ?
    
    if type == 'classifier' :
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.set(font_scale=1.2)
        sns.heatmap(
            cm,
            annot=True,
            cmap='coolwarm',
            cbar=True
        )
        plt.xlabel('Predicted')
        plt.ylabel('Originals')
        plt.title(title, fontdict={'fontsize' : 18}, pad=12)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    gini = 2*auc-1

    print(
        '##### SCORES #####',
        '\n\n',
        'Accuracy = ', accuracy,
        '\n',
        'Precision = ', precision,
        '\n',
        'Recall = ', recall,
        '\n',
        'F1 - Score = ', f1,
        '\n',
        'GINI = ', gini,
        '\n\n',
        '##################'  
    )








