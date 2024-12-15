# Importa le librerie necessarie
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import logging

# Configura il logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===================== FUNZIONI =====================

# Creazione della cartella di output
def create_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Cartella '{output_dir}' creata con successo!")
    else:
        logging.info(f"Cartella '{output_dir}' già esistente.")

# Caricamento dei dati
def load_data(train_path, test_path):
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logging.info("Dati caricati con successo!")
        return train_data, test_data
    except FileNotFoundError as e:
        logging.error(f"Errore: {e}")
        exit()
    except pd.errors.ParserError as e:
        logging.error(f"Errore nel parsing del CSV: {e}")
        exit()

# Feature Engineering Avanzata
# Feature Engineering Avanzata
def advanced_feature_engineering(dataset):
    try:
        # Creazione della colonna Title
        dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
        dataset['Title'] = dataset['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Sir', 'Lady', 'Don', 'Countess',
                                                     'Jonkheer', 'Capt', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

        # Creazione di CabinLetter
        dataset['CabinLetter'] = dataset['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'U')

        # Creazione di FamilySize
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

        # Creazione di IsAlone
        dataset['IsAlone'] = (dataset['FamilySize'] == 1).astype(int)

        # Creazione di FarePerPerson
        dataset['FarePerPerson'] = dataset['Fare'] / dataset['FamilySize']

        # Creazione di FareBin
        dataset['FareBin'] = pd.qcut(dataset['Fare'], 4, labels=False)

        # Creazione di AgeBin
        dataset['AgeBin'] = pd.cut(dataset['Age'], bins=[0, 12, 18, 35, 60, 100], labels=False)

        # Creazione di Pclass_Sex
        dataset['Pclass_Sex'] = dataset['Pclass'].astype(str) + "_" + dataset['Sex']

        # Creazione di Pclass_Sex_Title
        dataset['Pclass_Sex_Title'] = dataset['Pclass'].astype(str) + "_" + dataset['Sex'] + "_" + dataset['Title']

        # Creazione di FamilyType
        dataset['FamilyType'] = dataset['FamilySize'].apply(lambda x: 'Small' if x <= 2 else 'Large' if x > 4 else 'Medium')
    except Exception as e:
        logging.error(f"Errore durante il feature engineering avanzato: {e}")
        exit()

# Preprocessing e Codifica
def preprocess_and_encode(train, test, features):
    try:
        train_encoded = pd.get_dummies(train[features])
        test_encoded = pd.get_dummies(test[features])
        train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1)
        test_encoded = test_encoded.fillna(0)
        return train_encoded, test_encoded
    except Exception as e:
        logging.error(f"Errore durante il preprocessamento e la codifica: {e}")
        exit()

# Tuning e Training del modello
def train_xgboost(X_train, y_train, output_dir):
    try:
        model = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.01, random_state=42)
        model.fit(X_train, y_train)
        logging.info("Modello XGBoost addestrato con successo!")
        # Salvataggio del modello
        model_path = os.path.join(output_dir, "xgboost_model.pkl")
        joblib.dump(model, model_path)
        logging.info(f"Modello salvato in '{model_path}'")
        return model
    except Exception as e:
        logging.error(f"Errore durante il training del modello XGBoost: {e}")
        exit()

# Generazione delle predizioni
def generate_predictions(model, X_test, test_data, output_dir):
    try:
        test_data['Survived'] = model.predict(X_test)
        submission = test_data[['PassengerId', 'Survived']]
        output_submission = os.path.join(output_dir, "submission_optimized.csv")
        submission.to_csv(output_submission, index=False)
        logging.info(f"File di submission salvato in '{output_submission}'")
    except Exception as e:
        logging.error(f"Errore durante la generazione delle predizioni: {e}")
        exit()

# Visualizzazione delle feature importance
def plot_feature_importance(model, X_train):
    try:
        feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
        feature_importances.nlargest(10).plot(kind='barh')
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.show()
    except Exception as e:
        logging.error(f"Errore durante la visualizzazione delle feature importance: {e}")
        exit()

# ===================== SCRIPT PRINCIPALE =====================

if __name__ == "__main__":
    # Configurazioni
    output_dir = "output"
    train_path = "output/train_cleaned.csv"
    test_path = "test.csv"
    features = ['Pclass', 'Sex', 'CabinLetter', 'Title', 'FareBin', 'AgeBin',
                'FamilySize', 'IsAlone', 'Pclass_Sex', 'FarePerPerson', 'Pclass_Sex_Title', 'FamilyType']

    # Creazione della cartella di output
    create_output_directory(output_dir)

    # Caricamento dei dati
    train_data, test_data = load_data(train_path, test_path)

    # Backup dei dati originali
    train = train_data.copy()
    test = test_data.copy()

    # Feature engineering avanzata
    for dataset in [train, test]:
        advanced_feature_engineering(dataset)

    # Preprocessamento e codifica
    X_train, X_test = preprocess_and_encode(train, test, features)
    y_train = train['Survived']

    # Scaling delle feature numeriche
    scaler = StandardScaler()
    X_train[['FarePerPerson']] = scaler.fit_transform(X_train[['FarePerPerson']])
    X_test[['FarePerPerson']] = scaler.transform(X_test[['FarePerPerson']])

    # Training del modello con XGBoost
    model = train_xgboost(X_train, y_train, output_dir)

    # Visualizzazione delle feature importance
    plot_feature_importance(model, X_train)

    # Generazione delle predizioni
    generate_predictions(model, X_test, test_data, output_dir)
