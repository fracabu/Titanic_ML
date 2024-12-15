# Importa le librerie necessarie
import os
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
import matplotlib.pyplot as plt

# Configura il logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===================== FUNZIONI =====================

# Funzione per creare la cartella output
def create_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Cartella '{output_dir}' creata con successo!")
    else:
        logging.info(f"Cartella '{output_dir}' già esistente.")

# Funzione per il caricamento dei dati
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

# Funzione per il feature engineering
def feature_engineering(dataset):
    try:
        dataset['Cabin'] = dataset['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'U')
        dataset['FareBin'] = pd.qcut(dataset['Fare'], 4, labels=False)
        dataset['AgeBin'] = pd.cut(dataset['Age'], bins=[0, 12, 18, 35, 60, 100], labels=False)
        dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        dataset['Title'] = dataset['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Sir', 'Lady', 'Don', 'Countess',
                                                     'Jonkheer', 'Capt', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
        dataset['Pclass_Sex'] = dataset['Pclass'].astype(str) + "_" + dataset['Sex']
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        dataset['IsAlone'] = (dataset['FamilySize'] == 1).astype(int)
    except Exception as e:
        logging.error(f"Errore durante il feature engineering: {e}")
        exit()

# Funzione per la codifica delle feature
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

# Funzione per il training del modello
def train_model(X_train, y_train, output_dir):
    try:
        model = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        logging.info(f"Accuratezza media cross-validation: {scores.mean():.2f}")
        model.fit(X_train, y_train)
        logging.info("Modello addestrato con successo!")
        # Salva il modello
        model_path = os.path.join(output_dir, "random_forest_model.pkl")
        joblib.dump(model, model_path)
        logging.info(f"Modello salvato in '{model_path}'")
        return model
    except Exception as e:
        logging.error(f"Errore durante il training del modello: {e}")
        exit()

# Funzione per generare le predizioni
def generate_predictions(model, X_test, test_data, output_dir):
    try:
        test_data['Survived'] = model.predict(X_test)
        submission = test_data[['PassengerId', 'Survived']]
        output_submission = os.path.join(output_dir, "submission_advanced_features.csv")
        submission.to_csv(output_submission, index=False)
        logging.info(f"File di submission salvato in '{output_submission}'")
    except Exception as e:
        logging.error(f"Errore durante la generazione delle predizioni: {e}")
        exit()

# Funzione per visualizzare le feature importance
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
    features = ['Pclass', 'Sex', 'Cabin', 'Title', 'FareBin', 'AgeBin', 'FamilySize', 'IsAlone', 'Pclass_Sex']

    # Creazione della cartella di output
    create_output_directory(output_dir)

    # Caricamento dei dati
    train_data, test_data = load_data(train_path, test_path)

    # Backup dei dati originali
    train = train_data.copy()
    test = test_data.copy()

    # Feature engineering
    for dataset in [train, test]:
        feature_engineering(dataset)

    # Preprocessamento e codifica
    X_train, X_test = preprocess_and_encode(train, test, features)
    y_train = train_data['Survived']

    # Training del modello
    model = train_model(X_train, y_train, output_dir)

    # Visualizzazione delle feature importance
    plot_feature_importance(model, X_train)

    # Generazione delle predizioni
    generate_predictions(model, X_test, test_data, output_dir)
