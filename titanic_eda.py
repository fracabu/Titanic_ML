# Importa le librerie necessarie
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Creazione della cartella "output" se non esiste
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Cartella '{output_dir}' creata con successo!")

# Caricamento dei dati
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# 1. Informazioni generali sul dataset
print("### Informazioni sul dataset di training ###")
print(train_data.info())
print("\n### Prime righe del dataset ###")
print(train_data.head())

# 2. Valori mancanti
print("\n### Valori mancanti nel dataset di training ###")
print(train_data.isnull().sum())

# Visualizzazione dei valori mancanti
plt.figure(figsize=(10, 6))
sns.heatmap(train_data.isnull(), cbar=False, cmap='viridis')
plt.title("Mappa dei Valori Mancanti")
plt.show()

# 3. Gestione dei Valori Mancanti
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
train_data['Cabin'] = train_data['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)

# 4. Creazione di Nuove Features
# Creazione della feature FamilySize
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1

# Estrazione del titolo dai nomi
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Countess": "Rare", "Ms": "Miss", "Lady": "Rare",
    "Jonkheer": "Rare", "Don": "Rare", "Dona": "Rare", "Mme": "Mrs",
    "Capt": "Rare", "Sir": "Rare"
}
train_data['Title'] = train_data['Title'].map(title_mapping)

# Controllo dei dati aggiornati
print("\n### Valori mancanti dopo la pulizia ###")
print(train_data.isnull().sum())
print("\n### Prime righe con le nuove features ###")
print(train_data[['Age', 'Embarked', 'Cabin', 'FamilySize', 'Title']].head())

# 5. Analisi delle Nuove Features
# FamilySize
plt.figure(figsize=(6, 4))
sns.countplot(x='FamilySize', hue='Survived', data=train_data)
plt.title("Sopravvivenza in base alla Dimensione della Famiglia")
plt.show()

# Title
plt.figure(figsize=(10, 6))
sns.countplot(x='Title', hue='Survived', data=train_data)
plt.title("Sopravvivenza in base al Titolo")
plt.xticks(rotation=45)
plt.show()

# 6. Matrice di Correlazione
plt.figure(figsize=(10, 8))

# Seleziona solo colonne numeriche per il calcolo della correlazione
numeric_columns = train_data.select_dtypes(include=['number'])

correlation_matrix = numeric_columns.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Matrice di Correlazione")
plt.show()

# 7. Salvataggio del Dataset Pulito
output_path = os.path.join(output_dir, "train_cleaned.csv")

# Stampa di Debug: Controllo delle prime righe di train_data
print("\n### Controllo del contenuto di train_data prima del salvataggio ###")
print(train_data.head())

# Controllo del percorso completo
print(f"\n### Salvataggio del dataset in '{os.path.abspath(output_path)}' ###")

try:
    train_data.to_csv(output_path, index=False)
    print(f"### Dataset pulito salvato correttamente in '{output_path}' ###")
except Exception as e:
    print(f"Errore durante il salvataggio del file: {e}")
