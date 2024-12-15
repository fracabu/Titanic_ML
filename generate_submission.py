# 1. Importa le librerie necessarie
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 2. Creazione della cartella "output" se non esiste
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 3. Caricamento dei dati
train_data = pd.read_csv("output/train_cleaned.csv")
test_data = pd.read_csv("test.csv")

# 4. Preprocessing dei dati di training e test
# Feature "IsAlone" su train_data e test_data
train_data['IsAlone'] = (train_data['FamilySize'] == 1).astype(int)
test_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
test_data['Cabin'] = test_data['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
test_data['IsAlone'] = (test_data['FamilySize'] == 1).astype(int)

# Aggiunta della colonna Title
test_data['Title'] = test_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Countess": "Rare", "Ms": "Miss", "Lady": "Rare",
    "Jonkheer": "Rare", "Don": "Rare", "Dona": "Rare", "Mme": "Mrs",
    "Capt": "Rare", "Sir": "Rare"
}
test_data['Title'] = test_data['Title'].map(title_mapping)

# 5. Selezione delle feature per il modello
features = ['Pclass', 'Age', 'Fare', 'Cabin', 'FamilySize', 'IsAlone']
X_train = train_data[features]
y_train = train_data['Survived']
X_test = test_data[features]

# 6. Creazione e addestramento del modello Random Forest
print("### Addestramento del modello Random Forest ###")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Modello addestrato con successo!")

# 7. Predizione sui dati di test
print("### Generazione delle predizioni ###")
test_data['Survived'] = model.predict(X_test)

# 8. Creazione del file di submission
submission = test_data[['PassengerId', 'Survived']]
output_submission = os.path.join(output_dir, "submission_with_isalone.csv")
submission.to_csv(output_submission, index=False)
print(f"### File di submission salvato correttamente in '{output_submission}' ###")
