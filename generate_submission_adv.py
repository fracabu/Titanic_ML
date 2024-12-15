# Importa le librerie necessarie
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ===================== CREAZIONE DELLA CARTELLA OUTPUT =====================
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Cartella '{output_dir}' creata con successo!")

# ===================== CARICAMENTO DEI DATI =====================
train_data = pd.read_csv("output/train_cleaned.csv")
test_data = pd.read_csv("test.csv")

# Copia dei dati originali per sicurezza
train = train_data.copy()
test = test_data.copy()

# ===================== FEATURE ENGINEERING =====================

# 1. Cabin: Estraiamo il primo carattere (lettera) della cabina
for dataset in [train, test]:
    dataset['Cabin'] = dataset['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'U')  # 'U' per Unknown

# 2. Fare: Creiamo fasce di prezzo
for dataset in [train, test]:
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4, labels=False)

# 3. Age: Creiamo fasce d'età
for dataset in [train, test]:
    dataset['AgeBin'] = pd.cut(dataset['Age'], bins=[0, 12, 18, 35, 60, 100], labels=False)

# 4. Title: Riduciamo ulteriormente i titoli
for dataset in [train, test]:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Sir', 'Lady', 'Don', 'Countess',
                                                 'Jonkheer', 'Capt', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# 5. Pclass_Sex: Combiniamo Pclass e Sex
for dataset in [train, test]:
    dataset['Pclass_Sex'] = dataset['Pclass'].astype(str) + "_" + dataset['Sex']

# 6. FamilySize: Creiamo la colonna se non già presente
for dataset in [train, test]:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# 7. IsAlone: Identifichiamo se il passeggero è da solo
for dataset in [train, test]:
    dataset['IsAlone'] = (dataset['FamilySize'] == 1).astype(int)

# ===================== SELEZIONE E CODIFICA DELLE FEATURE =====================
# Features finali per il modello
features = ['Pclass', 'Sex', 'Cabin', 'Title', 'FareBin', 'AgeBin', 'FamilySize', 'IsAlone', 'Pclass_Sex']

# Encoding delle variabili categoriche usando get_dummies
train_encoded = pd.get_dummies(train[features])
test_encoded = pd.get_dummies(test[features])

# Allineiamo le colonne tra train e test
train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1)
test_encoded = test_encoded.fillna(0)  # Riempiamo eventuali NaN con 0

# ===================== TRAINING DEL MODELLO =====================
X_train = train_encoded
y_train = train_data['Survived']
X_test = test_encoded

print("### Addestramento del modello Random Forest ###")
model = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42)
model.fit(X_train, y_train)
print("Modello addestrato con successo!")

# ===================== PREDIZIONE E SALVATAGGIO =====================
print("### Generazione delle predizioni ###")
test_data['Survived'] = model.predict(X_test)

# Creazione del file di submission
submission = test_data[['PassengerId', 'Survived']]
output_submission = os.path.join(output_dir, "submission_advanced_features.csv")
submission.to_csv(output_submission, index=False)
print(f"### File di submission salvato correttamente in '{output_submission}' ###")
