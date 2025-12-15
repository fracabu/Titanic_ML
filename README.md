<h1 align="center">Titanic ML</h1>
<h3 align="center">Interactive Machine Learning on Titanic Dataset</h3>

<p align="center">
  <em>Advanced experiments with modern Streamlit UI - Kaggle score: 0.77751</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/XGBoost-2.0-189A00?style=flat-square" alt="XGBoost" />
  <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white" alt="Kaggle" />
</p>

<p align="center">
  :gb: <a href="#english">English</a> | :it: <a href="#italiano">Italiano</a>
</p>

---

<a name="english"></a>
## :gb: English

### Overview

A complete Machine Learning application combining advanced experiments with a modern user interface for Titanic survival analysis and prediction.

### Features

- **Interactive Training** - Real-time model training
- **Dynamic Visualizations** - Plotly charts and insights
- **Automated Feature Engineering** - Advanced data preprocessing
- **Multiple Submissions** - Optimized Kaggle submissions
- **Modern UI** - Responsive Streamlit dashboard

### Results

| Submission | Kaggle Score | Model |
|------------|--------------|-------|
| Advanced Features | **0.77751** | Random Forest (200 trees) |
| Optimized | 0.75598 | XGBoost tuned |
| Base | 0.61722 | Random Forest basic |

### Metrics

- Accuracy: 84.9%
- Precision: 82.2%
- Recall: 81.1%

### Tech Stack

| Category | Technologies |
|----------|--------------|
| Core | Python, Pandas, NumPy, Scikit-learn, XGBoost |
| Visualization | Streamlit, Plotly, Seaborn |
| Development | Git, Virtual Env, Jupyter |

### Quick Start

```bash
git clone https://github.com/fracabu/Titanic_ML.git
cd Titanic_ML

python -m venv venv
venv\Scripts\activate  # Windows

pip install -r requirements.txt
streamlit run titanic_app.py
```

---

<a name="italiano"></a>
## :it: Italiano

### Panoramica

Un'applicazione completa di Machine Learning che combina esperimenti avanzati con un'interfaccia utente moderna per l'analisi e la predizione della sopravvivenza sul Titanic.

### Funzionalita

- **Training Interattivo** - Training modello in real-time
- **Visualizzazioni Dinamiche** - Grafici Plotly e insights
- **Feature Engineering Automatico** - Preprocessing dati avanzato
- **Multiple Submission** - Submission Kaggle ottimizzate
- **UI Moderna** - Dashboard Streamlit responsive

### Risultati

| Submission | Score Kaggle | Modello |
|------------|--------------|---------|
| Advanced Features | **0.77751** | Random Forest (200 alberi) |
| Optimized | 0.75598 | XGBoost ottimizzato |
| Base | 0.61722 | Random Forest base |

### Metriche

- Accuratezza: 84.9%
- Precisione: 82.2%
- Recall: 81.1%

### Stack Tecnologico

| Categoria | Tecnologie |
|-----------|------------|
| Core | Python, Pandas, NumPy, Scikit-learn, XGBoost |
| Visualizzazione | Streamlit, Plotly, Seaborn |
| Development | Git, Virtual Env, Jupyter |

### Avvio Rapido

```bash
git clone https://github.com/fracabu/Titanic_ML.git
cd Titanic_ML

python -m venv venv
venv\Scripts\activate  # Windows

pip install -r requirements.txt
streamlit run titanic_app.py
```

---

## Project Structure

```
TITANIC_ML/
├── titanic_app.py              # Streamlit app
├── generate_submission.py      # Base script
├── generate_submission_adv.py  # Advanced features
├── generate_submission_pro.py  # Optimization
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── train_cleaned.csv
└── models/
    └── xgboost_model.pkl
```

## Requirements

- Python 3.8+
- 4GB+ RAM

## License

MIT

---

<p align="center">
  <a href="https://github.com/fracabu">
    <img src="https://img.shields.io/badge/Made_by-fracabu-8B5CF6?style=flat-square" alt="Made by fracabu" />
  </a>
</p>
