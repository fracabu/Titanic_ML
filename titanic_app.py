import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Configurazione pagina
st.set_page_config(
    page_title="Titanic ML Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling CSS avanzato
st.markdown("""
    <style>
    
         /* Main Container with background */
        .stApp > header {
            background-image: url("https://audiofilescontainer.blob.core.windows.net/audiocontainer/A_highly_detailed_and_artistic_depiction_of_the_Ti.jpeg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            padding: 3rem 0;
            margin-bottom: 2rem;
            height: 150px;
            
            
        }

        
        /* Overlay scuro semi-trasparente per leggibilità */
        .main .block-container {
            background-color: rgba(14, 17, 23, 0.8);
            padding: 2rem;
            border-radius: 10px;
        
            
          
        }
        

        .main {
            color: #ffffff;
            
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #ffffff;
            font-family: 'Helvetica Neue', sans-serif;
            
        }
        
        /* Custom Container */
        .custom-container {
            background-color: #1e2130;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
            
        }
        
        .metric-container {
    background-color: #262b3d;
    padding: 1.5rem;
    border-radius: 8px;
    text-align: center;
    margin: 0.5rem;
    display: flex;
    flex-direction: row; 
    align-items: center; 
    gap: 1rem; 
   
}

        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #00ff8c;
        }
        
        .metric-label {
            font-size: 1rem;
            color: #a3a8b8;
        }
        
        /* Buttons */
        .stButton > button {
            width: 100%;
            background-color: #00acee;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            font-weight: 600;
            transition: all 0.3s;
            
        }
        
        .stButton > button:hover {
            background-color: #0096d6;
            transform: translateY(-2px);
        }
        
        /* Sidebar */
        .sidebar .sidebar-content {
            background-color: #1e2130;
        }
        
        /* Progress Bar */
        .stProgress > div > div > div {
            background-color: #00acee;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background-color: #262b3d;
            padding: 0.75rem 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 45px;
            background-color: transparent;
            border: none;
            color: #ffffff;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            color: #00acee;
        }
        
        /* File Uploader */
        .uploadedFile {
            background-color: #262b3d;
            border-radius: 5px;
            padding: 1rem;
        }
        
        /* DataFrame */
        .dataframe {
            font-family: 'Helvetica Neue', sans-serif;
            font-size: 0.9rem;
            background-color: #262b3d;
            border-radius: 5px;
        }
        
        /* Plots */
        .plot-container {
            background-color: #262b3d;
            padding: 1rem;
            border-radius: 8px;
        }
        
        /* Loading Animation */
        .stSpinner > div {
            border-top-color: #00acee !important;
            
            
        }
        # Aggiungi al tuo CSS esistente
.section-description {
    background-color: #262b3d;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    color: #a3a8b8;
    font-size: 1.1rem;
    line-height: 1.5;
}

.chart-explanation {
    background-color: #262b3d;
    padding: 1.5rem;
    border-radius: 8px;
    margin-top: 1rem;
}

.chart-explanation h4 {
    color: #00acee;
    margin-bottom: 1rem;
}

.chart-explanation p {
    color: #a3a8b8;
    line-height: 1.6;
}

.chart-explanation ul {
    margin-top: 0.5rem;
    padding-left: 1.5rem;
    color: #a3a8b8;
}

.chart-explanation li {
    margin: 0.5rem 0;
}
    </style>
""", unsafe_allow_html=True)

# Funzioni di utilità
@st.cache_data
def load_data(train_path, test_path):
    """Carica e prepara i dataset"""
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        return train_data, test_data
    except Exception as e:
        st.error(f"Errore nel caricamento dei dati: {str(e)}")
        return None, None

def feature_engineering(dataset):
    """Feature engineering avanzato"""
    df = dataset.copy()
    
    # Titoli
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 'Mr',
        'Mrs': 'Mrs',
        'Miss': 'Miss',
        'Master': 'Master',
        'Dr': 'Rare',
        'Rev': 'Rare',
        'Col': 'Rare',
        'Major': 'Rare',
        'Mlle': 'Miss',
        'Countess': 'Rare',
        'Ms': 'Miss',
        'Lady': 'Rare',
        'Jonkheer': 'Rare',
        'Don': 'Rare',
        'Dona': 'Rare',
        'Mme': 'Mrs',
        'Capt': 'Rare',
        'Sir': 'Rare'
    }
    df['Title'] = df['Title'].map(title_mapping)
    
    # Features familiari
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Gestione età
    age_mapping = {
        'Title': {
            'Mr': df[df['Title'] == 'Mr']['Age'].median(),
            'Mrs': df[df['Title'] == 'Mrs']['Age'].median(),
            'Miss': df[df['Title'] == 'Miss']['Age'].median(),
            'Master': df[df['Title'] == 'Master']['Age'].median(),
            'Rare': df[df['Title'] == 'Rare']['Age'].median()
        }
    }
    df['Age'] = df.apply(lambda x: age_mapping['Title'][x['Title']] if pd.isnull(x['Age']) else x['Age'], axis=1)
    
    # Categorie età
    df['AgeGroup'] = pd.cut(df['Age'], 
                           bins=[0, 12, 18, 35, 50, 65, 100],
                           labels=['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior', 'Elderly'])
    
    # Features cabina
    df['CabinLetter'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'U')
    df['Deck'] = df['CabinLetter'].map({'A': 'ABC', 'B': 'ABC', 'C': 'ABC',
                                       'D': 'DE', 'E': 'DE',
                                       'F': 'FG', 'G': 'FG',
                                       'T': 'Other', 'U': 'Unknown'})
    
    # Gestione Embarked e Fare
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'), inplace=True)
    
    # Fare categories
    df['FareCategory'] = pd.qcut(df['Fare'], 4, labels=['Low', 'Medium', 'High', 'Very High'])
    
    return df

def create_visualizations(data):
    """Crea visualizzazioni interattive"""
    
    # Sopravvivenza per classe
    fig_class = px.bar(data.groupby('Pclass')['Survived'].mean().reset_index(),
                      x='Pclass', y='Survived',
                      title='Tasso di Sopravvivenza per Classe',
                      labels={'Survived': 'Tasso di Sopravvivenza', 'Pclass': 'Classe'},
                      color='Pclass')
    
    # Sopravvivenza per sesso
    fig_sex = px.pie(data, names='Sex', values='Survived',
                    title='Distribuzione Sopravvivenza per Sesso')
    
    # Distribuzione età
    fig_age = px.histogram(data, x='Age', color='Survived',
                          title='Distribuzione Età per Sopravvivenza',
                          nbins=30)
    
    return fig_class, fig_sex, fig_age

def train_model(X_train, y_train, params, progress_bar=None):
    """Addestra il modello con parametri dinamici"""
    model = XGBClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        random_state=None  # Rimuovo per variabilità
    )
    
    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
        X_train, y_train, test_size=0.2
    )
    
    if progress_bar is not None:
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
    
    model.fit(X_train_split, y_train_split)
    return model, X_valid_split, y_valid_split

def evaluate_model(model, X_valid, y_valid):
    """Valuta il modello e genera metriche"""
    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    conf_matrix = confusion_matrix(y_valid, y_pred)
    class_report = classification_report(y_valid, y_pred, output_dict=True)
    return accuracy, conf_matrix, class_report

# Header principale
st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h1 style='color: #ffffff; font-size: 3rem; margin-bottom: 1rem;'>🚢 Titanic Survival Predictor</h1>
        <p style='color: #a3a8b8; font-size: 1.2rem;'>Un'applicazione avanzata di Machine Learning per predire la sopravvivenza sul Titanic</p>
    </div>
""", unsafe_allow_html=True)

# Tabs principali
tabs = st.tabs(["🎯 Predizione", "📊 Analisi", "📖 Guida"])

# Tab Predizione
with tabs[0]:
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("""
            <div class='custom-container'>
                <h3>Configurazione</h3>
        """, unsafe_allow_html=True)
        
        train_file = st.file_uploader("Dataset Training", type="csv")
        test_file = st.file_uploader("Dataset Test", type="csv")
        
        if train_file and test_file:
            st.success("File caricati con successo!")
            
            with st.expander("Parametri Avanzati"):
                n_estimators = st.slider("Numero Estimatori", 100, 500, 200)
                max_depth = st.slider("Profondità Massima", 3, 10, 5)
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[0.001, 0.01, 0.1],
                    value=0.01
                )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        if train_file and test_file:
            train_data, test_data = load_data(train_file, test_file)
            
            if train_data is not None:
                # Preprocessing
                train_processed = feature_engineering(train_data)
                test_processed = feature_engineering(test_data)
                
                # Features
                features = ['Pclass', 'Sex', 'Age', 'Fare', 'Title', 'FamilySize', 
                          'IsAlone', 'Deck', 'FareCategory', 'AgeGroup']
                
                # Preparazione dati
                X_train = pd.get_dummies(train_processed[features])
                X_test = pd.get_dummies(test_processed[features])
                y_train = train_processed['Survived']
                
                # Allineamento colonne
                missing_cols = set(X_train.columns) - set(X_test.columns)
                for col in missing_cols:
                    X_test[col] = 0
                X_test = X_test[X_train.columns]
                
                # Scaling
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                if st.button("Addestra Modello", key='train_button'):
                    st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Training
                    status_text.text('Addestramento del modello in corso...')
                    params = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'learning_rate': learning_rate
}
                    model, X_valid, y_valid = train_model(X_train_scaled, y_train, params, progress_bar)
                    
                    # Evaluation
                    status_text.text('Valutazione del modello...')
                    accuracy, conf_matrix, class_report = evaluate_model(model, X_valid, y_valid)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("""
                            <div class='metric-container'>
                                <div class='metric-value'>{:.1%}</div>
                                <div class='metric-label'>Accuratezza</div>
                            </div>
                        """.format(accuracy), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                            <div class='metric-container'>
                                <div class='metric-value'>{:.1%}</div>
                                <div class='metric-label'>Precisione</div>
                            </div>
                        """.format(class_report['1']['precision']), unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown("""
                            <div class='metric-container'>
                                <div class='metric-value'>{:.1%}</div>
                                <div class='metric-label'>Recall</div>
                            </div>
                        """.format(class_report['1']['recall']), unsafe_allow_html=True)
                    
                    # Visualizzazione matrice di confusione
                    st.markdown("### Matrice di Confusione")
                    fig_conf = px.imshow(conf_matrix,
                                       labels=dict(x="Predetto", y="Reale"),
                                       x=['Non Sopravvissuto', 'Sopravvissuto'],
                                       y=['Non Sopravvissuto', 'Sopravvissuto'],
                                       color_continuous_scale='Blues')
                    st.plotly_chart(fig_conf, use_container_width=True)
                    
                    # Feature Importance
                    st.markdown("### Importanza delle Features")
                    importance = pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    fig_imp = px.bar(importance.head(10),
                                   x='importance',
                                   y='feature',
                                   orientation='h',
                                   title='Top 10 Feature più Importanti')
                    st.plotly_chart(fig_imp, use_container_width=True)
                    
                    # Generazione predizioni
                    predictions = model.predict(X_test_scaled)
                    submission = pd.DataFrame({
                        'PassengerId': test_processed['PassengerId'],
                        'Survived': predictions
                    })
                    
                    # Download predizioni
                    st.markdown("### Download Predizioni")
                    csv = submission.to_csv(index=False)
                    st.download_button(
                        label="📥 Scarica Predizioni CSV",
                        data=csv,
                        file_name="titanic_predictions.csv",
                        mime="text/csv",
                    )
                    
                    st.markdown("</div>", unsafe_allow_html=True)

# Tab Analisi
with tabs[1]:
    if train_file is not None and 'train_data' in locals():
        st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
        st.markdown("## 📊 Analisi Esplorativa dei Dati")
        
        # Metriche generali con spiegazioni
        st.markdown("""
            <div class='section-description'>
                <h4>🔍 Overview del Dataset</h4>
                Le seguenti metriche forniscono una panoramica generale dei passeggeri del Titanic, 
                permettendo di comprendere rapidamente la composizione demografica e il tasso di sopravvivenza complessivo.
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
                <div class='metric-container'>
                    <div class='metric-value'>{}</div>
                    <div class='metric-label'>Totale Passeggeri</div>
                </div>
            """.format(len(train_data)), unsafe_allow_html=True)
        
        with col2:
            survival_rate = train_data['Survived'].mean()
            st.markdown("""
                <div class='metric-container'>
                    <div class='metric-value'>{:.1%}</div>
                    <div class='metric-label'>Tasso di Sopravvivenza</div>
                </div>
            """.format(survival_rate), unsafe_allow_html=True)
        
        with col3:
            avg_age = train_data['Age'].mean()
            st.markdown("""
                <div class='metric-container'>
                    <div class='metric-value'>{:.1f}</div>
                    <div class='metric-label'>Età Media</div>
                </div>
            """.format(avg_age), unsafe_allow_html=True)

        # Statistiche dettagliate
        st.markdown("### 📈 Statistiche Dettagliate")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class='stat-box'>
                    <h4>👥 Composizione Passeggeri</h4>
                    <ul>
                        <li>Uomini: {:.1%}</li>
                        <li>Donne: {:.1%}</li>
                        <li>Bambini (<12 anni): {:.1%}</li>
                    </ul>
                </div>
            """.format(
                len(train_data[train_data['Sex'] == 'male'])/len(train_data),
                len(train_data[train_data['Sex'] == 'female'])/len(train_data),
                len(train_data[train_data['Age'] < 12])/len(train_data)
            ), unsafe_allow_html=True)

        with col2:
            st.markdown("""
                <div class='stat-box'>
                    <h4>💰 Classi Passeggeri</h4>
                    <ul>
                        <li>Prima Classe: {:.1%}</li>
                        <li>Seconda Classe: {:.1%}</li>
                        <li>Terza Classe: {:.1%}</li>
                    </ul>
                </div>
            """.format(
                len(train_data[train_data['Pclass'] == 1])/len(train_data),
                len(train_data[train_data['Pclass'] == 2])/len(train_data),
                len(train_data[train_data['Pclass'] == 3])/len(train_data)
            ), unsafe_allow_html=True)

        # Visualizzazioni
        st.markdown("### 📊 Analisi della Sopravvivenza")
        
        # Creazione visualizzazioni
        fig_class, fig_sex, fig_age = create_visualizations(train_data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_class, use_container_width=True)
            st.markdown("""
                <div class='chart-explanation'>
                    <h4>💡 Sopravvivenza per Classe</h4>
                    <p>L'analisi mostra una chiara correlazione tra classe e sopravvivenza:</p>
                    <ul>
                        <li>La prima classe ha il più alto tasso di sopravvivenza ({:.1%})</li>
                        <li>La terza classe ha il più basso tasso di sopravvivenza ({:.1%})</li>
                        <li>La posizione delle cabine e l'accesso ai canotti di salvataggio hanno influenzato significativamente la sopravvivenza</li>
                    </ul>
                </div>
            """.format(
                train_data[train_data['Pclass'] == 1]['Survived'].mean(),
                train_data[train_data['Pclass'] == 3]['Survived'].mean()
            ), unsafe_allow_html=True)
        
        with col2:
            st.plotly_chart(fig_sex, use_container_width=True)
            st.markdown("""
                <div class='chart-explanation'>
                    <h4>💡 Analisi per Genere</h4>
                    <p>Il genere è stato un fattore determinante per la sopravvivenza:</p>
                    <ul>
                        <li>Donne: {:.1%} tasso di sopravvivenza</li>
                        <li>Uomini: {:.1%} tasso di sopravvivenza</li>
                        <li>La politica "donne e bambini prima" è chiaramente riflessa nei dati</li>
                    </ul>
                </div>
            """.format(
                train_data[train_data['Sex'] == 'female']['Survived'].mean(),
                train_data[train_data['Sex'] == 'male']['Survived'].mean()
            ), unsafe_allow_html=True)

        # Distribuzione età
        st.markdown("### 👥 Analisi Demografica")
        st.plotly_chart(fig_age, use_container_width=True)
        st.markdown("""
            <div class='chart-explanation'>
                <h4>💡 Distribuzione dell'Età e Sopravvivenza</h4>
                <p>L'analisi dell'età rivela pattern significativi:</p>
                <ul>
                    <li>Bambini (0-10 anni): maggiore probabilità di sopravvivenza</li>
                    <li>Fascia 20-40 anni: la più numerosa tra i passeggeri</li>
                    <li>Età media dei sopravvissuti: {:.1f} anni</li>
                    <li>Età media dei non sopravvissuti: {:.1f} anni</li>
                </ul>
            </div>
        """.format(
            train_data[train_data['Survived'] == 1]['Age'].mean(),
            train_data[train_data['Survived'] == 0]['Age'].mean()
        ), unsafe_allow_html=True)

        # Correlazioni con spiegazione
        st.markdown("### 🔗 Analisi delle Correlazioni")
        numeric_cols = train_data.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = train_data[numeric_cols].corr()
        fig_corr = px.imshow(corr_matrix,
                            labels=dict(color="Correlazione"),
                            color_continuous_scale='RdBu')
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("""
            <div class='chart-explanation'>
                <h4>💡 Interpretazione delle Correlazioni</h4>
                <p>La matrice di correlazione evidenzia relazioni importanti:</p>
                <ul>
                    <li>Forte correlazione negativa tra Classe e Tariffa (-0.55)</li>
                    <li>Correlazione positiva tra SibSp e Parch (0.35)</li>
                    <li>La sopravvivenza è più correlata con:</li>
                    <ul>
                        <li>Classe: correlazione negativa (-0.34)</li>
                        <li>Tariffa: correlazione positiva (0.26)</li>
                        <li>Età: leggera correlazione negativa (-0.07)</li>
                    </ul>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Carica il dataset di training per visualizzare l'analisi dei dati.")
        
# Tab Guida
with tabs[2]:
    st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
    st.markdown("""
    # 📖 Documentazione Tecnica Titanic Predictor

    ## 🛠️ Stack Tecnologico

    ### Librerie Core
    - **Streamlit (v1.32.0)**
      - Framework per l'interfaccia web
      - Componenti interattivi
      - Gestione stato applicazione
      
    - **Pandas (v2.2.0)**
      - Manipolazione dataset
      - Feature engineering
      - Analisi dati
      
    - **NumPy (v1.26.3)**
      - Operazioni matematiche
      - Manipolazione array
      - Supporto calcolo numerico
      
    - **Scikit-learn (v1.4.0)**
      - Preprocessing dati
      - Train-test split
      - Metriche valutazione
      
    - **XGBoost (v2.0.3)**
      - Algoritmo di classificazione
      - Gestione automatica valori mancanti
      - Feature importance

    ### Librerie Visualizzazione
    - **Plotly (v5.18.0)**
      - Grafici interattivi
      - Matrice di confusione
      - Feature importance plot
      
    - **Matplotlib (v3.8.2)**
      - Visualizzazioni statiche
      - Supporto grafici base
      
    - **Seaborn (v0.13.1)**
      - Visualizzazioni statistiche
      - Heatmap correlazioni

    ## 🔍 Funzionalità Dettagliate

    ### 1. Preprocessing Dati
    
    #### Feature Engineering
    ```python
    - Title Extraction: Nome → Mr, Mrs, Miss, etc.
    - Family Features: SibSp + Parch + 1 → FamilySize
    - Age Categories: [0-12, 12-18, 18-35, 35-50, 50-65, 65+]
    - Fare Categories: Quartili → [Low, Medium, High, Very High]
    - Cabin Processing: Prima lettera → Deck category
    ```

    #### Gestione Valori Mancanti
    - **Age**: Mediana per titolo
    - **Embarked**: Moda
    - **Fare**: Mediana per classe
    - **Cabin**: Categoria 'U' per sconosciuto

    ### 2. Modello ML

    #### Parametri XGBoost
    | Parametro | Range | Default | Effetto |
    |-----------|--------|---------|---------|
    | n_estimators | 100-1000 | 200 | Numero di alberi |
    | max_depth | 1-15 | 5 | Profondità alberi |
    | learning_rate | 0.001-0.3 | 0.01 | Velocità apprendimento |

    #### Ottimizzazione
    - **Underfitting**: Aumentare n_estimators/max_depth
    - **Overfitting**: Ridurre max_depth, aumentare min_child_weight
    - **Performance**: Bilanciare learning_rate e n_estimators

    ### 3. Metriche Valutazione

    #### Principali
    ```python
    - Accuratezza = (VP + VN) / (VP + VN + FP + FN)
    - Precisione = VP / (VP + FP)
    - Recall = VP / (VP + FN)
    ```
    Dove:
    - VP: Veri Positivi
    - VN: Veri Negativi
    - FP: Falsi Positivi
    - FN: Falsi Negativi

    ## 📋 Guida Operativa

    ### 1. Preparazione Dataset
    
    #### Formato CSV Richiesto
    ```csv
    PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
    ```

    #### Requisiti File
    - Encoding: UTF-8
    - Separatore: Virgola
    - Header: Richiesto
    - Dimensione: < 200MB

    ### 2. Workflow Analisi

    #### Step 1: Caricamento Dati
    1. Upload `train.csv`
    2. Upload `test.csv`
    3. Verifica preview dati
    4. Controllo tipi colonne

    #### Step 2: Feature Engineering
    1. Esecuzione automatica
    2. Verifica nuove features
    3. Analisi distribuzioni

    #### Step 3: Training
    1. Configurazione parametri
    2. Avvio training
    3. Monitoraggio metriche

    #### Step 4: Valutazione
    1. Analisi metriche
    2. Studio feature importance
    3. Verifica predizioni

    ## ❓ FAQ Tecniche

    ### Modello
    
    **Q: Perché XGBoost?**
    A: XGBoost offre:
    - Gestione automatica valori mancanti
    - Ottima performance su dataset strutturati
    - Feature importance nativa
    - Efficiente su CPU

    **Q: Come gestire l'overfitting?**
    A: Strategie:
    1. Ridurre max_depth
    2. Aumentare min_child_weight
    3. Usare early_stopping
    4. Implementare cross-validation

    **Q: Come migliorare l'accuratezza?**
    A: Approcci:
    1. Feature engineering avanzato
    2. Tuning iperparametri
    3. Ensemble di modelli
    4. Analisi errori

    ### Performance

    **Q: Ottimizzazione tempi?**
    A: Suggerimenti:
    1. Ridurre n_estimators
    2. Limitare max_depth
    3. Usare subset di features
    4. Implementare caching

    **Q: Gestione memoria?**
    A: Best practices:
    1. Ridurre precisione numerica
    2. Eliminare colonne inutili
    3. Usare chunking per dataset grandi
    4. Implementare garbage collection

    ## 🔧 Troubleshooting

    ### Errori Comuni

    #### 1. Caricamento Dati
    ```text
    Errore: "Invalid CSV format"
    - Verifica encoding
    - Controlla separatore
    - Valida header
    ```

    #### 2. Training
    ```text
    Errore: "Memory Error"
    - Riduci dimensioni batch
    - Ottimizza parametri
    - Libera memoria
    ```

    #### 3. Predizioni
    ```text
    Errore: "Feature mismatch"
    - Allinea colonne
    - Verifica preprocessing
    - Controlla encoding
    ```

    ## 📊 Interpretazione Risultati

    ### Matrice di Confusione
    - **Veri Positivi**: Sopravvissuti correttamente predetti
    - **Veri Negativi**: Non sopravvissuti correttamente predetti
    - **Falsi Positivi**: Predetti sopravvissuti ma non realmente
    - **Falsi Negativi**: Predetti non sopravvissuti ma realmente sopravvissuti

    ### Feature Importance
    - **Sex**: Genere (tipicamente più importante)
    - **Pclass**: Classe biglietto
    - **Age**: Età passeggero
    - **Fare**: Prezzo biglietto
    - **Title**: Titolo estratto dal nome

    ## 🔐 Sicurezza e Limitazioni

    ### Sicurezza Dati
    - Nessun dato salvato permanentemente
    - Processing in-memory
    - Cache temporanea
    - No dati sensibili

    ### Limitazioni Tecniche
    - Max file size: 200MB
    - Formato: Solo CSV
    - RAM richiesta: 4GB+
    - CPU: Single-thread
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #a3a8b8;'>
        <p>Developed with ❤️ using Streamlit and Python</p>
        <p>© 2024 Titanic Survival Predictor</p>
    </div>
""", unsafe_allow_html=True)