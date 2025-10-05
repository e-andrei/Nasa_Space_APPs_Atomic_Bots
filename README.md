# 🚀 Exoplanet AI - Manager de Modele cu Streamlit

**Aplicație web pentru antrenarea, încărcarea și utilizarea modelelor de clasificare a exoplanetelor - totul în browser, fără backend!**

## ✨ Funcționalități Principale

### 🔄 Workflow Complet în Browser
- **📤 Încarcă propriile modele** - Upload direct de fișiere `.joblib` 
- **🛠️ Antrenează modele noi** - Cu fișierele CSV pe care le introduci
- **💾 Descarcă rezultatele** - Modele antrenate și predicții direct în browser
- **🔮 Predicții avansate** - Pe fișiere CSV sau introducere manuală

### 🎯 Fără Backend Necesar
- Totul rulează în Streamlit - nu e nevoie de server separat
- Stocare temporară în memoria aplicației
- Descărcare directă din browser
- Funcționează local sau în cloud

## 📦 Instalare Rapidă

```bash
# 1. Clonează repository-ul
git clone https://github.com/your-username/exoplanet-ai.git
cd exoplanet-ai

# 2. Instalează dependențele
pip install -r requirements.txt

# 3. Pornește aplicația
streamlit run app/streamlit_app.py
```

Aplicația se va deschide în browser la `http://localhost:8501`

## 🎮 Cum să Folosești

### 1️⃣ Încarcă un Model

**Opțiunea A: Model Custom**
- Selectează "📤 Încarcă model custom" din sidebar
- Încarcă fișierul `.joblib` cu modelul tău
- Aplicația detectează automat caracteristicile

**Opțiunea B: Model Pre-antrenat**
- Selectează "📁 Folosește model existent"
- Alege din modelele disponibile în `models/`

### 2️⃣ Realizează Predicții

**📄 Cu Fișier CSV:**
```
Tab: 🔮 Predicții → 📄 Încarcă fișier CSV
```
1. Încarcă fișierul CSV cu datele
2. Aplicația mapează automat coloanele
3. Configurează pragul de încredere
4. Primești predicții cu probabilități
5. Descarcă rezultatele complete

**✍️ Manual:**
```
Tab: 🔮 Predicții → ✍️ Introducere manuală
```
Completează valorile pentru fiecare caracteristică

### 3️⃣ Antrenează Model Nou

```
Tab: 🛠️ Antrenare Nouă
```

1. **Încarcă date de antrenare** - Unul sau mai multe fișiere CSV
2. **Configurează parametrii:**
   - Tip model: XGBoost / Random Forest
   - Optimizare hiperparametri (opțional)
   - Greutăți pentru clase
3. **Începe antrenarea** - Aplicația procesează automat
4. **Descarcă modelul** - Direct din tab-ul "💾 Descărcări"

### 4️⃣ Descarcă Rezultatele

```
Tab: 💾 Descărcări
```

- **Modelul curent** - Descarcă modelul încărcat
- **Modelul nou antrenat** - După antrenare
- **Predicțiile** - Rezultatele ultimelor predicții
- **Metadate JSON** - Informații despre model

## 📊 Formate de Date Acceptate

### Pentru Predicții
Orice combinație din coloanele:
```csv
koi_period,koi_depth,koi_duration,koi_prad,koi_teq,koi_insol,koi_impact,koi_steff,koi_srad,koi_slogg,koi_model_snr,koi_score,koi_fpflag_nt,koi_fpflag_ss,koi_fpflag_co,koi_fpflag_ec
```

### Pentru Antrenare
Date + coloană cu adevărul de teren:
```csv
koi_period,koi_depth,koi_disposition,...
365.25,100.5,CONFIRMED,...
582.7,85.2,FALSE POSITIVE,...
```

**Coloane acceptate pentru adevărul de teren:**
- `disposition`, `koi_disposition`, `tfopwg_disp`

**Valori acceptate:**
- `CONFIRMED` / `CP` / `KP`
- `CANDIDATE` / `PC` / `APC` 
- `FALSE POSITIVE` / `FP` / `FA`

## 🔧 Caracteristici Avansate

### Mapare Automată de Coloane
Aplicația recunoaște automat formatele:
- **Kepler**: `koi_*` 
- **K2/PANDC**: `pl_*`
- **TOI**: `tfopwg_*`
- **Exoplanet Archive**: `st_*`

### Engineering de Caracteristici
- **Transit depth** calculat automat din `planet_radius` și `stellar_radius`
- **Normalizare** automată a valorilor
- **Gestionarea valorilor lipsă**

### Optimizare Modele
- **Hyperparameter tuning** cu Randomized Search
- **Cross-validation** configurabilă
- **Greutăți personalizate** pentru clase
- **Metrics detaliate** de evaluare

## 📈 Modelele Incluse

Aplicația vine cu modele pre-antrenate:

| Model | Descriere | Accuracy | F1 Score |
|-------|-----------|----------|----------|
| `unified_xgb_tuned.joblib` | XGBoost optimizat pe date multiple | ~94% | ~0.89 |
| `unified_rf_tuned.joblib` | Random Forest optimizat | ~93% | ~0.87 |
| `multi_toi_classifier.joblib` | Specialist pentru date TOI | ~92% | ~0.88 |

## 🚀 Deployment

### Local
```bash
streamlit run app/streamlit_app.py
```

### Cloud (Streamlit Cloud, Heroku, etc.)
1. Asigură-te că `models/` este inclus în deployment
2. Entry point: `streamlit run app/streamlit_app.py`
3. Variabile de mediu (dacă e nevoie):
   ```
   STREAMLIT_SERVER_PORT=8501
   STREAMLIT_SERVER_ADDRESS=0.0.0.0
   ```

## 🛠️ Dezvoltare

### Structura Proiectului
```
exoplanet-ai/
├── app/
│   ├── streamlit_app.py          # Aplicația principală
│   └── streamlit_app_backup.py   # Backup versiune anterioară
├── models/                       # Modele pre-antrenate
├── src/                         # Cod sursă pentru antrenare
├── data/                        # Date de test/demo
├── notebooks/                   # Jupyter notebooks
├── GHID_UTILIZARE.md           # Ghid detaliat în română
└── requirements.txt            # Dependențe Python
```

### Adăugarea de Modele Noi
```python
# Pentru a folosi modelele în alte aplicații
import joblib

model_data = joblib.load('your_model.joblib')
pipeline = model_data['pipeline']
label_encoder = model_data['label_encoder']
features = model_data['numeric_cols']
```

## 📚 Documentație

- **[GHID_UTILIZARE.md](GHID_UTILIZARE.md)** - Ghid complet în română
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Ghid de deployment
- **[notebooks/quickstart_tutorial.ipynb](notebooks/quickstart_tutorial.ipynb)** - Tutorial interactiv

## 🤝 Contribuții

Contribuțiile sunt binevenite! 

1. Fork repository-ul
2. Creează o branch pentru feature-ul tău
3. Commit modificările
4. Push la branch
5. Deschide un Pull Request

## 📄 Licență

Acest proiect este sub licența MIT. Vezi fișierul `LICENSE` pentru detalii.

## 🆘 Suport

### Probleme Comune

**Modelul nu se încarcă:**
- Verifică că fișierul `.joblib` este valid
- Asigură-te că ai toate dependențele instalate

**Antrenarea eșuează:**
- Verifică formatul datelor CSV
- Asigură-te că există coloana cu adevărul de teren

**Deployment în cloud:**
- Verifică că directorul `models/` este inclus
- Consultă [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

### Contact

Pentru întrebări sau probleme:
- Deschide un issue pe GitHub
- Consultă documentația din `GHID_UTILIZARE.md`

---

**🌟 Clasifică exoplanetele cu încredere și precizie! 🌟**

*Dezvoltat pentru NASA Space Apps Challenge 2025*