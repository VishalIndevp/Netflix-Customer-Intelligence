# 🎬 ChurnLens — Netflix Customer Churn Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> **Predict whether a Netflix customer will churn — before it happens.**  
> An AI-powered web app built with Streamlit and XGBoost, featuring a cinematic dark UI.

---

## 📸 Preview

```
┌─────────────────────────────────────────────────────────────┐
│  CHURNLENS                    PREDICT CHURN BEFORE IT        │
│  Netflix Customer Intelligence  HAPPENS.                     │
│                                                              │
│  👤 Demographics               [ XGB ] [ 12 ] [ <10ms ]     │
│  📦 Subscription               ─────────────────────────    │
│  🌍 Location & Device          ⚠ LIKELY TO CANCEL           │
│  🎬 Viewing Behavior           Churn Probability: 78.3%      │
│                                ████████████████░░░░          │
│  ⚡ ANALYZE CUSTOMER           Risk Level: CRITICAL          │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

- 🤖 **XGBoost ML Model** — trained on 12 behavioral & demographic signals
- ⚡ **Real-time Prediction** — sub-10ms inference with probability score
- 🎨 **Cinematic Dark UI** — custom CSS with Netflix-inspired aesthetics
- 📊 **Risk Level Badges** — LOW / MODERATE / HIGH / CRITICAL classification
- 👤 **Customer Snapshot** — live summary card of all inputs
- 📱 **Responsive Layout** — wide two-column layout with sidebar inputs

---

## 🧠 How It Works

The model analyzes **12 input features** and outputs a churn probability score:

| Category | Features |
|---|---|
| **Demographics** | Age, Gender |
| **Subscription** | Plan Type, Monthly Fee, Number of Profiles, Payment Method |
| **Location & Device** | Region, Primary Device |
| **Viewing Behavior** | Watch Hours/Month, Avg Watch Time/Day, Days Since Last Login, Favorite Genre |

### Prediction Pipeline

```
User Input → One-Hot Encoding → Column Alignment → Standard Scaling → XGBoost → Probability Score
```

- **Churn (1)** → Red result card with churn probability %
- **Stay (0)** → Green result card with loyalty confidence %

---

## 🗂️ Project Structure

```
churnlens/
│
├── app.py                 # Main Streamlit application
├── model_XGB.pkl          # Trained XGBoost model
├── scaler.pkl             # Fitted StandardScaler
├── model_columns.pkl      # Encoded column structure
└── README.md              # You are here
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/VishalIndevp/churnlens.git
cd churnlens
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
streamlit
pandas
scikit-learn
xgboost
joblib
```

### 3. Add Model Files

Make sure the following files are in the root directory:

```
model_XGB.pkl
scaler.pkl
model_columns.pkl
```

> These are generated during model training. See the training notebook if included.

### 4. Run the App

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.8+** | Core language |
| **Streamlit** | Web UI framework |
| **XGBoost** | Churn prediction model |
| **Scikit-learn** | Preprocessing (StandardScaler, encoding) |
| **Pandas** | Data manipulation |
| **Joblib** | Model serialization |
| **Font Awesome 6** | Social icons in footer |
| **Google Fonts** | Bebas Neue + DM Sans typography |

---

## 📊 Model Details

- **Algorithm:** XGBoost Classifier
- **Input:** 12 features (after one-hot encoding, column count varies)
- **Output:** Binary classification (0 = Stay, 1 = Churn) + probability score
- **Preprocessing:** StandardScaler on numerical features, One-Hot Encoding on categoricals

---

## 🎨 UI Highlights

- Deep `#080a0f` background with red radial glow
- **Bebas Neue** display font for headlines
- **DM Sans** for body text
- Animated shimmer progress bar on prediction results
- Risk-level chips with contextual color coding
- Full sidebar input panel with sectioned categories

---

## 🤝 Connect With Me

If you found this project useful, let's connect!

[![LinkedIn](https://img.shields.io/badge/LinkedIn-vishal--singh--here-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/vishal-singh-here/)
[![X](https://img.shields.io/badge/X-vishalindev-000000?style=flat-square&logo=x)](https://x.com/vishalindev)
[![GitHub](https://img.shields.io/badge/GitHub-VishalIndevp-6e5494?style=flat-square&logo=github)](https://github.com/VishalIndevp)
[![Instagram](https://img.shields.io/badge/Instagram-vishalindev-E4405F?style=flat-square&logo=instagram)](https://www.instagram.com/vishalindev)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

<p align="center">
  Made with ❤️ by <strong>Vishal Singh</strong>
</p>
