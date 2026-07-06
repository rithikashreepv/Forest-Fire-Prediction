# Forest-Fire-Prediction
Machine Learning project for predic# 🌲 Forest Fire Burned Area Prediction

A Machine Learning web application that predicts the estimated burned area of a forest fire using meteorological conditions and fire weather indices.

The application is built using **Scikit-learn**, **Gradio**, and **Random Forest Regression**, and is deployed as an interactive web application.

---

## 📌 Project Overview

Forest fires can have significant environmental and economic impacts. This project predicts the expected burned area of a forest fire using weather conditions and fire weather indices from the UCI Forest Fires dataset.

The project demonstrates an end-to-end machine learning workflow including:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training
- Model comparison
- Hyperparameter tuning
- Model deployment using Gradio

---

## 📊 Dataset

**Dataset:** UCI Machine Learning Repository – Forest Fires Dataset

The dataset contains **517 observations** and **13 attributes**.

### Features

| Feature | Description |
|----------|-------------|
| X | X-axis spatial coordinate |
| Y | Y-axis spatial coordinate |
| month | Month of the year |
| day | Day of the week |
| FFMC | Fine Fuel Moisture Code |
| DMC | Duff Moisture Code |
| DC | Drought Code |
| ISI | Initial Spread Index |
| temp | Temperature (°C) |
| RH | Relative Humidity (%) |
| wind | Wind speed |
| rain | Rainfall |
| area | Burned area (Target Variable) |

---

# ⚙️ Data Preprocessing

The following preprocessing steps were performed:

- Missing value analysis
- One-Hot Encoding for categorical variables (`month` and `day`)
- Train-Test Split
- Log Transformation of the target variable (`log1p(area)`) to reduce skewness

---

# 🤖 Machine Learning Models Evaluated

The following regression algorithms were implemented and compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regression (SVR)
- K-Nearest Neighbors (KNN)

Performance was evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

# 🔍 Hyperparameter Tuning

The Random Forest model was optimized using **GridSearchCV** with **5-Fold Cross Validation**.

Parameters tuned included:

- Number of Trees (`n_estimators`)
- Maximum Tree Depth (`max_depth`)
- Minimum Samples Split (`min_samples_split`)
- Minimum Samples per Leaf (`min_samples_leaf`)

---

# 🚀 Deployment

The final model was deployed using **Gradio**, providing an interactive web interface where users can:

- Enter weather conditions
- Select month and day
- Predict the estimated burned area of a forest fire

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Gradio
- Google Colab
- GitHub

---

# 📁 Project Structure

```
Forest-Fire-Prediction/
│
├── app.py
├── forest_fire_random_forest.pkl
├── preprocessor.pkl
├── requirements.txt
├── README.md
```

---

# ▶️ Run Locally

Clone the repository

```bash
git clone https://github.com/rithikashreepv/Forest-Fire-Prediction.git
```

Navigate to the project directory

```bash
cd Forest-Fire-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

---

# 📈 Future Improvements

- Improve prediction accuracy using ensemble boosting models (e.g., XGBoost or LightGBM)
- Perform additional feature engineering
- Add interactive visualizations
- Deploy with Docker
- Integrate real-time weather APIs



---

## ⭐ If you found this project useful, consider giving it a star!ting forest fire burned area using Random Forest and Gradio
