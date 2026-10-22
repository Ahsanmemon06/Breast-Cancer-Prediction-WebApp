# 🩺 Breast Cancer Risk Assessment & Prediction Web Application

An end-to-end Machine Learning web application designed to evaluate and predict breast cancer risk probability using demographic, lifestyle, physiological, and clinical diagnostic factors. The system is backed by a hyperparameter-tuned **Random Forest Classifier** and features a responsive, dark-mode user interface built with **Flask, HTML5, and CSS3**.

---

## 📌 Project Overview
Early detection of breast cancer plays a vital role in clinical outcomes. This project bridges medical risk factors with predictive modeling by training and evaluating multiple classification algorithms on clinical records.

To prevent critical false negatives, the final model was tuned using **RandomizedSearchCV** with class balancing to optimize **Recall (0.80)** and **F1-Score (0.81)** while sustaining an overall **Accuracy of 93%**.

---

## 🚀 Key Features
- **End-to-End Pipeline:** Data cleaning, handling missing values, label/one-hot encoding, feature scaling (`StandardScaler`), and model deployment.
- **Multi-Model Benchmark:** Evaluated Logistic Regression, KNN, Support Vector Machine (SVM), Decision Tree, and Random Forest.
- **Hyperparameter Optimization:** Utilized `RandomizedSearchCV` (`n_estimators=300`, `class_weight='balanced'`) to boost recall for cancer-positive samples.
- **Interactive Sliding UI:** A structured, multi-step card layout that groups 18 patient attributes into 5 clean assessment sections with horizontal sliding animations.
- **Risk Probability Output:** Displays risk classification along with predicted probability percentages.

---

## 📊 Model Evaluation & Benchmarks

| Model | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Overall Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **K-Nearest Neighbors (KNN)** | 0.20 | 0.06 | 0.09 | 77% |
| **Decision Tree** | 0.74 | 0.48 | 0.59 | 87% |
| **Logistic Regression** | 0.78 | 0.72 | 0.75 | 91% |
| **Support Vector Machine (SVM)** | 0.82 | 0.68 | 0.75 | 91% |
| **Random Forest (Default)** | 0.96 | 0.74 | 0.84 | 94% |
| **Random Forest (Tuned)** ⭐ | **0.82** | **0.80** | **0.81** | **93%** |

> **Note on Medical Recall:** The tuned Random Forest model was selected as the final candidate because it significantly reduced false negatives (missed cases dropped from ~97 to 75), which is crucial in clinical decision support.

---

## 🔍 Top Predictive Features
Feature importance analysis revealed that the model relies most heavily on:
1. **Tumor Size (`Tumor_Size_cm`)** — ~14% Importance
2. **Body Mass Index (`BMI`)** — ~13.7% Importance
3. **Patient Age (`Age`)** — ~12.7% Importance
4. **Family History & Smoking Status**

---

## 📂 Project Architecture

```text
Breast-Cancer-Prediction-WebApp/
│
├── static/
│   └── style.css                             # Dark-mode dashboard styles & slide transitions
├── templates/
│   └── index.html                            # 5-step interactive assessment interface
├── app.py                                    # Flask backend handling routes & inference
├── breast cancer model.pkl                   # Bundled pickle file (Model, Scaler, Feature Names)
├── breast cancer prediction project.ipynb   # Complete exploratory data analysis & training notebook
├── requirements.txt                          # Python dependencies
├── .gitignore                                # Excluded temporary files
└── README.md                                 # Project documentation
🛠️ Installation & Local Setup
1. Clone the Repository
Bash
git clone [https://github.com/Ahsanmemon06/Breast-Cancer-Prediction-WebApp.git](https://github.com/Ahsanmemon06/Breast-Cancer-Prediction-WebApp.git)
cd Breast-Cancer-Prediction-WebApp
2. Create and Activate a Virtual Environment (Optional)
Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Run the Application
Bash
python app.py
5. Access the Web Dashboard
Open your browser and visit:

Plaintext
[http://127.0.0.1:5000/](http://127.0.0.1:5000/)
💡 Tech Stack
Languages: Python

Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

Backend: Flask

Frontend: HTML5, CSS3, JavaScript (Vanilla Slider DOM)

Serialization: Pickle

👤 Author
GitHub: @Ahsanmemon06

Domain: Artificial Intelligence & Machine Learning


---

### Is file ko update karne ke steps:

1. VS Code mein `README.md` file banayein ya open karein.
2. Upar diya gaya poora markdown text usme paste karke save karein (`Ctrl + S`).
3. Terminal mein yeh 3 commands run karein:
   ```bash
   git add README.md
   git commit -m "Docs: Add comprehensive README with benchmarks and architecture"
   git push origin main
