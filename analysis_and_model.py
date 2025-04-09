import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# Инициализируем session_state, если его нет
if "best_model_name" not in st.session_state:
    st.session_state.best_model_name = None
if "results" not in st.session_state:
    st.session_state.results = {}
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}

def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
    data['Type'] = LabelEncoder().fit_transform(data['Type'])
    return data

def analysis_and_model_page():
    st.title("Анализ данных и предсказание отказов")

    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")

    if uploaded_file is not None:
        data = load_data(uploaded_file)

        # Разделение данных
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Масштабирование данных
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            "SVM": SVC(kernel='linear', probability=True, random_state=42)
        }

        best_roc_auc = 0  # Лучшее значение ROC-AUC
        st.session_state.results.clear()
        st.session_state.trained_models.clear()

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)

            st.session_state.results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "conf_matrix": confusion_matrix(y_test, y_pred),
                "roc_auc": roc_auc
            }

            st.session_state.trained_models[name] = model  # Сохраняем обученные модели

            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                st.session_state.best_model_name = name

        st.session_state.scaler = scaler  # Сохраняем масштабировщик

        st.subheader("Результаты моделей")
        for name, result in st.session_state.results.items():
            st.write(f"**{name}:** Accuracy = {result['accuracy']:.2f}, ROC-AUC = {result['roc_auc']:.2f}")
            fig, ax = plt.subplots()
            sns.heatmap(result["conf_matrix"], annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

        # 🔮 Интерфейс для предсказаний
        st.subheader("Предсказание на новых данных")
        with st.form("prediction_form"):
            type_ = st.selectbox("Тип продукта", ["L", "M", "H"])
            air_temp = st.number_input("Температура воздуха (K)")
            process_temp = st.number_input("Рабочая температура (K)")
            rotational_speed = st.number_input("Скорость вращения (rpm)")
            torque = st.number_input("Крутящий момент (Nm)")
            tool_wear = st.number_input("Износ инструмента (мин)")

            submit_button = st.form_submit_button("Предсказать")

            if submit_button:
                if st.session_state.best_model_name and st.session_state.scaler:
                    input_data = pd.DataFrame([[type_, air_temp, process_temp, rotational_speed, torque, tool_wear]],
                                              columns=X.columns)
                    input_data['Type'] = LabelEncoder().fit_transform([input_data['Type'][0]])
                    input_data = st.session_state.scaler.transform(input_data)

                    best_model = st.session_state.trained_models[st.session_state.best_model_name]
                    prediction = best_model.predict(input_data)
                    prediction_proba = best_model.predict_proba(input_data)[:, 1]

                    st.write(f"**Предсказание:** {'Отказ' if prediction[0] == 1 else 'Нет отказа'}")
                    st.write(f"**Вероятность отказа:** {prediction_proba[0]:.2f}")
                else:
                    st.warning("Сначала проведите анализ данных!")
