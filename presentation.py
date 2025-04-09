import streamlit as st

def presentation_page():
    st.title("Презентация проекта")

    best_model_name = st.session_state.get("best_model_name", None)
    results = st.session_state.get("results", {})

    if results:
        best_model_text = f"**Лучшая модель:** {best_model_name}"
        avg_accuracy = sum(model["accuracy"] for model in results.values()) / len(results)
        avg_roc_auc = sum(model["roc_auc"] for model in results.values()) / len(results)

        improvements = []
        if avg_accuracy < 0.85:
            improvements.append("Использование более сложных моделей, таких как нейросети.")
        if avg_roc_auc < 0.90:
            improvements.append("Добавление новых признаков или инженерия существующих.")
        if best_model_name == "Logistic Regression":
            improvements.append("Попробовать нелинейные модели, такие как Random Forest или XGBoost.")

        improvements_text = "- " + "\n- ".join(improvements) if improvements else "Текущая модель показывает хорошие результаты, но можно провести дополнительные эксперименты."

        results_text = f"""
        **Средняя точность (Accuracy) по всем моделям:** {avg_accuracy:.2f}  
        **Средний ROC-AUC по всем моделям:** {avg_roc_auc:.2f}
        """
    else:
        best_model_text = "**Лучшая модель:** Данные ещё не загружены"
        results_text = ""
        improvements_text = "Загрузите данные, чтобы провести анализ и увидеть возможные улучшения."

    st.markdown(f"""
    ## Бинарная классификация для предиктивного обслуживания оборудования  

    **Цель:** Разработать модель машинного обучения, предсказывающую отказы оборудования.  

    ---
    ## Этапы работы
    1. **Загрузка данных**  
    2. **Предобработка** (удаление ненужных столбцов, кодирование категорий, масштабирование)  
    3. **Обучение моделей** (Logistic Regression, Random Forest, XGBoost, SVM)  
    4. **Оценка моделей** (Accuracy, ROC-AUC, Confusion Matrix)  
    5. **Streamlit-приложение**  
    ---
    ## Итоги  
    {best_model_text}  
    {results_text}  
    ---
    ## Возможные улучшения  
    {improvements_text}
    """)
