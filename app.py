import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Настройка страницы приложения
st.set_page_config(
    page_title="Прогнозирование свойств композитов",
    page_icon="🧪",
    layout="wide"
)

# Загрузка сохраненных моделей, скейлера и списков признаков (с кэшированием для скорости)
@st.cache_resource
def load_artifacts():
    scaler = joblib.load('models/minmax_scaler.pkl')
    ml_features = joblib.load('models/ml_features.pkl')
    nn_features = joblib.load('models/nn_features.pkl')
    model_modulus = joblib.load('models/best_model_modulus.pkl')
    model_strength = joblib.load('models/best_model_strength.pkl')
    model_nn = tf.keras.models.load_model('models/nn_matrix_filler.keras')
    df_clean = pd.read_csv('data/df_clean.csv', index_col=0)
    return scaler, ml_features, nn_features, model_modulus, model_strength, model_nn, df_clean

scaler, ml_features, nn_features, model_modulus, model_strength, model_nn, df_clean = load_artifacts()
all_columns = list(df_clean.columns)

# Вспомогательная функция для нормализации пользовательского ввода через обученный MinMaxScaler
def scale_input(user_dict):
    row_df = pd.DataFrame([user_dict], columns=all_columns)
    scaled_array = scaler.transform(row_df)
    return pd.DataFrame(scaled_array, columns=all_columns)

# Заголовок приложения
st.title("🧪 Прогнозирование свойств композиционных материалов")
st.markdown(
    "Интерактивная система для расчета конечных механических характеристик композита "
    "и рекомендации рецептуры на основе моделей машинного обучения и нейронных сетей "
    "*(кейс Центра НТИ МГТУ им. Н.Э. Баумана)*."
)

# Создаем две вкладки: под Задание 1 и Задание 2
tab1, tab2 = st.tabs([
    "📊 Прогноз механических свойств (ML)",
    "🧠 Рекомендация соотношения матрица-наполнитель (Нейросеть)"
])

# ==============================================================================
# ВКЛАДКА 1: ПРОГНОЗ МОДУЛЯ УПРУГОСТИ И ПРОЧНОСТИ ПРИ РАСТЯЖЕНИИ (ЗАДАНИЕ 1)
# ==============================================================================
with tab1:
    st.subheader("Введите технологические параметры композита")
    st.caption("По умолчанию подставлены средние значения из очищенной выборки.")

    cols = st.columns(3)
    user_input_ml = {col: float(df_clean[col].mean()) for col in all_columns}

    for idx, feat in enumerate(ml_features):
        min_val = float(df_clean[feat].min())
        max_val = float(df_clean[feat].max())
        mean_val = float(df_clean[feat].mean())

        with cols[idx % 3]:
            if feat == 'Угол нашивки, град':
                user_input_ml[feat] = float(st.selectbox(feat, options=[0.0, 90.0], index=0, key=f"ml_{feat}"))
            else:
                user_input_ml[feat] = st.number_input(
                    feat, min_value=min_val * 0.5, max_value=max_val * 1.5, value=mean_val, format="%.4f", key=f"ml_{feat}"
                )

    if st.button("Рассчитать механические свойства", type="primary"):
        scaled_df = scale_input(user_input_ml)
        X_input = scaled_df[ml_features]

        pred_mod = float(model_modulus.predict(X_input).item())
        pred_str = float(model_strength.predict(X_input).item())

        st.success("Расчет успешно выполнен!")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Модуль упругости при растяжении", f"{pred_mod:.2f} ГПа")
        res_col2.metric("Прочность при растяжении", f"{pred_str:.2f} МПа")

# ==============================================================================
# ВКЛАДКА 2: РЕКОМЕНДАЦИЯ СООТНОШЕНИЯ МАТРИЦА-НАПОЛНИТЕЛЬ (ЗАДАНИЕ 2)
# ==============================================================================
with tab2:
    st.subheader("Введите целевые свойства и параметры армирования")
    st.caption("Нейронная сеть рассчитает рекомендуемое соотношение матрица-наполнитель.")

    cols_nn = st.columns(3)
    user_input_nn = {col: float(df_clean[col].mean()) for col in all_columns}

    for idx, feat in enumerate(nn_features):
        min_val = float(df_clean[feat].min())
        max_val = float(df_clean[feat].max())
        mean_val = float(df_clean[feat].mean())

        with cols_nn[idx % 3]:
            if feat == 'Угол нашивки, град':
                user_input_nn[feat] = float(st.selectbox(feat, options=[0.0, 90.0], index=0, key=f"nn_{feat}"))
            else:
                user_input_nn[feat] = st.number_input(
                    feat, min_value=min_val * 0.5, max_value=max_val * 1.5, value=mean_val, format="%.4f", key=f"nn_{feat}"
                )

    if st.button("Рекомендовать соотношение", type="primary"):
        scaled_df = scale_input(user_input_nn)
        X_input_nn = scaled_df[nn_features].values

        pred_ratio = float(model_nn.predict(X_input_nn, verbose=0).item())

        st.success("Рекомендация сформирована!")
        st.metric("Рекомендуемое соотношение матрица-наполнитель", f"{pred_ratio:.4f}")
