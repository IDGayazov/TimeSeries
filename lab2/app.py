import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

st.set_page_config(page_title="Прогнозирование временных рядов", layout="wide")

st.title("📈 Интерактивное прогнозирование временных рядов")

# Загрузка данных
uploaded_file = st.file_uploader("Загрузите CSV/Parquet файл", type=['csv', 'parquet'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        st.success(f"Загружено {len(df)} строк, {len(df.columns)} колонок")
        
        # Автоматически находим колонки с датами и числами
        date_columns = []
        numeric_columns = []
        
        for col in df.columns:
            # Пробуем преобразовать в дату
            try:
                pd.to_datetime(df[col])
                date_columns.append(col)
            except:
                pass
            
            # Проверяем числовые колонки
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_columns.append(col)
        
        # Выбор переменных
        col1, col2 = st.columns(2)
        with col1:
            if numeric_columns:
                target_col = st.selectbox("Целевая переменная", numeric_columns)
            else:
                st.error("❌ Нет числовых колонок в данных")
                st.stop()
                
        with col2:
            if date_columns:
                date_col = st.selectbox("Временная метка", date_columns)
                # Преобразуем дату
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
            else:
                st.warning("⚠️ Дата не найдена, использую индекс")
                date_col = None
        
        # Подготовка временного ряда
        series = df[target_col].dropna()
        
        # Настройки прогноза
        st.subheader("Настройки прогноза")
        h = st.slider("Горизонт прогнозирования", 7, 90, 30)
        
        # Декомпозиция - используем session_state для сохранения состояния
        st.subheader("Декомпозиция ряда")
        
        # Инициализация состояния
        if 'show_decomposition' not in st.session_state:
            st.session_state.show_decomposition = False
        
        # Кнопка для показа/скрытия декомпозиции
        if st.button("Показать/скрыть декомпозицию"):
            st.session_state.show_decomposition = not st.session_state.show_decomposition
        
        # Показываем декомпозицию если включено
        if st.session_state.show_decomposition:
            decomp_type = st.radio("Тип декомпозиции", ["Аддитивная", "Мультипликативная"])
            
            try:
                # Выбираем период для сезонности
                period = st.slider("Период сезонности", 7, 365, 30)
                
                if len(series) >= 2 * period:
                    decomposition = seasonal_decompose(series, 
                                                     model='additive' if decomp_type == "Аддитивная" else 'multiplicative', 
                                                     period=period)
                    
                    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
                    decomposition.observed.plot(ax=axes[0], title='Исходный ряд')
                    decomposition.trend.plot(ax=axes[1], title='Тренд')
                    decomposition.seasonal.plot(ax=axes[2], title='Сезонность')
                    decomposition.resid.plot(ax=axes[3], title='Остатки')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning(f"⚠️ Для декомпозиции нужно минимум {2 * period} точек, доступно {len(series)}")
                    
            except Exception as e:
                st.error(f"Ошибка декомпозиции: {e}")

        # Прогнозирование
        st.subheader("Построение прогнозов")
        
        # Используем session_state для прогнозирования
        if 'show_forecasts' not in st.session_state:
            st.session_state.show_forecasts = False
        
        if st.button("Запустить/остановить прогнозирование"):
            st.session_state.show_forecasts = not st.session_state.show_forecasts
        
        if st.session_state.show_forecasts:
            if len(series) < 30:
                st.error("❌ Слишком мало данных для прогнозирования")
                st.stop()
                
            train_size = int(len(series) * 0.8)
            train = series.iloc[:train_size]
            test = series.iloc[train_size:train_size+h]
            
            models = {}
            
            # Наивный прогноз
            models['Naive'] = np.full(len(test), train.iloc[-1])
            
            # Экспоненциальное сглаживание
            try:
                with st.spinner('Обучаем модели...'):
                    ses = ExponentialSmoothing(train, trend=None, seasonal=None).fit()
                    models['SES'] = ses.forecast(len(test))
                    
                    holt_add = ExponentialSmoothing(train, trend='add', seasonal=None).fit()
                    models['Holt Add'] = holt_add.forecast(len(test))
                    
                    if train.min() > 0:
                        holt_mul = ExponentialSmoothing(train, trend='mul', seasonal=None).fit()
                        models['Holt Mul'] = holt_mul.forecast(len(test))
                        
            except Exception as e:
                st.error(f"Ошибка в моделях сглаживания: {e}")

            # Визуализация прогнозов
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Последние 100 точек обучающих данных для контекста
            plot_start = max(0, len(train) - 100)
            ax.plot(train.index[plot_start:], train.iloc[plot_start:], 'b-', label='История', alpha=0.7)
            ax.plot(test.index, test.values, 'ko-', label='Факт', linewidth=2)
            
            for name, pred in models.items():
                ax.plot(test.index, pred, 'o--', label=name, alpha=0.7, markersize=4)
            
            ax.legend()
            ax.set_title('Сравнение прогнозов моделей')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Таблица метрик
            metrics_data = []
            for name, pred in models.items():
                actual = test.values
                mae = np.mean(np.abs(actual - pred))
                rmse = np.sqrt(np.mean((actual - pred)**2))
                
                metrics_data.append({
                    'Модель': name,
                    'MAE': f"{mae:.2f}",
                    'RMSE': f"{rmse:.2f}"
                })
            
            st.subheader("Сравнение метрик качества")
            st.table(pd.DataFrame(metrics_data))
            
            # Экспорт результатов
            st.subheader("Экспорт результатов")
            results_df = pd.DataFrame({
                'date': test.index,
                'actual': test.values
            })
            
            for name, pred in models.items():
                results_df[name] = pred
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Скачать прогнозы (CSV)",
                data=csv,
                file_name="forecasts.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ Ошибка обработки файла: {e}")

else:
    st.info("👆 Загрузите файл для начала анализа")

with st.sidebar:
    st.header("📋 Инструкция")
    st.markdown("""
    1. **Загрузите** CSV/Parquet файл
    2. **Выберите** целевую переменную
    3. **Настройте** горизонт прогноза
    4. **Нажмите кнопку** для декомпозиции
    5. **Нажмите кнопку** для прогнозирования
    6. **Сравните** результаты и экспортируйте
    """)
    
    st.header("📊 Поддерживаемые модели")
    st.markdown("""
    - Naive (последнее значение)
    - Simple Exponential Smoothing
    - Holt's Linear Trend (Additive)
    - Holt's Linear Trend (Multiplicative)
    """)