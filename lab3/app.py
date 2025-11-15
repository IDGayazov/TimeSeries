import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from statsmodels.tsa.stattools import acf
import io

# =============================================================================
# МОДЕЛИ ПРОГНОЗИРОВАНИЯ
# =============================================================================

def naive_forecast(train_series, horizon):
    """Naive forecast - последнее значение"""
    return np.full(horizon, train_series[-1])

def snaive_forecast(train_series, horizon, seasonality=7):
    """Seasonal naive forecast - сезонное повторение"""
    if len(train_series) < seasonality:
        return naive_forecast(train_series, horizon)
    
    predictions = []
    for i in range(horizon):
        predictions.append(train_series[-(seasonality - (i % seasonality))])
    return np.array(predictions)

def exponential_smoothing_forecast(train_series, horizon, seasonal_periods=7):
    """Exponential smoothing"""
    try:
        if len(train_series) > 2 * seasonal_periods:
            model = ExponentialSmoothing(
                train_series, 
                seasonal_periods=seasonal_periods,
                trend='add', 
                seasonal='add'
            )
            fitted_model = model.fit()
            return fitted_model.forecast(horizon)
        else:
            model = ExponentialSmoothing(train_series, trend='add')
            fitted_model = model.fit()
            return fitted_model.forecast(horizon)
    except:
        return snaive_forecast(train_series, horizon)

def arima_forecast(train_series, horizon, order=(1,1,1)):
    """ARIMA модель"""
    try:
        model = ARIMA(train_series, order=order)
        fitted_model = model.fit()
        return fitted_model.forecast(horizon)
    except:
        return exponential_smoothing_forecast(train_series, horizon)

def linear_regression_forecast(train_series, horizon, window=10):
    """Линейная регрессия для временных рядов"""
    try:
        X, y = [], []
        for i in range(window, len(train_series)):
            X.append(train_series[i-window:i])
            y.append(train_series[i])
        
        model = LinearRegression()
        model.fit(X, y)
        
        predictions = []
        current_window = train_series[-window:].copy()
        
        for _ in range(horizon):
            pred = model.predict([current_window])[0]
            predictions.append(pred)
            current_window = np.append(current_window[1:], pred)
        
        return np.array(predictions)
    except:
        return naive_forecast(train_series, horizon)

def random_forest_forecast(train_series, horizon, window=10):
    """Случайный лес для временных рядов"""
    try:
        X, y = [], []
        for i in range(window, len(train_series)):
            X.append(train_series[i-window:i])
            y.append(train_series[i])
        
        model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X, y)
        
        predictions = []
        current_window = train_series[-window:].copy()
        
        for _ in range(horizon):
            pred = model.predict([current_window])[0]
            predictions.append(pred)
            current_window = np.append(current_window[1:], pred)
        
        return np.array(predictions)
    except:
        return linear_regression_forecast(train_series, horizon)

# =============================================================================
# УТИЛИТЫ И МЕТРИКИ
# =============================================================================

def calculate_metrics(y_true, y_pred):
    """Вычисление метрик качества"""
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

def prepare_data(df, date_col, target_col, test_size=0.2):
    """Подготовка данных для обучения"""
    df_sorted = df.sort_values(date_col)
    series = df_sorted[target_col].values
    
    split_idx = int(len(series) * (1 - test_size))
    train_data = series[:split_idx]
    test_data = series[split_idx:]
    
    return train_data, test_data, series

def apply_boxcox(series, lambda_val=None):
    """Применение преобразования Бокса-Кокса"""
    if lambda_val is None:
        # Автоматический подбор lambda
        transformed, fitted_lambda = stats.boxcox(series + 1e-9)  # добавляем маленькое число для положительных значений
    else:
        # Ручное значение lambda
        if lambda_val == 0:
            transformed = np.log(series + 1e-9)
        else:
            transformed = (series ** lambda_val - 1) / lambda_val
        fitted_lambda = lambda_val
    
    return transformed, fitted_lambda

def inverse_boxcox(transformed, lambda_val):
    """Обратное преобразование Бокса-Кокса"""
    if lambda_val == 0:
        return np.exp(transformed)
    else:
        return (transformed * lambda_val + 1) ** (1 / lambda_val)

# =============================================================================
# ОСНОВНОЕ ПРИЛОЖЕНИЕ STREAMLIT
# =============================================================================

# Конфигурация страницы
st.set_page_config(
    page_title="Time Series Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок
st.title("📈 Прогнозирование временных рядов")
st.markdown("Загрузите данные, выберите модель и получите прогноз")

# Боковая панель
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Загрузка файла
    uploaded_file = st.file_uploader(
        "Загрузите файл данных", 
        type=['csv', 'parquet'],
        help="Поддерживаются CSV и Parquet файлы"
    )
    
    # Настройки данных
    if uploaded_file is not None:
        st.subheader("Настройки данных")
        
        # Чтение файла
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_parquet(uploaded_file)
        
        # Выбор колонок
        date_col = st.selectbox("Выберите колонку с датой", df.columns)
        target_col = st.selectbox("Выберите целевую переменную", df.columns)
        
        # Преобразование даты
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        # Горизонт прогнозирования
        horizon = st.selectbox(
            "Горизонт прогнозирования",
            [1, 7, 30, 90],
            index=1,
            help="Количество шагов для прогноза"
        )
        
        # Разделение данных
        test_size = st.slider(
            "Доля тестовых данных (%)",
            min_value=10,
            max_value=50,
            value=20,
            help="Процент данных для тестирования"
        )

        # Настройки моделей
        st.subheader("Выбор моделей")
        
        models_to_use = {}
        
        # Бенчмарки
        st.markdown("**Бенчмарки:**")
        models_to_use['Naive'] = st.checkbox("Naive", value=True)
        models_to_use['Seasonal Naive'] = st.checkbox("Seasonal Naive", value=True)
        models_to_use['Exponential Smoothing'] = st.checkbox("Exponential Smoothing", value=True)
        
        # Продвинутые модели
        st.markdown("**Продвинутые модели:**")
        models_to_use['ARIMA'] = st.checkbox("ARIMA", value=True)
        models_to_use['Linear Regression'] = st.checkbox("Linear Regression", value=True)
        models_to_use['Random Forest'] = st.checkbox("Random Forest", value=True)
        
        # Настройки преобразований
        st.subheader("Преобразования")
        transformation = st.radio(
            "Преобразование Бокса-Кокса",
            ["Авто", "Ручной λ", "Нет"],
            help="Стабилизация дисперсии"
        )
        
        if transformation == "Ручной λ":
            lambda_val = st.slider(
                "λ значение",
                min_value=-2.0,
                max_value=2.0,
                value=0.5,
                step=0.1
            )
        else:
            lambda_val = None

        # Прогнозирование
        if st.button("🚀 Запустить прогнозирование", type="primary", use_container_width=True):
            with st.spinner("Выполняется прогнозирование..."):
                results = {}
                metrics = {}
                
                # Подготовка данных
                train_data, test_data, full_series = prepare_data(
                    df, date_col, target_col, test_size/100
                )
                
                # Применяем преобразование если нужно
                if transformation != "Нет":
                    train_transformed, fitted_lambda = apply_boxcox(train_data, lambda_val)
                else:
                    train_transformed = train_data
                    fitted_lambda = None
                
                # Прогноз для каждой выбранной модели
                for model_name, use_model in models_to_use.items():
                    if use_model:
                        try:
                            if model_name == 'Naive':
                                pred = naive_forecast(train_transformed, horizon)
                            elif model_name == 'Seasonal Naive':
                                pred = snaive_forecast(train_transformed, horizon, seasonality=7)
                            elif model_name == 'Exponential Smoothing':
                                pred = exponential_smoothing_forecast(train_transformed, horizon)
                            elif model_name == 'ARIMA':
                                pred = arima_forecast(train_transformed, horizon)
                            elif model_name == 'Linear Regression':
                                pred = linear_regression_forecast(train_transformed, horizon)
                            elif model_name == 'Random Forest':
                                pred = random_forest_forecast(train_transformed, horizon)
                            
                            # Обратное преобразование если нужно
                            if transformation != "Нет":
                                pred = inverse_boxcox(pred, fitted_lambda)
                            
                            results[model_name] = pred
                            metrics[model_name] = calculate_metrics(test_data[:horizon], pred)
                            
                        except Exception as e:
                            st.error(f"Ошибка в модели {model_name}: {str(e)}")
                
                # Сохраняем результаты в session state
                st.session_state.results = results
                st.session_state.metrics = metrics
                st.session_state.train_data = train_data
                st.session_state.test_data = test_data
                st.session_state.horizon = horizon
                st.session_state.df = df
                st.session_state.date_col = date_col
                st.session_state.target_col = target_col

# Отображение результатов
if hasattr(st.session_state, 'results') and st.session_state.results:
    results = st.session_state.results
    metrics = st.session_state.metrics
    train_data = st.session_state.train_data
    test_data = st.session_state.test_data
    horizon = st.session_state.horizon
    df = st.session_state.df
    date_col = st.session_state.date_col
    target_col = st.session_state.target_col
    
    # Вкладки
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Визуализация", 
        "📈 Метрики", 
        "🔍 Диагностика",
        "💾 Экспорт"
    ])
    
    with tab1:
        # Выбор модели для визуализации
        selected_model = st.selectbox(
            "Выберите модель для визуализации",
            list(results.keys())
        )
        
        # Создание графика
        fig = go.Figure()
        
        # Исторические данные
        fig.add_trace(go.Scatter(
            x=list(range(len(train_data))),
            y=train_data,
            name="Обучающие данные",
            line=dict(color='blue')
        ))
        
        # Тестовые данные
        fig.add_trace(go.Scatter(
            x=list(range(len(train_data), len(train_data) + len(test_data[:horizon]))),
            y=test_data[:horizon],
            name="Фактические значения",
            line=dict(color='green')
        ))
        
        # Прогноз
        fig.add_trace(go.Scatter(
            x=list(range(len(train_data), len(train_data) + horizon)),
            y=results[selected_model],
            name=f"Прогноз ({selected_model})",
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f"Прогноз модели {selected_model}",
            xaxis_title="Время",
            yaxis_title="Значение",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Таблица метрик
        metrics_df = pd.DataFrame(metrics).T
        metrics_df = metrics_df.round(4)
        
        # Ранжирование по RMSE
        metrics_df['Rank'] = metrics_df['RMSE'].rank()
        
        st.subheader("Сравнение метрик моделей")
        st.dataframe(metrics_df.style.highlight_min(axis=0, color='lightgreen'))
        
        # Визуализация сравнения метрик
        fig_metrics = go.Figure()
        
        for metric in ['RMSE', 'MAE', 'MAPE']:
            if metric in metrics_df.columns:
                fig_metrics.add_trace(go.Bar(
                    name=metric,
                    x=metrics_df.index,
                    y=metrics_df[metric],
                    text=metrics_df[metric].round(3)
                ))
        
        fig_metrics.update_layout(
            title="Сравнение метрик по моделям",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    with tab3:
        # Диагностика для выбранной модели
        diag_model = st.selectbox(
            "Выберите модель для диагностики",
            list(results.keys()),
            key="diag_select"
        )
        
        if diag_model in results:
            residuals = test_data[:horizon] - results[diag_model]
            
            # Графики диагностики
            fig_diag = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Остатки во времени', 
                    'Гистограмма остатков',
                    'Q-Q Plot', 
                    'ACF остатков'
                )
            )
            
            # Остатки во времени
            fig_diag.add_trace(
                go.Scatter(y=residuals, mode='lines', name='Остатки'),
                row=1, col=1
            )
            fig_diag.add_hline(y=0, line_dash="dash", row=1, col=1)
            
            # Гистограмма
            fig_diag.add_trace(
                go.Histogram(x=residuals, name='Распределение'),
                row=1, col=2
            )
            
            # Q-Q plot
            qq = stats.probplot(residuals, dist="norm")
            fig_diag.add_trace(
                go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Q-Q'),
                row=2, col=1
            )
            fig_diag.add_trace(
                go.Scatter(x=qq[0][0], y=qq[0][0]*qq[1][0] + qq[1][1], 
                          mode='lines', name='Теоретическая'),
                row=2, col=1
            )
            
            # ACF
            acf_vals = acf(residuals, nlags=min(10, len(residuals)-1))
            fig_diag.add_trace(
                go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name='ACF'),
                row=2, col=2
            )
            
            fig_diag.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig_diag, use_container_width=True)
    
    with tab4:
        st.subheader("Экспорт прогнозов")
        
        # Выбор моделей для экспорта
        models_to_export = st.multiselect(
            "Выберите модели для экспорта",
            list(results.keys()),
            default=list(results.keys())[:2]
        )
        
        if models_to_export:
            # Создаем DataFrame для экспорта
            export_data = {}
            
            # Добавляем даты если возможно
            last_date = df[date_col].iloc[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), 
                periods=horizon, 
                freq='D'
            )
            
            export_data['date'] = future_dates
            
            for model in models_to_export:
                export_data[model] = results[model]
            
            export_df = pd.DataFrame(export_data)
            
            # Предпросмотр
            st.write("Предпросмотр данных для экспорта:")
            st.dataframe(export_df.head(10))
            
            # Форматы экспорта
            export_format = st.radio("Формат экспорта", ['CSV', 'Excel'])
            
            # Кнопки скачивания
            col1, col2 = st.columns(2)
            
            with col1:
                if export_format == 'CSV':
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        "📥 Скачать CSV",
                        data=csv,
                        file_name=f"forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if export_format == 'Excel':
                    excel_buffer = io.BytesIO()
                    export_df.to_excel(excel_buffer, index=False)
                    st.download_button(
                        "📥 Скачать Excel",
                        data=excel_buffer.getvalue(),
                        file_name=f"forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.ms-excel"
                    )

else:
    # Стартовый экран
    st.markdown("""
    ## Инструкция по использованию:
    
    1. **Загрузите данные** - CSV или Parquet файл с временным рядом
    2. **Настройте параметры** - выберите целевую переменную и горизонт
    3. **Выберите модели** - отметьте модели для сравнения
    4. **Запустите прогнозирование** - нажмите кнопку в боковой панели
    
    ### Поддерживаемые модели:
    - 📊 **Бенчмарки**: Naive, Seasonal Naive, Exponential Smoothing
    - 🧠 **Продвинутые**: ARIMA, Linear Regression, Random Forest
    
    ### Возможности:
    - Визуализация прогнозов
    - Сравнение метрик качества
    - Диагностика остатков моделей
    - Экспорт результатов
    """)
