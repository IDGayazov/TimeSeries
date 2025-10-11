import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import io
import base64

# Статистические библиотеки
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import scipy.stats as stats

# Настройки страницы
st.set_page_config(
    page_title="Анализатор временных рядов",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS для улучшения внешнего вида
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class TimeSeriesAnalyzer:
    def __init__(self):
        self.data = None
        self.target_column = None
        self.date_column = None
        
    def load_data(self, file):
        """Загрузка данных из файла"""
        try:
            if file.name.endswith('.csv'):
                self.data = pd.read_csv(file)
            elif file.name.endswith('.parquet'):
                self.data = pd.read_parquet(file)
            else:
                st.error("Поддерживаются только CSV и Parquet файлы")
                return False
            return True
        except Exception as e:
            st.error(f"Ошибка загрузки файла: {e}")
            return False
    
    def load_example_data(self):
        """Загрузка примеров данных"""
        example_option = st.selectbox(
            "Выберите пример датасета:",
            ["Air Passengers", "Daily Temperature", "Sales Data", "Stock Prices"]
        )
        
        if example_option == "Air Passengers":
            # Данные пассажиров авиалиний
            dates = pd.date_range('1949-01', '1960-12', freq='M')
            trend = np.linspace(100, 500, len(dates))
            seasonal = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
            noise = np.random.normal(0, 20, len(dates))
            values = trend + seasonal + noise
            self.data = pd.DataFrame({'date': dates, 'passengers': values})
            
        elif example_option == "Daily Temperature":
            # Ежедневные температурные данные
            dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
            trend = 15 + 0.01 * np.arange(len(dates))
            seasonal = 10 * np.sin(2 * np.pi * (dates.dayofyear / 365.25))
            noise = np.random.normal(0, 3, len(dates))
            values = trend + seasonal + noise
            self.data = pd.DataFrame({'date': dates, 'temperature': values})
            
        elif example_option == "Sales Data":
            # Данные продаж с трендом и сезонностью
            dates = pd.date_range('2018-01-01', '2022-12-31', freq='D')
            trend = 1000 + 2 * np.arange(len(dates))
            weekly_seasonal = 200 * np.sin(2 * np.pi * dates.dayofweek / 7)
            yearly_seasonal = 500 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
            noise = np.random.normal(0, 100, len(dates))
            values = trend + weekly_seasonal + yearly_seasonal + noise
            self.data = pd.DataFrame({'date': dates, 'sales': values})
            
        elif example_option == "Stock Prices":
            # Имитация цен акций (геометрическое броуновское движение)
            dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            self.data = pd.DataFrame({'date': dates, 'price': prices})
        
        return True

    def prepare_data(self):
        """Подготовка данных для анализа"""
        if self.data is None:
            return False
            
        # Выбор столбца с датой
        date_columns = self.data.select_dtypes(include=['datetime64', 'object']).columns
        if len(date_columns) == 0:
            st.error("Не найден столбец с датами")
            return False
            
        self.date_column = st.selectbox("Выберите столбец с датой:", date_columns)
        
        # Преобразование в datetime
        try:
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
            self.data = self.data.sort_values(self.date_column).reset_index(drop=True)
        except Exception as e:
            st.error(f"Ошибка преобразования даты: {e}")
            return False
        
        # Выбор целевой переменной
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            st.error("Не найдены числовые столбцы для анализа")
            return False
            
        self.target_column = st.selectbox("Выберите целевую переменную:", numeric_columns)
        
        return True

    def plot_time_series(self, window=30):
        """Построение графика временного ряда с трендом и скользящим средним"""
        fig = make_subplots(rows=1, cols=1)
        
        # Основной ряд
        fig.add_trace(
            go.Scatter(
                x=self.data[self.date_column],
                y=self.data[self.target_column],
                name='Исходный ряд',
                line=dict(color='blue', width=1)
            )
        )
        
        # Скользящее среднее
        if window > 1:
            rolling_mean = self.data[self.target_column].rolling(window=window).mean()
            fig.add_trace(
                go.Scatter(
                    x=self.data[self.date_column],
                    y=rolling_mean,
                    name=f'Скользящее среднее ({window})',
                    line=dict(color='red', width=2)
                )
            )
        
        fig.update_layout(
            title=f'Временной ряд: {self.target_column}',
            xaxis_title='Дата',
            yaxis_title=self.target_column,
            height=400
        )
        
        return fig

    def plot_correlation_heatmap(self):
        """Построение heatmap корреляций"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            st.warning("Недостаточно числовых столбцов для корреляционного анализа")
            return None
            
        corr_matrix = numeric_data.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title='Матрица корреляций'
        )
        
        return fig

    def plot_acf_pacf(self, max_lag=40):
        """Построение ACF и PACF графиков"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Автокорреляционная функция (ACF)', 
                          'Частная автокорреляционной функции (PACF)')
        )
        
        # ACF
        acf_values = acf(self.data[self.target_column].dropna(), nlags=max_lag)
        lags = list(range(len(acf_values)))
        
        fig.add_trace(
            go.Bar(x=lags, y=acf_values, name='ACF'),
            row=1, col=1
        )
        
        # PACF
        pacf_values = pacf(self.data[self.target_column].dropna(), nlags=max_lag)
        
        fig.add_trace(
            go.Bar(x=lags, y=pacf_values, name='PACF'),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=False)
        fig.update_xaxes(title_text="Лаг", row=2, col=1)
        fig.update_yaxes(title_text="ACF", row=1, col=1)
        fig.update_yaxes(title_text="PACF", row=2, col=1)
        
        return fig

    def decompose_series(self, period=30, model='additive'):
        """Декомпозиция временного ряда"""
        try:
            # Убедимся, что данные отсортированы и без пропусков
            series_data = self.data[self.target_column].dropna()
            if len(series_data) < period * 2:
                st.error(f"Для декомпозиции с периодом {period} нужно как минимум {period * 2} наблюдений")
                return None, None
                
            decomposition = seasonal_decompose(
                series_data,
                period=period,
                model=model
            )
            
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Исходный ряд', 'Тренд', 'Сезонность', 'Остатки')
            )
            
            dates = self.data[self.date_column].iloc[:len(series_data)]
            
            # Исходный ряд
            fig.add_trace(
                go.Scatter(x=dates, y=decomposition.observed, name='Исходный'),
                row=1, col=1
            )
            
            # Тренд
            fig.add_trace(
                go.Scatter(x=dates, y=decomposition.trend, name='Тренд'),
                row=2, col=1
            )
            
            # Сезонность
            fig.add_trace(
                go.Scatter(x=dates, y=decomposition.seasonal, name='Сезонность'),
                row=3, col=1
            )
            
            # Остатки
            fig.add_trace(
                go.Scatter(x=dates, y=decomposition.resid, name='Остатки'),
                row=4, col=1
            )
            
            fig.update_layout(height=800, showlegend=False)
            
            return fig, decomposition
            
        except Exception as e:
            st.error(f"Ошибка декомпозиции: {e}")
            return None, None

    def run_stationarity_tests(self):
        """Тесты на стационарность"""
        series = self.data[self.target_column].dropna()
        
        # ADF тест
        adf_result = adfuller(series)
        adf_statistic, adf_pvalue = adf_result[0], adf_result[1]
        
        # KPSS тест
        try:
            kpss_result = kpss(series, regression='c')
            kpss_statistic, kpss_pvalue = kpss_result[0], kpss_result[1]
        except:
            kpss_statistic, kpss_pvalue = np.nan, np.nan
        
        return {
            'ADF': {'statistic': adf_statistic, 'pvalue': adf_pvalue},
            'KPSS': {'statistic': kpss_statistic, 'pvalue': kpss_pvalue}
        }

def create_download_link(content, filename, file_type='html'):
    """Создание ссылки для скачивания - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
    if file_type == 'csv':
        data = content
        mime_type = 'text/csv'
    elif file_type == 'html':
        data = content
        mime_type = 'text/html'
    
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">📥 Скачать {filename}</a>'
    return href

def generate_html_report(analyzer, test_results, seasonality_period, decomposition_model, max_lag, rolling_window):
    """Генерация HTML отчета"""
    html_report = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Анализ временного ряда - {analyzer.target_column}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                color: #1f77b4;
                border-bottom: 3px solid #1f77b4;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .section {{
                margin: 30px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #1f77b4;
            }}
            .test-result {{
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                border: 1px solid #ddd;
            }}
            .stationary {{
                background-color: #d4edda;
                border-color: #c3e6cb;
            }}
            .non-stationary {{
                background-color: #f8d7da;
                border-color: #f5c6cb;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .metric-value {{
                font-size: 1.5em;
                font-weight: bold;
                color: #1f77b4;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #1f77b4;
                color: white;
            }}
            .timestamp {{
                text-align: right;
                color: #666;
                font-style: italic;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Анализ временного ряда</h1>
                <h2>Переменная: {analyzer.target_column}</h2>
            </div>
            
            <div class="section">
                <h3>📋 Основная информация</h3>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Целевая переменная</div>
                        <div class="metric-value">{analyzer.target_column}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Количество наблюдений</div>
                        <div class="metric-value">{len(analyzer.data):,}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Период данных</div>
                        <div class="metric-value">
                            {analyzer.data[analyzer.date_column].min().strftime('%Y-%m-%d')} 
                            до {analyzer.data[analyzer.date_column].max().strftime('%Y-%m-%d')}
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h3>🎛️ Параметры анализа</h3>
                <table>
                    <tr>
                        <th>Параметр</th>
                        <th>Значение</th>
                    </tr>
                    <tr>
                        <td>Период сезонности</td>
                        <td>{seasonality_period}</td>
                    </tr>
                    <tr>
                        <td>Модель декомпозиции</td>
                        <td>{decomposition_model}</td>
                    </tr>
                    <tr>
                        <td>Максимальный лаг</td>
                        <td>{max_lag}</td>
                    </tr>
                    <tr>
                        <td>Окно скользящего среднего</td>
                        <td>{rolling_window}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h3>📈 Результаты тестов на стационарность</h3>
                
                <div class="test-result {'stationary' if test_results['ADF']['pvalue'] < 0.05 else 'non-stationary'}">
                    <h4>ADF тест (Augmented Dickey-Fuller)</h4>
                    <p><strong>Статистика:</strong> {test_results['ADF']['statistic']:.4f}</p>
                    <p><strong>P-value:</strong> {test_results['ADF']['pvalue']:.4f}</p>
                    <p><strong>Вывод:</strong> Ряд {'<span style="color: green;">СТАЦИОНАРЕН</span>' if test_results['ADF']['pvalue'] < 0.05 else '<span style="color: red;">НЕСТАЦИОНАРЕН</span>'}</p>
                </div>
                
                <div class="test-result {'stationary' if test_results['KPSS']['pvalue'] > 0.05 else 'non-stationary'}">
                    <h4>KPSS тест (Kwiatkowski-Phillips-Schmidt-Shin)</h4>
                    <p><strong>Статистика:</strong> {test_results['KPSS']['statistic']:.4f}</p>
                    <p><strong>P-value:</strong> {test_results['KPSS']['pvalue']:.4f}</p>
                    <p><strong>Вывод:</strong> Ряд {'<span style="color: green;">СТАЦИОНАРЕН</span>' if test_results['KPSS']['pvalue'] > 0.05 else '<span style="color: red;">НЕСТАЦИОНАРЕН</span>'}</p>
                </div>
            </div>
            
            <div class="section">
                <h3>📊 Статистика ряда</h3>
                <table>
                    <tr>
                        <th>Метрика</th>
                        <th>Значение</th>
                    </tr>
                    <tr>
                        <td>Среднее значение</td>
                        <td>{analyzer.data[analyzer.target_column].mean():.2f}</td>
                    </tr>
                    <tr>
                        <td>Стандартное отклонение</td>
                        <td>{analyzer.data[analyzer.target_column].std():.2f}</td>
                    </tr>
                    <tr>
                        <td>Минимальное значение</td>
                        <td>{analyzer.data[analyzer.target_column].min():.2f}</td>
                    </tr>
                    <tr>
                        <td>Максимальное значение</td>
                        <td>{analyzer.data[analyzer.target_column].max():.2f}</td>
                    </tr>
                    <tr>
                        <td>Медиана</td>
                        <td>{analyzer.data[analyzer.target_column].median():.2f}</td>
                    </tr>
                </table>
            </div>
            
            <div class="timestamp">
                Отчет сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
    </body>
    </html>
    """
    return html_report

def main():
    st.markdown('<h1 class="main-header">📈 Анализатор временных рядов</h1>', unsafe_allow_html=True)
    
    # Инициализация анализатора
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = TimeSeriesAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Боковая панель для загрузки данных
    with st.sidebar:
        st.header("📁 Загрузка данных")
        
        data_source = st.radio(
            "Источник данных:",
            ["Загрузить файл", "Примеры данных"]
        )
        
        if data_source == "Загрузить файл":
            uploaded_file = st.file_uploader(
                "Выберите файл (CSV/Parquet)",
                type=['csv', 'parquet']
            )
            if uploaded_file is not None:
                if analyzer.load_data(uploaded_file):
                    st.success("Данные успешно загружены!")
        else:
            if analyzer.load_example_data():
                st.success("Пример данных загружен!")
    
    # Основная область приложения
    if analyzer.data is not None:
        # Подготовка данных
        if not analyzer.prepare_data():
            st.stop()
        
        # Панель управления
        st.markdown('<h2 class="section-header">🎛️ Панель управления</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            seasonality_period = st.number_input(
                "Период сезонности:",
                min_value=2,
                max_value=365,
                value=30,
                help="Период для декомпозиции (7-дни, 30-месяц, 365-год)"
            )
            
            decomposition_model = st.selectbox(
                "Модель декомпозиции:",
                ["additive", "multiplicative"]
            )
        
        with col2:
            max_lag = st.number_input(
                "Максимальный лаг для ACF/PACF:",
                min_value=5,
                max_value=100,
                value=40
            )
            
            rolling_window = st.number_input(
                "Окно скользящего среднего:",
                min_value=1,
                max_value=100,
                value=30
            )
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Количество наблюдений", len(analyzer.data))
            st.metric("Период данных", 
                     f"{analyzer.data[analyzer.date_column].min().date()} - {analyzer.data[analyzer.date_column].max().date()}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Визуализации
        st.markdown('<h2 class="section-header">📊 Визуализация и анализ</h2>', unsafe_allow_html=True)
        
        # Временной ряд
        st.subheader("Временной ряд")
        ts_fig = analyzer.plot_time_series(window=rolling_window)
        st.plotly_chart(ts_fig, use_container_width=True)
        
        # Корреляционная матрица
        st.subheader("Корреляционная матрица")
        corr_fig = analyzer.plot_correlation_heatmap()
        if corr_fig:
            st.plotly_chart(corr_fig, use_container_width=True)
        
        # ACF и PACF
        st.subheader("Автокорреляционные функции")
        acf_fig = analyzer.plot_acf_pacf(max_lag=max_lag)
        st.plotly_chart(acf_fig, use_container_width=True)
        
        # Декомпозиция
        st.subheader("Декомпозиция временного ряда")
        decomp_fig, decomposition = analyzer.decompose_series(
            period=seasonality_period,
            model=decomposition_model
        )
        if decomp_fig:
            st.plotly_chart(decomp_fig, use_container_width=True)
        
        # Тесты на стационарность
        st.subheader("Тесты на стационарность")
        test_results = analyzer.run_stationarity_tests()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### ADF тест (Augmented Dickey-Fuller)")
            st.write(f"Статистика: {test_results['ADF']['statistic']:.4f}")
            st.write(f"P-value: {test_results['ADF']['pvalue']:.4f}")
            if test_results['ADF']['pvalue'] < 0.05:
                st.success("Ряд стационарен (отвергаем H0)")
            else:
                st.error("Ряд нестационарен (не отвергаем H0)")
        
        with col2:
            st.markdown("##### KPSS тест (Kwiatkowski-Phillips-Schmidt-Shin)")
            if not np.isnan(test_results['KPSS']['statistic']):
                st.write(f"Статистика: {test_results['KPSS']['statistic']:.4f}")
                st.write(f"P-value: {test_results['KPSS']['pvalue']:.4f}")
                if test_results['KPSS']['pvalue'] > 0.05:
                    st.success("Ряд стационарен (не отвергаем H0)")
                else:
                    st.error("Ряд нестационарен (отвергаем H0)")
            else:
                st.warning("KPSS тест не может быть выполнен для этих данных")
        
        # Экспорт отчета
        st.markdown('<h2 class="section-header">📤 Экспорт отчета</h2>', unsafe_allow_html=True)
        
        if st.button("📊 Сгенерировать HTML отчет"):
            html_report = generate_html_report(
                analyzer, test_results, seasonality_period, 
                decomposition_model, max_lag, rolling_window
            )
            
            # Исправленный вызов функции
            download_link = create_download_link(
                html_report,  # передаем строку, а не DataFrame
                f"ts_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html", 
                'html'
            )
            st.markdown(download_link, unsafe_allow_html=True)
            
            # Также покажем предпросмотр отчета
            with st.expander("Предпросмотр отчета"):
                st.components.v1.html(html_report, height=800, scrolling=True)
    
    else:
        # Стартовая страница
        st.info("👈 Пожалуйста, загрузите данные или выберите пример из боковой панели чтобы начать анализ.")
        
        # Описание возможностей
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📁 Загрузка данных")
            st.write("""
            - Поддержка CSV и Parquet форматов
            - Автоматическое определение типов данных
            - Валидация временных рядов
            """)
        
        with col2:
            st.markdown("### 📊 Анализ")
            st.write("""
            - Визуализация временных рядов
            - Декомпозиция на компоненты
            - Автокорреляционный анализ
            - Тесты на стационарность
            """)
        
        with col3:
            st.markdown("### 📈 Визуализация")
            st.write("""
            - Интерактивные графики Plotly
            - Heatmap корреляций
            - ACF/PACF диаграммы
            - Декомпозиция ряда
            """)

if __name__ == "__main__":
    main()