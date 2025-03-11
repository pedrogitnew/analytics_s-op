import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
import matplotlib.pyplot as plt
import base64
import plotly.express as px

# Função para gerar dados sintéticos
def generate_data(n=100):
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=n)
    novo_scale = 500 / 3  # σ = 166.67
    actual = np.random.normal(loc=500, scale=novo_scale, size=n).round().astype(int)
    forecast = actual * np.random.normal(loc=0.9, scale=0.2, size=n) + 100
    return pd.DataFrame({
        'Date': dates,
        'Actual': actual.astype(int),
        'Forecast': forecast.astype(int)
    })

# Funções de cálculo de métricas
def calculate_metrics(df):
    df['Error'] = df['Forecast'] - df['Actual']
    df['Absolute_Error'] = np.abs(df['Error'])
    df['Percentage_Error'] = (df['Absolute_Error'] / df['Actual']).replace(np.inf, 0)
    
    mape = df['Percentage_Error'].mean() * 100
    wmape = (df['Absolute_Error'].sum() / df['Actual'].sum()) * 100
    bias = df['Error'].mean()
    accuracy = 100 - mape
    
    return {
        'MAPE': mape,
        'WMAPE': wmape,
        'BIAS': bias,
        'Forecast_Accuracy': accuracy
    }

# Função para criar PDF
def create_pdf(metrics, plot_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Relatório de Previsão - S&OP Metrics", ln=1, align='C')
    
    pdf.cell(200, 10, txt="Métricas Calculadas:", ln=1)
    for key, value in metrics.items():
        pdf.cell(200, 10, txt=f"{key}: {value:.2f}%", ln=1)
    
    pdf.image(plot_path, x=10, y=100, w=180)
    
    return pdf.output(dest='S').encode('latin-1')

# Configuração da página
st.set_page_config(page_title="Analytics S&OP", layout="wide")

# Título principal
st.title("📊 Painel de Métricas S&OP")

# Criando abas
tab1, tab2 = st.tabs(["📚 Explicação das Métricas", "🔍 Análise dos Dados"])

# Aba 1 - Explicação das Métricas
with tab1:
    st.header("Explicação Detalhada das Métricas")
    
    with st.expander("MAPE - Erro Percentual Absoluto Médio"):
        st.markdown("""
        **Fórmula:**
        ```latex
        MAPE = (1/n) * Σ(|(Actual - Forecast)/Actual|) * 100%
        ```
        - Mede o erro percentual médio absoluto
        - Ideal para comparar desempenho entre diferentes produtos
        - Limitação: Pode distorcer resultados quando há valores reais muito pequenos
        """)
    
    with st.expander("WMAPE - Erro Percentual Absoluto Médio Ponderado"):
        st.markdown("""
        **Fórmula:**
        ```latex
        WMAPE = (Σ|Actual - Forecast| / ΣActual) * 100%
        ```
        - Versão ponderada do MAPE
        - Dá mais peso a erros em períodos de maior volume
        - Mais adequado para análise de portfólio com variação de demanda
        """)
    
    with st.expander("BIAS - Viés de Previsão"):
        st.markdown("""
        **Fórmula:**
        ```latex
        BIAS = (1/n) * Σ(Forecast - Actual)
        ```
        - Indica tendência de super/subestimação
        - Valor positivo = Previsões consistentemente altas em relação ao real
        - Valor negativo = Previsões consistentemente baixas em relação ao real
        """)
    
    with st.expander("Forecast Accuracy - Acurácia da Previsão"):
        st.markdown("""
        **Fórmula:**
        ```latex
        Forecast Accuracy = 100% - MAPE (Ou WMAPE)
        ```
        - Representa a porcentagem de acerto geral
        - Valores acima de 85% são considerados excelentes
        - Deve ser analisado em conjunto com outras métricas
        """)

# Aba 2 - Análise dos Dados
with tab2:
    st.header("Análise de Previsão")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")
        n_records = st.slider("Número de Registros", 100, 500, 100)
        base_actual = st.slider("Média da Demanda Real", 300, 700, 500)
        noise_level = st.slider("Nível de Ruído (%)", 0, 30, 10)
    
    # Geração de dados
    df = generate_data(n_records)
    metrics = calculate_metrics(df)
    
    # Layout de colunas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Visualização Temporal")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['Date'], df['Actual'], label='Actual')
        ax.plot(df['Date'], df['Forecast'], label='Forecast', alpha=0.7)
        ax.set_title("Comparação Actual vs Forecast")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        st.subheader("Dados Completos")
        st.dataframe(df, height=300)
    
    with col2:
        st.subheader("Métricas-Chave")
        st.metric("MAPE", f"{metrics['MAPE']:.2f}%", help="Erro percentual médio absoluto")
        st.metric("WMAPE", f"{metrics['WMAPE']:.2f}%", help="Erro percentual ponderado")
        st.metric("BIAS", f"{metrics['BIAS']:.2f}", help="Viés médio das previsões")
        st.metric("Acurácia", f"{metrics['Forecast_Accuracy']:.2f}%")
        
        st.subheader("Distribuição de Erros")
        # Convertendo os intervalos para strings e criando um DataFrame
        error_bins = pd.cut(df['Error'], bins=5).value_counts().reset_index()
        error_bins.columns = ['Intervalo', 'Contagem']
        error_bins['Intervalo'] = error_bins['Intervalo'].astype(str)

        # Usando matplotlib para maior controle
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(error_bins['Intervalo'], error_bins['Contagem'])
        ax.set_title("Distribuição de Erros")
        ax.set_ylabel("Frequência")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Relatório
        if st.button("📥 Gerar Relatório Completo"):
            plt.savefig('temp_plot.png')
            pdf_output = create_pdf(metrics, 'temp_plot.png')
            b64 = base64.b64encode(pdf_output).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="relatorio_SOP.pdf">Baixar Relatório</a>'
            st.markdown(href, unsafe_allow_html=True)

# Rodapé
st.markdown("---")
st.caption("Desenvolvido por Pedro Cavalcanti - Painel de Analytics para S&OP")
