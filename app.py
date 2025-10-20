import streamlit as st
import joblib
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.express as px

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Dashboard de Gestão Condominial",
    page_icon="🏢",
    layout="wide"
)

# --- FUNÇÕES DE CARREGAMENTO (COM CACHE) ---

@st.cache_resource
def load_model():
    """Carrega o modelo e o vetorizador salvos."""
    try:
        model = joblib.load('modelo_classificador.joblib')
        vectorizer = joblib.load('vetorizador_tfidf.joblib')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Ficheiros do modelo de Machine Learning (.joblib) não encontrados. Verifique o upload no GitHub.")
        return None, None

@st.cache_data(ttl=600)
def load_data():
    """Carrega os dados da planilha do Google Sheets usando st.secrets."""
    try:
        # Lê as credenciais do gerenciador de segredos do Streamlit
        scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)

        # Abrir a planilha e carregar os dados
        sheet = client.open('DADOS DO PI').worksheet('LISTA ATUAL')
        data = sheet.get_all_records()
        df = pd.DataFrame(data)

        # O resto do processamento continua o mesmo
        if 'EXPEDIDA' in df.columns:
            df['DATA'] = pd.to_datetime(df['EXPEDIDA'], format='%d/%m/%Y', errors='coerce')
            df.dropna(subset=['DATA'], inplace=True)
            df['ANO'] = df['DATA'].dt.year
        else:
            st.error("Coluna de data 'EXPEDIDA' não foi encontrada na planilha.")
            return pd.DataFrame()

        df['MOTIVO'] = df['MOTIVO'].str.strip()
        mapeamento = {
            'DEJETOS PET': 'PET', 'PET DEJETO': 'PET', 'PET EM LOCAL INDEVIDO': 'PET', 'PET SEM GUIA': 'PET',
            'USO DA VAGA': 'VAGAS E ESTACIONAMENTO', 'USO DE VAGA': 'VAGAS E ESTACIONAMENTO', 'USO DO BOLSÃO': 'VAGAS E ESTACIONAMENTO',
            'PERNOITE VEICULO': 'VAGAS E ESTACIONAMENTO', 'ESTACIONAR EM LOCAL INDEVIDO': 'VAGAS E ESTACIONAMENTO', 'ABANDONO DE VEÍCULO': 'VAGAS E ESTACIONAMENTO',
            'ÓLEO NA VAGA': 'VAGAS E ESTACIONAMENTO', 'PERNOITE BOLSÃO': 'VAGAS E ESTACIONAMENTO', 'BIKE NA VAGA': 'VAGAS E ESTACIONAMENTO',
            'OBSTRUÇÃO CARGA E DESCARGA': 'OBSTRUCAO', 'OBSTRUÇÃO FAIXA PEDESTRES': 'OBSTRUCAO', 'OBSTRUÇÃO VAGA DEFIS': 'OBSTRUCAO',
            'OBSTRUÇÃO SAÍDA DE EMERGÊNCIA': 'OBSTRUCAO', 'OBSTRUÇÃO CARGA E DESCARGA/DEFIS/FAIXA': 'OBSTRUCAO',
            'OBSTRUÇÃO DE VAGA DEFIS/CARGA DESCARGA': 'OBSTRUCAO', 'OBSTRUÇÃO VAGA DEFIS/CARGA E DESCARGA': 'OBSTRUCAO', 'OBSTRUÇÃO DA FAIXA DE PEDESTRES': 'OBSTRUCAO',
            'ALTA VELOCIDADE/COLISÃO/DIREÇÃO PERIGOSA': 'DIRECAO PERIGOSA', 'DIREÇÃO PERIGOSA/COLISÃO': 'DIRECAO PERIGOSA', 'VEÍCULO NA CONTRAMÃO': 'DIRECAO PERIGOSA', 'COLISÃO': 'DIRECAO PERIGOSA',
            'DESCARTE DE LIXO': 'DESCARTE DE LIXO', 'LIXO - USO DO HALL': 'DESCARTE DE LIXO', 'DESCARTE INDEVIDO DE LIXO': 'DESCARTE DE LIXO',
            'OBJETOS NA VAGA': 'OBJETOS EM AREA COMUM', 'OBJETO NA VAGA': 'OBJETOS EM AREA COMUM', 'OBJETO EM ÁREA COMUM': 'OBJETOS EM AREA COMUM',
            'SOLICITAÇÃO DE MANUTENÇÃO': 'MANUTENCAO', 'SOLICITAÇÃO DE MANUTENÇÃO INTERNA': 'MANUTENCAO', 'SOLICITAÇÃO DE REPARO': 'MANUTENCAO'
        }
        df['MOTIVO_AGRUPADO'] = df['MOTIVO'].replace(mapeamento)
        return df

    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar os dados: {e}")
        st.info("Verifique se as credenciais 'gcp_service_account' foram configuradas corretamente nos Segredos (Secrets) do Streamlit.")
        return None

# --- O RESTO DO APP CONTINUA IGUAL ---
model, vectorizer = load_model()
df_original = load_data()

st.title("Dashboard de Gestão Condominial 🏢")
st.write("Análise de dados históricos e classificação de novas notificações.")
st.divider()

if df_original is not None and not df_original.empty:
    st.sidebar.header("Filtros do Dashboard")
    anos_disponiveis = sorted(df_original['ANO'].unique(), reverse=True)
    anos_selecionados = st.sidebar.multiselect("Selecione o(s) Ano(s):", anos_disponiveis, default=anos_disponiveis)
    motivos_disponiveis = sorted(df_original['MOTIVO_AGRUPADO'].unique())
    motivos_selecionados = st.sidebar.multiselect("Selecione o(s) Motivo(s):", motivos_disponiveis, default=motivos_disponiveis)
    df_filtrado = df_original[
        (df_original['ANO'].isin(anos_selecionados)) &
        (df_original['MOTIVO_AGRUPADO'].isin(motivos_selecionados))
    ]
    st.subheader("Visão Geral do Período Selecionado")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Notificações", f"{df_filtrado.shape[0]:,}".replace(",", "."))
    col2.metric("Categorias Únicas", f"{df_filtrado['MOTIVO_AGRUPADO'].nunique():,}".replace(",", "."))
    if not df_filtrado.empty:
        col3.metric("Período de", f"{df_filtrado['DATA'].min().strftime('%d/%m/%Y')} a {df_filtrado['DATA'].max().strftime('%d/%m/%Y')}")
    else:
        col3.metric("Período de", "N/A")
    st.divider()
    st.subheader("Análises Visuais")
    col_graf1, col_graf2 = st.columns(2)
    with col_graf1:
        top_5_motivos = df_filtrado['MOTIVO_AGRUPADO'].value_counts().nlargest(5)
        fig_bar = px.bar(top_5_motivos, x=top_5_motivos.index, y=top_5_motivos.values, title="Top 5 Motivos de Notificação", labels={'x': 'Motivo', 'y': 'Quantidade'}, text_auto=True)
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_graf2:
        df_filtrado['MES_ANO'] = df_filtrado['DATA'].dt.to_period('M').astype(str)
        notificacoes_por_mes = df_filtrado.groupby('MES_ANO').size().reset_index(name='Contagem')
        fig_line = px.line(notificacoes_por_mes, x='MES_ANO', y='Contagem', title="Tendência de Notificações por Mês", labels={'MES_ANO': 'Mês/Ano', 'Contagem': 'Quantidade de Notificações'}, markers=True)
        st.plotly_chart(fig_line, use_container_width=True)
    st.divider()
    st.subheader("🤖 Ferramenta de Classificação de Novas Notificações")
    user_input = st.text_area("Digite o texto da notificação para classificar:", height=150, placeholder="Ex: Morador reclama de vazamento no teto do banheiro da unidade...")
    if st.button("Classificar Notificação", type="primary"):
        if model is not None and vectorizer is not None and user_input.strip():
            input_vectorized = vectorizer.transform([user_input])
            prediction = model.predict(input_vectorized)[0]
            st.success(f"**Categoria Prevista:** {prediction}")
        elif not user_input.strip():
            st.warning("Por favor, insira um texto para classificar.")
else:
    st.warning("Não foi possível carregar os dados. Verifique as configurações e a conexão com a planilha.")
