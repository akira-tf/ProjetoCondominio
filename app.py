import streamlit as st
import joblib
import pandas as pd

# --- CONFIGURAÇÃO DA PÁGINA ---
# Define o título da página, ícone e layout.
# A configuração da página deve ser o primeiro comando do Streamlit no script.
st.set_page_config(
    page_title="Análise de Notificações Condominiais",
    page_icon="🏢",
    layout="centered"
)

# --- FUNÇÃO DE CACHE PARA CARREGAR O MODELO ---
# Usamos @st.cache_resource para garantir que o modelo e o vetorizador
# sejam carregados apenas uma vez, otimizando a performance do app.
@st.cache_resource
def load_model():
    """Carrega o modelo e o vetorizador salvos."""
    try:
        model = joblib.load('modelo_classificador.joblib')
        vectorizer = joblib.load('vetorizador_tfidf.joblib')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Arquivos de modelo não encontrados. Certifique-se que 'modelo_classificador.joblib' e 'vetorizador_tfidf.joblib' estão na mesma pasta que o app.py.")
        return None, None

# Carrega os artefatos de machine learning
model, vectorizer = load_model()

# --- INTERFACE DO USUÁRIO (UI) ---

# Título principal do aplicativo
st.title("Sistema Inteligente de Gestão Condominial 🤖")

# Subtítulo com uma breve descrição
st.write(
    "Esta aplicação utiliza um modelo de Machine Learning para classificar "
    "automaticamente as notificações de condomínio, otimizando o tempo de resposta e a organização."
)

# Adiciona um separador visual
st.divider()

# Campo de texto para o usuário inserir a notificação
user_input = st.text_area(
    "Digite o texto da notificação para classificar:",
    height=150,
    placeholder="Ex: Morador reclama de vazamento no teto do banheiro da unidade..."
)

# Botão para iniciar a classificação
if st.button("Classificar Notificação", type="primary"):
    # Verifica se os modelos foram carregados e se o usuário inseriu algum texto
    if model is not None and vectorizer is not None and user_input.strip():
        # 1. Vetorizar o texto de entrada do usuário usando o vetorizador carregado
        input_vectorized = vectorizer.transform([user_input])

        # 2. Fazer a predição com o modelo
        prediction = model.predict(input_vectorized)[0]  # Pega o primeiro (e único) resultado
        prediction_proba = model.predict_proba(input_vectorized)

        # 3. Mostrar o resultado com destaque
        st.subheader("Resultado da Classificação:")
        st.success(f"**Categoria Prevista:** {prediction}")

        # 4. Opcional: Mostrar a confiança do modelo em cada categoria
        st.subheader("Confiança da Predição:")
        # Cria um DataFrame do pandas para visualizar melhor as probabilidades
        proba_df = pd.DataFrame(prediction_proba, columns=model.classes_).T
        proba_df.rename(columns={0: 'Probabilidade'}, inplace=True)
        # Mostra um gráfico de barras com as probabilidades
        st.bar_chart(proba_df)

    elif not user_input.strip():
        # Alerta caso o usuário clique no botão sem digitar nada
        st.warning("Por favor, insira um texto para classificar.")
