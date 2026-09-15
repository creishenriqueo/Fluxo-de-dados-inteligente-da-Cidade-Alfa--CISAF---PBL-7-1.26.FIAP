# =====================================================================
# FASE 5 - DASHBOARD INTERATIVO CISAF (STREAMLIT)
# =====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
from scipy import stats # type: ignore
import warnings

# Ocultar avisos e configurar tema dos gráficos
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------
# 1. Configuração da Página do Streamlit
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="CISAF - Analytics", 
    page_icon="🛡️", 
    layout="wide"
)

st.title("🛡️ CISAF - Dashboard de Inteligência e Segurança")
st.markdown("Painel interativo para análise estatística e investigação de dados de segurança pública.")
st.markdown("---")

# ---------------------------------------------------------------------
# 2. Carregamento e Preparação dos Dados (Com Cache)
# ---------------------------------------------------------------------
# O decorador @st.cache_data evita que os dados sejam recarregados toda vez que interagimos com o dashboard
@st.cache_data
def load_and_prepare_data():
    # ATENÇÃO: Ajuste o caminho abaixo se necessário para apontar para o seu arquivo
    caminho_arquivo = "/Users/macintosh/Desktop/CISAF_SegurancaPublica_ENTREGA_PBL_FASE_5/ext_sbdg_9m26.xlsx"
    
    try:
        df = pd.read_excel(caminho_arquivo, sheet_name="Base_SBDG")
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {caminho_arquivo}")
        return pd.DataFrame() # Retorna dataframe vazio em caso de erro

    # --- 1º DESAFIO: PREPARAÇÃO E QUALIDADE DE DADOS ---
    # Tratamento de Nulos
    colunas_texto = ['loc_reg', 'tp_sex'] 
    for col in colunas_texto:
        if col in df.columns:
            df[col] = df[col].fillna('Não Informado')

    # Removendo registros nulos da coluna de idade
    if 'tp_ida' in df.columns:
        df.dropna(subset=['tp_ida'], inplace=True)
        # Limpeza de dados inválidos (idades irreais)
        df = df[(df['tp_ida'] >= 0) & (df['tp_ida'] <= 120)]

    # Tratamento de Duplicatas
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)

    # Limpeza de dados inválidos (quantidades negativas)
    colunas_qtd = ['qtd_obj', 'qtd_inf', 'qtd_test']
    for col in colunas_qtd:
        if col in df.columns:
            df = df[df[col] >= 0]

    return df

# Executando a função de carregamento
df = load_and_prepare_data()

# ---------------------------------------------------------------------
# 3. Construção do Dashboard
# ---------------------------------------------------------------------
if not df.empty:
    
    # --- BARRA LATERAL (SIDEBAR) PARA FILTROS ---
    st.sidebar.header("Filtros Interativos")
    st.sidebar.write("Explore as manchas criminais e perfil das vítimas.")
    
    # Criando um filtro baseado na coluna de Região ('loc_reg')
    if 'loc_reg' in df.columns:
        regioes = ["Todas"] + list(df['loc_reg'].dropna().unique())
        regiao_selecionada = st.sidebar.selectbox("Selecione a Região:", regioes)
        
        # Aplicando o filtro ao dataframe
        if regiao_selecionada != "Todas":
            df_filtrado = df[df['loc_reg'] == regiao_selecionada]
        else:
            df_filtrado = df
    else:
        df_filtrado = df

    # --- SEPARANDO O CONTEÚDO EM ABAS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Visão Geral e Qualidade", 
        "📈 Estatística Descritiva", 
        "🔥 Correlação", 
        "🧪 Teste de Hipóteses"
    ])
    
    # ABA 1: Visão Geral e Amostragem
    with tab1:
        st.subheader("Base de Dados Refinada")
        st.write(f"**Total de Registros (após limpeza e filtros):** {df_filtrado.shape[0]}")
        st.dataframe(df_filtrado.head(10)) # Exibe as primeiras linhas interativas
        
        st.subheader("Amostragem Aleatória Simples (10%)")
        amostra_simples = df_filtrado.sample(frac=0.10, random_state=42)
        st.write(f"Tamanho da amostra extraída: **{amostra_simples.shape[0]} registros**")
        
    # ABA 2: Estatística Descritiva
    with tab2:
        st.subheader("Resumo Estatístico")
        colunas_numericas = ['tp_ida', 'qtd_obj', 'qtd_inf', 'qtd_test']
        cols_num_existentes = [c for c in colunas_numericas if c in df_filtrado.columns]
        
        if cols_num_existentes:
            st.write("**Variáveis Numéricas:**")
            st.dataframe(df_filtrado[cols_num_existentes].describe())
            
            st.write("**Assimetria (Skewness):**")
            st.dataframe(df_filtrado[cols_num_existentes].skew())
            
            # Dividindo a tela em duas colunas para as tabelas de frequência
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Frequência por Sexo da Vítima (%)**")
                if 'tp_sex' in df_filtrado.columns:
                    freq_sexo = df_filtrado['tp_sex'].value_counts(normalize=True) * 100
                    st.dataframe(freq_sexo)
            with col2:
                st.write("**Frequência por Região (Absoluto)**")
                if 'loc_reg' in df_filtrado.columns:
                    freq_reg = df_filtrado['loc_reg'].value_counts()
                    st.dataframe(freq_reg)
        else:
            st.warning("Colunas numéricas necessárias não encontradas.")

    # ABA 3: Correlação
    with tab3:
        st.subheader("Matriz de Correlação de Pearson")
        st.write("Identifica se o aumento de uma variável numérica impacta em outra.")
        
        if cols_num_existentes:
            matriz_correlacao = df_filtrado[cols_num_existentes].corr(method='pearson')
            
            # Desenhando o gráfico com Matplotlib/Seaborn e enviando para o Streamlit
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1, ax=ax)
            st.pyplot(fig) # Comando crucial para renderizar o gráfico no web app
        else:
            st.warning("Dados insuficientes para correlação.")

    # ABA 4: Testes de Hipóteses
    with tab4:
        st.subheader("Teste T de Student")
        st.markdown("""
        **Cenário de Investigação:**
        Existe diferença estatisticamente significativa na **IDADE MÉDIA** entre vítimas do sexo Masculino e Feminino?
        * **H0 (Nula):** Não há diferença na idade média.
        * **H1 (Alternativa):** Há diferença.
        """)
        
        if 'tp_sex' in df_filtrado.columns and 'tp_ida' in df_filtrado.columns:
            idade_homens = df_filtrado[df_filtrado['tp_sex'] == 'Masculino']['tp_ida'].dropna()
            idade_mulheres = df_filtrado[df_filtrado['tp_sex'] == 'Feminino']['tp_ida'].dropna()
            
            if not idade_homens.empty and not idade_mulheres.empty:
                t_stat, p_valor = stats.ttest_ind(idade_homens, idade_mulheres, equal_var=False)
                
                # Exibindo métricas com visual em destaque
                col1, col2, col3 = st.columns(3)
                col1.metric("Média de Idade (Masculino)", f"{idade_homens.mean():.2f}")
                col2.metric("Média de Idade (Feminino)", f"{idade_mulheres.mean():.2f}")
                col3.metric("Valor-p", f"{p_valor:.4f}")
                
                alpha = 0.05 
                st.markdown("---")
                if p_valor < alpha:
                    st.success("🎯 **Resultado: Rejeitamos a Hipótese Nula (H0).**\n\nExiste uma diferença estatisticamente significativa na idade das vítimas entre homens e mulheres na região selecionada.")
                else:
                    st.info("⚠️ **Resultado: Falhamos em rejeitar a Hipótese Nula (H0).**\n\nNão há evidências estatísticas suficientes para afirmar que a idade difere significativamente entre os sexos nesta amostra.")
            else:
                st.warning("Dados insuficientes para comparar homens e mulheres na região atual.")