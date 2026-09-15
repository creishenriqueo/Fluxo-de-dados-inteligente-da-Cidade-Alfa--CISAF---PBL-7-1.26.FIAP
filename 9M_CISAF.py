# =====================================================================
# FASE 5 - ANÁLISE ESTATÍSTICA E INVESTIGAÇÃO DE DADOS
# =====================================================================

# ---------------------------------------------------------------------
# 1. Importação das Bibliotecas Necessárias
# ---------------------------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
from scipy import stats # type: ignore

# Configuração visual para os gráficos
sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings('ignore') # Ocultar avisos para deixar o output mais limpo

# ---------------------------------------------------------------------
# 2. Carregamento dos Dados
# ---------------------------------------------------------------------
print("--- 2. Carregamento dos Dados ---")
# Carregando a aba 'Base_SBDG' do arquivo Excel
# Substitua o caminho abaixo pelo local real onde o arquivo está salvo
caminho_arquivo = "/Users/macintosh/Desktop/CISAF_SegurancaPublica_ENTREGA_PBL_FASE_5/ext_sbdg_9m26.xlsx"
df = pd.read_excel(caminho_arquivo, sheet_name="Base_SBDG")

print(f"Base carregada com sucesso! O dataset possui {df.shape[0]} linhas e {df.shape[1]} colunas.\n")
print(df.head(3))

# ---------------------------------------------------------------------
# 3. Estatística Descritiva
# ---------------------------------------------------------------------
print("\n--- 3. Estatística Descritiva ---")
# Selecionando algumas colunas numéricas de interesse (Idade, Qtd Objetos, Qtd Infratores, Qtd Testemunhas)
colunas_numericas = ['tp_ida', 'qtd_obj', 'qtd_inf', 'qtd_test']

# Resumo estatístico das variáveis numéricas
descritiva_num = df[colunas_numericas].describe()
print("Resumo Estatístico (Variáveis Numéricas):")
print(descritiva_num)

# Análise de distribuição (Assimetria/Skewness)
print("\nAssimetria (Skewness):")
print(df[colunas_numericas].skew())

# Resumo estatístico de variáveis categóricas (Frequência de Sexo e Tipo de Crime/Infração)
print("\nFrequência por Sexo da Vítima (tp_sex):")
print(df['tp_sex'].value_counts(normalize=True) * 100) # Em porcentagem

print("\nFrequência por Região (loc_reg):")
print(df['loc_reg'].value_counts())


# ---------------------------------------------------------------------
# 4. Amostragem
# ---------------------------------------------------------------------
print("\n--- 4. Amostragem ---")
# 4.1. Amostragem Aleatória Simples (10% da base)
amostra_simples = df.sample(frac=0.10, random_state=42)
print(f"Tamanho da amostra simples (10%): {amostra_simples.shape[0]} registros")

# 4.2. Amostragem Estratificada (Garantindo a proporção da variável Sexo - tp_sex)
# Coletaremos uma amostra de 20% mantendo a mesma proporção de homens e mulheres da base original
tamanho_frac = 0.20

# Usando o método .sample() nativo do groupby (Modo seguro que não apaga a coluna)
amostra_estratificada = df.groupby('tp_sex').sample(frac=tamanho_frac, random_state=42)

print("\nDistribuição na amostra estratificada (Qtd):")
print(amostra_estratificada['tp_sex'].value_counts())


# ---------------------------------------------------------------------
# 5. Análise de Correlação
# ---------------------------------------------------------------------
print("\n--- 5. Análise de Correlação ---")
# Calculando a matriz de correlação de Pearson
matriz_correlacao = df[colunas_numericas].corr(method='pearson')

print("Matriz de Correlação:")
print(matriz_correlacao)

# Plotando o Heatmap da Correlação
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1)
plt.title('Matriz de Correlação - Base SBDG')
plt.show()


# ---------------------------------------------------------------------
# 6. Testes de Hipóteses
# ---------------------------------------------------------------------
print("\n--- 6. Testes de Hipóteses ---")
# Cenário de Investigação: 
# Existe diferença estatisticamente significativa na IDADE MÉDIA ('tp_ida') 
# entre vítimas do sexo Masculino e Feminino?

# H0 (Hipótese Nula): Não há diferença na idade média entre os gêneros.
# H1 (Hipótese Alternativa): Há diferença na idade média.

# Separando os grupos
idade_homens = df[df['tp_sex'] == 'Masculino']['tp_ida'].dropna()
idade_mulheres = df[df['tp_sex'] == 'Feminino']['tp_ida'].dropna()

# Aplicando o Teste T de Student para amostras independentes
t_stat, p_valor = stats.ttest_ind(idade_homens, idade_mulheres, equal_var=False)

print(f"Idade média Masculino: {idade_homens.mean():.2f}")
print(f"Idade média Feminino: {idade_mulheres.mean():.2f}")
print(f"Estatística T: {t_stat:.4f}")
print(f"Valor-p: {p_valor:.4f}")

# Nível de significância (Alpha = 5%)
alpha = 0.05 

print("\n--- Conclusão do Teste ---")
if p_valor < alpha:
    print("Resultado: Rejeitamos a Hipótese Nula (H0).")
    print("Interpretação: Existe uma diferença estatisticamente significativa na idade das vítimas entre homens e mulheres.")
else:
    print("Resultado: Falhamos em rejeitar a Hipótese Nula (H0).")
    print("Interpretação: Não há evidências estatísticas suficientes para afirmar que a idade difere significativamente entre os sexos.")
