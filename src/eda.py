# importações

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

DATA_DIR = "/home/rusch/Área de trabalho/Projeto_Cielo/Desafio2/dados"

tb_parque = pd.read_csv(                                 # abre, interpreta seu conteúdo e converte o arquivo em um objeto do tipo DataFrame
    os.path.join(DATA_DIR, "tb_parque_terminais.csv"),   # O caminho completo até o arquivo
    parse_dates=["data_registro"]                        # instrui explicitamente o pandas a interpretar a coluna data_registro como um tipo 
)                                                        # de dado temporal (datetime), e não como uma simples string)

tb_transacoes = pd.read_csv(
    os.path.join(DATA_DIR, "tb_transacoes.csv"),
    parse_dates=["data_transacao"]
)

tb_chamados = pd.read_csv(
    os.path.join(DATA_DIR, "tb_chamados.csv"),
    parse_dates=["data_abertura_chamado", "data_fechamento_chamado"]
)


tb_parque.head()

tb_transacoes.head()

tb_chamados.head()

tb_parque.info()

tb_transacoes.info

tb_chamados.info()


# Percentual de transações com terminal válido
tx_validas = tb_transacoes["id_terminal"].isin(tb_parque["id_terminal"]).mean() * 100

# Percentual de chamados com terminal válido
ch_validos = tb_chamados["id_terminal"].isin(tb_parque["id_terminal"]).mean() * 100

print(tx_validas.round(2), ch_validos.round(2))

(tb_parque["modelo"].value_counts(normalize=True) * 100).round(2)

sns.countplot(data=tb_parque, x="modelo")
plt.title("Distribuição dos modelos de terminais")
plt.show()


# Tabela de contingência
distrib_segmento_setor = pd.crosstab(
    tb_parque["segmento_cliente"], # linhas da tabela de contigência
    tb_parque["setor_cliente"],    # colunas da tabela de contigência
    normalize="index"
) * 100

(distrib_segmento_setor).round(2)


# Gráficos
sns.histplot(tb_transacoes["valor_transacao"], bins=50)
plt.title("Distribuição do valor das transações")
plt.xlabel("Valor das transações")
plt.show()

sns.countplot(data=tb_chamados, x="tipo_chamado")
plt.title("Distribuição dos tipos de chamados")
plt.show()

tb_chamados["tempo_resolucao_dias"] = (
    tb_chamados["data_fechamento_chamado"] -
    tb_chamados["data_abertura_chamado"]
).dt.days

plt.figure(figsize=(12, 5))
sns.countplot(
    data=tb_parque,
    x="modelo",
    order=tb_parque["modelo"].value_counts().index
)
plt.title("Distribuição dos Modelos de Terminais")
plt.xlabel("Modelo do Terminal")
plt.ylabel("Quantidade de Terminais")
plt.tight_layout()
plt.show()


# Volume de transações por modelo
tx_por_modelo = (
    tb_transacoes
    .merge(tb_parque[["id_terminal", "modelo"]], on="id_terminal", how="inner")
    .groupby("modelo")
    .size()
    .reset_index(name="qtd_transacoes")
)

plt.figure(figsize=(12, 5))
sns.barplot(
    data=tx_por_modelo,
    x="modelo",
    y="qtd_transacoes",
    order=tx_por_modelo.sort_values("qtd_transacoes", ascending=False)["modelo"]
)
plt.title("Volume de Transações por Modelo de Terminal")
plt.xlabel("Modelo do Terminal")
plt.ylabel("Quantidade de Transações")
plt.tight_layout()
plt.show()


# Chamados por modelo de terminal
chamados_modelo = (
    tb_chamados
    .merge(tb_parque[["id_terminal", "modelo"]], on="id_terminal", how="inner")
)

plt.figure(figsize=(12, 6))
sns.countplot(
    data=chamados_modelo,
    x="modelo",
    hue="tipo_chamado"
)
plt.title("Tipos de Chamados por Modelo de Terminal")
plt.xlabel("Modelo do Terminal")
plt.ylabel("Quantidade de Chamados")
plt.legend(title="Tipo de Chamado")
plt.tight_layout()
plt.show()

# Chamados x Volume de transações por terminal

# número total de transações por id_terminal
tx_por_terminal = (
    tb_transacoes
    .groupby("id_terminal")
    .size()
    .reset_index(name="qtd_transacoes")
)

#número total de chamados por id_terminal
ch_por_terminal = (
    tb_chamados
    .groupby("id_terminal")
    .size()
    .reset_index(name="qtd_chamados")
)

# junção dessas duas métricas no mesmo grão (terminal)
base_corr = (
    tx_por_terminal
    .merge(ch_por_terminal, on="id_terminal", how="left")
    .fillna(0)
)

base_corr_modelo = (
    base_corr
    .merge(tb_parque[["id_terminal", "modelo"]], on="id_terminal")
)

plt.figure(figsize=(12, 5))
ax = sns.scatterplot(
    data=base_corr_modelo,
    x="qtd_chamados",
    y="qtd_transacoes",
    hue="modelo"
)

# ajuste do eixo X 
ax.set_xticks(np.arange(0, 11, 1))
ax.set_xlim(-0.5, 10.5)

# --- títulos e rótulos ---
ax.set_title("Relação entre Chamados Técnicos e Volume de Transações")
ax.set_xlabel("Quantidade de Chamados")
ax.set_ylabel("Quantidade de Transações")

# legenda fora do gráfico 
ax.legend(
    title="Modelo",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0
)

plt.tight_layout()
plt.show()


# Transações normalizadas por mês por terminal
# 1) Transações por terminal (total) + meses ativos 
tb_transacoes["mes_ref"] = tb_transacoes["data_transacao"].dt.to_period("M")

tx_por_terminal = (
    tb_transacoes
    .groupby("id_terminal")
    .agg(
        qtd_transacoes=("id_terminal", "size"),
        meses_ativos=("mes_ref", "nunique")
    )
    .reset_index()
)

# evitar divisão por zero (por segurança)
tx_por_terminal["meses_ativos"] = tx_por_terminal["meses_ativos"].clip(lower=1)

# transações normalizadas por mês ativo
tx_por_terminal["tx_por_mes"] = tx_por_terminal["qtd_transacoes"] / tx_por_terminal["meses_ativos"]

# 2) Chamados por terminal 
ch_por_terminal = (
    tb_chamados
    .groupby("id_terminal")
    .size()
    .reset_index(name="qtd_chamados")
)

# 3) Base no mesmo grão (terminal) + modelo 
base_corr_mes = (
    tx_por_terminal
    .merge(ch_por_terminal, on="id_terminal", how="left")
    .fillna({"qtd_chamados": 0})
    .merge(tb_parque[["id_terminal", "modelo"]], on="id_terminal", how="left")
)

# 4) Plot: chamados vs transações/mês 
plt.figure(figsize=(12, 5))
ax = sns.scatterplot(
    data=base_corr_mes,
    x="qtd_chamados",
    y="tx_por_mes",
    hue="modelo"
)

ax.set_xticks(np.arange(0, 11, 1))
ax.set_xlim(-0.5, 10.5)

ax.set_title("Chamados Técnicos vs Transações Normalizadas por Mês Ativo")
ax.set_xlabel("Quantidade de Chamados")
ax.set_ylabel("Transações por mês ativo (média)")

ax.legend(
    title="Modelo",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0
)

plt.tight_layout()
plt.show()

# Chamados x Transações por modelo
corr_modelo = (
    base_corr
    .merge(tb_parque[["id_terminal", "modelo"]], on="id_terminal")
    .groupby("modelo")
    .agg(
        media_transacoes=("qtd_transacoes", "mean"),
        media_chamados=("qtd_chamados", "mean")
    )
    .reset_index()
)

corr_modelo

# Correlação numérica (Spearman + robusta)
corr_ch_tx = base_corr_mes[["qtd_chamados", "tx_por_mes"]].corr(method="spearman")

corr_ch_tx

# Correlação numérica (Spearman + robusta)
corr_ch_tx = base_corr_mes[["qtd_chamados", "tx_por_mes"]].corr(method="spearman")

corr_ch_tx

# Tipo de chamado dominante por terminal
tipo_dominante = (
    tb_chamados
    .groupby(["id_terminal", "tipo_chamado"])
    .size()
    .reset_index(name="qtd")
    .sort_values(["id_terminal", "qtd"], ascending=False)
    .drop_duplicates("id_terminal")
)

base_tipo = (
    base_corr_mes
    .merge(tipo_dominante[["id_terminal", "tipo_chamado"]],
           on="id_terminal", how="left")
)

plt.figure(figsize=(12, 5))
sns.boxplot(
    data=base_tipo,
    x="tipo_chamado",
    y="tx_por_mes"
)

plt.title("Distribuição das Transações Mensais por Tipo de Chamado Dominante")
plt.xlabel("Tipo de Chamado Dominante")
plt.ylabel("Transações por mês ativo")
plt.tight_layout()
plt.show()

# Concentração de transações por terminal (efeito Pareto)

tx_por_terminal_sorted = (
    tb_transacoes
    .groupby("id_terminal")
    .size()
    .sort_values(ascending=False)
)

tx_por_terminal_sorted.head(10)

# Curva acumulada (Pareto)
tx_cumsum = tx_por_terminal_sorted.cumsum() / tx_por_terminal_sorted.sum()

plt.figure(figsize=(12, 5))
plt.plot(tx_cumsum.values)
plt.axhline(0.8, color="red", linestyle="--")
plt.title("Curva de Concentração do Volume de Transações (Pareto)")
plt.xlabel("Terminais ordenados por volume")
plt.ylabel("Proporção acumulada de transações")
plt.tight_layout()
plt.show()

# Identificação de Terminais com Baixa Atividade ou Inatividade
ultima_tx = (
    tb_transacoes
    .groupby("id_terminal")["data_transacao"]
    .max()
    .reset_index(name="ultima_transacao")
)

tb_parque_atividade = (
    tb_parque
    .merge(ultima_tx, on="id_terminal", how="left")
)

tb_parque_atividade["dias_sem_transacao"] = (
    pd.Timestamp("today") - tb_parque_atividade["ultima_transacao"]
).dt.days

sns.histplot(tb_parque_atividade["dias_sem_transacao"].dropna(), bins=30)
plt.title("Distribuição de Dias sem Transações por Terminal")
plt.xlabel("Dias sem transações")
plt.ylabel("Quantidade de terminais")
plt.tight_layout()
plt.show()


# Tempo de Resolução de Chamados por Modelo de Terminal

tb_chamados["tempo_resolucao_dias"] = (
    tb_chamados["data_fechamento_chamado"] -
    tb_chamados["data_abertura_chamado"]
).dt.days

ch_modelo = (
    tb_chamados
    .merge(tb_parque[["id_terminal", "modelo"]], on="id_terminal")
)

plt.figure(figsize=(12, 5))
sns.boxplot(
    data=ch_modelo,
    x="modelo",
    y="tempo_resolucao_dias"
)
plt.title("Distribuição do Tempo de Resolução de Chamados por Modelo")
plt.xlabel("Modelo do Terminal")
plt.ylabel("Tempo de resolução (dias)")
plt.tight_layout()
plt.show()


# Relação entre Idade do Terminal e Volume de Transações

tb_parque["idade_terminal_dias"] = (
    pd.Timestamp("today") - tb_parque["data_registro"]
).dt.days

base_idade = (
    tb_parque[["id_terminal", "idade_terminal_dias"]]
    .merge(tx_por_terminal.reset_index(), on="id_terminal", how="left")
)

sns.scatterplot(
    data=base_idade,
    x="idade_terminal_dias",
    y=0
)
plt.title("Idade do Terminal vs Volume Total de Transações")
plt.xlabel("Idade do terminal (dias)")
plt.ylabel("Quantidade de transações")
plt.tight_layout()
plt.show()


# Definição de Métricas Estratégicas para Monitoramento do Parque de Terminais

# % de Modelos de Terminais por Setor do Cliente
dist_modelo_setor = (
    tb_parque
    .groupby(["setor_cliente", "modelo"])
    .size()
    .reset_index(name="qtd")
)

dist_modelo_setor["percentual"] = (
    dist_modelo_setor
    .groupby("setor_cliente")["qtd"]
    .transform(lambda x: 100 * x / x.sum())
)

dist_modelo_setor = (
    tb_parque
    .groupby(["setor_cliente", "modelo"])
    .size()
    .reset_index(name="qtd")
)

dist_modelo_setor["percentual"] = (
    dist_modelo_setor
    .groupby("setor_cliente")["qtd"]
    .transform(lambda x: 100 * x / x.sum())
)

dist_modelo_setor["percentual"] = dist_modelo_setor["percentual"].round(2)

dist_modelo_setor

# Média de Transações Mensais por Modelo de Terminal
media_tx_modelo = (
    base_corr_mes
    .groupby("modelo")["tx_por_mes"]
    .mean()
    .reset_index(name="media_transacoes_mensais")
)

media_tx_modelo

# Modelos de Terminal com Maior Número de Chamados (Últimos 3 Meses)
data_limite = tb_chamados["data_abertura_chamado"].max() - pd.DateOffset(months=3)

chamados_3m = (
    tb_chamados[tb_chamados["data_abertura_chamado"] >= data_limite]
    .merge(tb_parque[["id_terminal", "modelo"]], on="id_terminal")
    .groupby("modelo")
    .size()
    .reset_index(name="qtd_chamados_3m")
)

chamados_3m

# Tempo Médio de Resolução de Chamados por Modelo de Terminal
tempo_resolucao_modelo = (
    tb_chamados
    .assign(
        tempo_resolucao_dias=lambda x: (
            x["data_fechamento_chamado"] - x["data_abertura_chamado"]
        ).dt.days
    )
    .merge(tb_parque[["id_terminal", "modelo"]], on="id_terminal")
    .groupby("modelo")["tempo_resolucao_dias"]
    .mean()
    .reset_index(name="tempo_medio_resolucao_dias")
)

tempo_resolucao_modelo

# Percentual de Terminais Inativos
limite_inatividade = 30  # dias

perc_inativos = (
    tb_parque_atividade["dias_sem_transacao"] > limite_inatividade
).mean() * 100

print(perc_inativos.round(2))


# Percentual de Terminais Críticos (Alta Concentração de Transações)
total_tx = tx_por_terminal_sorted.sum()
tx_acumulado = tx_por_terminal_sorted.cumsum() / total_tx

perc_terminais_criticos = (tx_acumulado <= 0.8).mean() * 100

print(perc_terminais_criticos.round(2))

# Taxa de Chamados por 1.000 Transações (por Modelo de Terminal)
taxa_chamados_modelo = (
    tb_transacoes
    .groupby("id_terminal")
    .size()
    .reset_index(name="tx")
    .merge(
        tb_chamados.groupby("id_terminal").size().reset_index(name="ch"),
        on="id_terminal",
        how="left"
    )
    .fillna({"ch": 0})
    .merge(tb_parque[["id_terminal", "modelo"]], on="id_terminal")
    .groupby("modelo")
    .agg(
        total_tx=("tx", "sum"),
        total_chamados=("ch", "sum")
    )
    .assign(taxa_chamados_1000=lambda x: (x["total_chamados"] / x["total_tx"]) * 1000)
    .reset_index()
)

taxa_chamados_modelo.round(2)

# Percentual de Chamados Reabertos por Modelo de Terminal
# Garante ordenação temporal correta
tb_chamados = tb_chamados.sort_values(
    ["id_terminal", "data_abertura_chamado"]
)

# Data de fechamento do chamado anterior por terminal
tb_chamados["data_fechamento_anterior"] = (
    tb_chamados
    .groupby("id_terminal")["data_fechamento_chamado"]
    .shift(1)
)

# Diferença de dias entre a nova abertura e o fechamento anterior
tb_chamados["dias_entre_chamados"] = (
    tb_chamados["data_abertura_chamado"] -
    tb_chamados["data_fechamento_anterior"]
).dt.days

# Critério de reabertura (ex.: até 7 dias)
LIMITE_REABERTURA_DIAS = 7

tb_chamados["reaberto"] = (
    tb_chamados["dias_entre_chamados"] <= LIMITE_REABERTURA_DIAS
)

# Chamados sem histórico anterior não são reabertos
tb_chamados["reaberto"] = tb_chamados["reaberto"].fillna(False)

perc_reabertura = (
    tb_chamados
    .merge(tb_parque[["id_terminal", "modelo"]], on="id_terminal")
    .groupby("modelo")
    .agg(
        total_chamados=("id_chamado", "count"),
        chamados_reabertos=("reaberto", "sum")
    )
    .assign(
        percentual_reabertura=lambda x:
        100 * x["chamados_reabertos"] / x["total_chamados"]
    )
    .reset_index()
)

perc_reabertura.round(2)

# Percentual de Terminais com Violação de SLA (Service Level Agreements)
SLA_DIAS = 5

sla_violado = (
    tb_chamados
    .assign(
        tempo_resolucao=lambda x: (
            x["data_fechamento_chamado"] - x["data_abertura_chamado"]
        ).dt.days
    )
    .assign(sla_violado=lambda x: x["tempo_resolucao"] > SLA_DIAS)
    .merge(tb_parque[["id_terminal", "modelo"]], on="id_terminal")
    .groupby("modelo")["sla_violado"]
    .mean()
    .mul(100)
    .reset_index(name="percentual_sla_violado")
)

sla_violado.round(2)

# Índice de Criticalidade do Terminal
from sklearn.preprocessing import MinMaxScaler

score_base = (
    base_corr_mes
    .merge(
        tb_chamados.groupby("id_terminal")
        .agg(tempo_medio_res=("tempo_resolucao_dias", "mean"))
        .reset_index(),
        on="id_terminal",
        how="left"
    )
    .fillna(0)
)

scaler = MinMaxScaler()
score_base[["tx_norm", "ch_norm", "tr_norm"]] = scaler.fit_transform(
    score_base[["tx_por_mes", "qtd_chamados", "tempo_medio_res"]]
)

score_base["criticidade_score"] = (
    0.5 * score_base["tx_norm"] +
    0.3 * score_base["ch_norm"] +
    0.2 * score_base["tr_norm"]
)

score_base[["id_terminal", "criticidade_score"]].sort_values(
    "criticidade_score", ascending=False
).head().round(3)


