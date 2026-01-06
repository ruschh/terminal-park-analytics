import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# =========================
# Config
# =========================
st.set_page_config(page_title="Monitoramento de Terminais", layout="wide")

CIELO_BLUE = "#005AA3"
CIELO_BLUE_DARK = "#003B6B"
CIELO_BG = "#F5F7FB"
CIELO_TEXT = "#0B1F35"

#================================
# CSS - Cascading Style Sheets
#================================

st.markdown(
    f"""
    <style>
      /* ===== Base ===== */
      .stApp {{
        background: linear-gradient(180deg, {CIELO_BG} 0%, #FFFFFF 35%);
        color: {CIELO_TEXT};
      }}

      /* Títulos */
      h1, h2, h3 {{
        color: {CIELO_BLUE_DARK} !important;
        letter-spacing: -0.2px;
      }}

      /* Texto geral (evita “apagado”) */
      p, li, label, span, div {{
        color: {CIELO_TEXT};
      }}

      /* ===== Topbar ===== */
      .cielo-topbar {{
        background: {CIELO_BLUE};
        padding: 14px 24px;
        border-radius: 14px;
        color: #FFFFFF !important;
        font-weight: 700;
        font-size: 18px;
        line-height: 1.4px;
        letter-spacing: 0.3px;
        margin: 12px; auto 24px auto;
        max-width: 1100px;
        text-align: center;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
      }}

      /* ===== Sidebar ===== */
      section[data-testid="stSidebar"] {{
        background: #FFFFFF;
        border-right: 1px solid rgba(0,0,0,0.08);
      }}
      section[data-testid="stSidebar"] * {{
        color: {CIELO_TEXT} !important;
        opacity: 1 !important;
      }}

      /* ===== Tabs ===== */
      button[data-baseweb="tab"] {{
        font-weight: 700 !important;
        color: {CIELO_TEXT} !important;
      }}
      button[data-baseweb="tab"][aria-selected="true"] {{
        border-bottom: 3px solid {CIELO_BLUE} !important;
        color: {CIELO_BLUE_DARK} !important;
      }}

      /* ===== KPI (st.metric) ===== */
      div[data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid rgba(0,0,0,0.08);
        padding: 14px 16px;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
      }}

      /* Label do KPI */
      div[data-testid="stMetricLabel"] * {{
        color: {CIELO_BLUE_DARK} !important;
        opacity: 1 !important;
        font-weight: 700 !important;
      }}

      /* Valor do KPI */
      div[data-testid="stMetricValue"] * {{
        color: {CIELO_TEXT} !important;
        opacity: 1 !important;
        font-weight: 800 !important;
      }}

      /* Delta (quando existir) */
      div[data-testid="stMetricDelta"] * {{
        opacity: 1 !important;
        font-weight: 700 !important;
      }}

      /* ===== “Card” para gráficos Plotly ===== */
      .plotly-card {{
        background: #FFFFFF;
        border: 1px solid rgba(0,0,0,0.08);
        padding: 10px 12px;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        margin-bottom: 14px;
      }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="cielo-topbar">
      Monitoramento do Parque de Terminais
    </div>
    """,
    unsafe_allow_html=True
)


st.title("Monitoramento do Parque de Terminais — Protótipo")
st.caption("Dashboard interativo com métricas estratégicas, filtros globais e visões operacionais.")

# Diretório padrão (ajuste conforme seu projeto)
DATA_DIR_DEFAULT = "/home/rusch/Área de trabalho/Projeto_Cielo/Desafio2/dados"

# =========================
# Helpers
# =========================
@st.cache_data(show_spinner=False)
def load_data(data_dir: str):
    tb_parque = pd.read_csv(os.path.join(data_dir, "tb_parque_terminais.csv"))
    tb_transacoes = pd.read_csv(os.path.join(data_dir, "tb_transacoes.csv"))
    tb_chamados = pd.read_csv(os.path.join(data_dir, "tb_chamados.csv"))

    # Parse datas
    tb_parque["data_registro"] = pd.to_datetime(tb_parque["data_registro"])
    tb_transacoes["data_transacao"] = pd.to_datetime(tb_transacoes["data_transacao"])
    tb_chamados["data_abertura_chamado"] = pd.to_datetime(tb_chamados["data_abertura_chamado"])
    tb_chamados["data_fechamento_chamado"] = pd.to_datetime(tb_chamados["data_fechamento_chamado"])

    return tb_parque, tb_transacoes, tb_chamados


def add_reaberto(tb_chamados: pd.DataFrame, limite_dias: int = 7) -> pd.DataFrame:
    """
    Heurística: chamado reaberto quando um novo chamado para o mesmo terminal
    abre até 'limite_dias' após o fechamento do chamado anterior.
    """
    df = tb_chamados.sort_values(["id_terminal", "data_abertura_chamado"]).copy()
    df["data_fechamento_anterior"] = df.groupby("id_terminal")["data_fechamento_chamado"].shift(1)
    df["dias_entre_chamados"] = (df["data_abertura_chamado"] - df["data_fechamento_anterior"]).dt.days
    df["reaberto"] = (df["dias_entre_chamados"] <= limite_dias).fillna(False)
    return df


def compute_base_corr_mes(tb_parque, tb_transacoes, tb_chamados):
    # transações por mês ativo (terminal)
    tx = tb_transacoes.copy()
    tx["mes_ref"] = tx["data_transacao"].dt.to_period("M")

    tx_por_terminal = (
        tx.groupby("id_terminal")
          .agg(qtd_transacoes=("id_terminal", "size"),
               meses_ativos=("mes_ref", "nunique"))
          .reset_index()
    )
    tx_por_terminal["meses_ativos"] = tx_por_terminal["meses_ativos"].clip(lower=1)
    tx_por_terminal["tx_por_mes"] = tx_por_terminal["qtd_transacoes"] / tx_por_terminal["meses_ativos"]

    # chamados por terminal
    ch_por_terminal = (
        tb_chamados.groupby("id_terminal")
        .size()
        .reset_index(name="qtd_chamados")
    )

    base = (
        tx_por_terminal
        .merge(ch_por_terminal, on="id_terminal", how="left")
        .fillna({"qtd_chamados": 0})
        .merge(tb_parque[["id_terminal", "modelo", "segmento_cliente", "setor_cliente"]], on="id_terminal", how="left")
    )
    return base


def date_filter_transacoes(tb_transacoes, start_date, end_date):
    if start_date is None or end_date is None:
        return tb_transacoes
    mask = (tb_transacoes["data_transacao"] >= pd.to_datetime(start_date)) & (tb_transacoes["data_transacao"] <= pd.to_datetime(end_date))
    return tb_transacoes.loc[mask].copy()


def date_filter_chamados(tb_chamados, start_date, end_date):
    if start_date is None or end_date is None:
        return tb_chamados
    mask = (tb_chamados["data_abertura_chamado"] >= pd.to_datetime(start_date)) & (tb_chamados["data_abertura_chamado"] <= pd.to_datetime(end_date))
    return tb_chamados.loc[mask].copy()


# =========================
# Sidebar: Data + Filters
# =========================
st.sidebar.header("Dados")

data_dir = st.sidebar.text_input("Diretório dos CSVs", value=DATA_DIR_DEFAULT)
use_upload = st.sidebar.toggle("Usar upload (em vez do diretório)", value=False)

if use_upload:
    up_parque = st.sidebar.file_uploader("tb_parque_terminais.csv", type=["csv"])
    up_tx = st.sidebar.file_uploader("tb_transacoes.csv", type=["csv"])
    up_ch = st.sidebar.file_uploader("tb_chamados.csv", type=["csv"])
    if up_parque and up_tx and up_ch:
        tb_parque = pd.read_csv(up_parque)
        tb_transacoes = pd.read_csv(up_tx)
        tb_chamados = pd.read_csv(up_ch)
        tb_parque["data_registro"] = pd.to_datetime(tb_parque["data_registro"])
        tb_transacoes["data_transacao"] = pd.to_datetime(tb_transacoes["data_transacao"])
        tb_chamados["data_abertura_chamado"] = pd.to_datetime(tb_chamados["data_abertura_chamado"])
        tb_chamados["data_fechamento_chamado"] = pd.to_datetime(tb_chamados["data_fechamento_chamado"])
    else:
        st.info("Faça upload dos 3 CSVs para iniciar.")
        st.stop()
else:
    try:
        tb_parque, tb_transacoes, tb_chamados = load_data(data_dir)
    except Exception as e:
        st.error(f"Falha ao carregar arquivos no diretório informado. Detalhes: {e}")
        st.stop()

# parâmetros operacionais
st.sidebar.header("Parâmetros")
reabertura_dias = st.sidebar.slider("Janela para 'reabertura' (dias)", min_value=1, max_value=30, value=7)
sla_dias = st.sidebar.slider("SLA (dias) para violação", min_value=1, max_value=30, value=5)
inatividade_dias = st.sidebar.slider("Inatividade (dias) para alerta/KPI", min_value=1, max_value=180, value=30)

# período (aplicado em transações e chamados)
st.sidebar.header("Filtros Globais")
min_dt = min(tb_transacoes["data_transacao"].min(), tb_chamados["data_abertura_chamado"].min())
max_dt = max(tb_transacoes["data_transacao"].max(), tb_chamados["data_abertura_chamado"].max())
period = st.sidebar.date_input("Período de análise", value=(min_dt.date(), max_dt.date()), format="DD/MM/YYYY")
start_date, end_date = (period[0], period[1]) if isinstance(period, (list, tuple)) and len(period) == 2 else (None, None)

# filtros categóricos (derivados do parque)
modelos = sorted(tb_parque["modelo"].dropna().unique().tolist())
segmentos = sorted(tb_parque["segmento_cliente"].dropna().unique().tolist())
setores = sorted(tb_parque["setor_cliente"].dropna().unique().tolist())

f_modelo = st.sidebar.multiselect("Modelo", modelos, default=modelos)
f_segmento = st.sidebar.multiselect("Segmento", segmentos, default=segmentos)
f_setor = st.sidebar.multiselect("Setor", setores, default=setores)

# aplicar filtros ao parque
parque_f = tb_parque[
    tb_parque["modelo"].isin(f_modelo) &
    tb_parque["segmento_cliente"].isin(f_segmento) &
    tb_parque["setor_cliente"].isin(f_setor)
].copy()

# aplicar filtros temporais a transações e chamados, e depois garantir terminais no filtro do parque
tx_f = date_filter_transacoes(tb_transacoes, start_date, end_date)
ch_f = date_filter_chamados(tb_chamados, start_date, end_date)

tx_f = tx_f[tx_f["id_terminal"].isin(parque_f["id_terminal"])].copy()
ch_f = ch_f[ch_f["id_terminal"].isin(parque_f["id_terminal"])].copy()

# enriquecer chamados com reaberto
ch_f = add_reaberto(ch_f, limite_dias=reabertura_dias)

# base de correlação (terminal)
base_corr_mes = compute_base_corr_mes(parque_f, tx_f, ch_f)

# =========================
# Métricas 1–10
# =========================
# 1) % de modelos por setor
dist_modelo_setor = (
    parque_f.groupby(["setor_cliente", "modelo"])
    .size()
    .reset_index(name="qtd")
)
dist_modelo_setor["percentual"] = (
    dist_modelo_setor.groupby("setor_cliente")["qtd"]
    .transform(lambda x: 100 * x / x.sum())
).round(2)

# 2) média transações mensais por modelo
m2_media_tx_modelo = (
    base_corr_mes.groupby("modelo")["tx_por_mes"]
    .mean()
    .reset_index(name="media_transacoes_mensais")
)
m2_media_tx_modelo["media_transacoes_mensais"] = m2_media_tx_modelo["media_transacoes_mensais"].round(2)

# 3) modelos com mais chamados últimos 3 meses (relativo ao fim do período filtrado)
ref_end = pd.to_datetime(end_date) if end_date is not None else ch_f["data_abertura_chamado"].max()
data_limite_3m = ref_end - pd.DateOffset(months=3)
m3_chamados_3m = (
    ch_f[ch_f["data_abertura_chamado"] >= data_limite_3m]
    .merge(parque_f[["id_terminal", "modelo"]], on="id_terminal", how="inner")
    .groupby("modelo")
    .size()
    .reset_index(name="qtd_chamados_3m")
    .sort_values("qtd_chamados_3m", ascending=False)
)

# 4) tempo médio resolução por modelo
ch_f["tempo_resolucao_dias"] = (ch_f["data_fechamento_chamado"] - ch_f["data_abertura_chamado"]).dt.days
m4_tempo_medio = (
    ch_f.merge(parque_f[["id_terminal", "modelo"]], on="id_terminal", how="inner")
    .groupby("modelo")["tempo_resolucao_dias"]
    .mean()
    .reset_index(name="tempo_medio_resolucao_dias")
)
m4_tempo_medio["tempo_medio_resolucao_dias"] = m4_tempo_medio["tempo_medio_resolucao_dias"].round(2)

# 5) % terminais inativos (sem transação nos últimos X dias)
ultima_tx = (
    tx_f.groupby("id_terminal")["data_transacao"]
    .max()
    .reset_index(name="ultima_transacao")
)
parque_atividade = parque_f.merge(ultima_tx, on="id_terminal", how="left")
parque_atividade["dias_sem_transacao"] = (pd.Timestamp("today") - parque_atividade["ultima_transacao"]).dt.days
m5_perc_inativos = (parque_atividade["dias_sem_transacao"] > inatividade_dias).mean() * 100
m5_perc_inativos = float(np.round(m5_perc_inativos, 2))

# 6) % terminais críticos (mínimo % de terminais que fazem 80% das transações)
tx_por_terminal_sorted = tx_f.groupby("id_terminal").size().sort_values(ascending=False)
if len(tx_por_terminal_sorted) > 0:
    tx_cumsum = tx_por_terminal_sorted.cumsum() / tx_por_terminal_sorted.sum()
    k_star = int((tx_cumsum < 0.8).sum() + 1)
    n_terms = len(tx_por_terminal_sorted)
    m6_perc_criticos = np.round(100 * k_star / n_terms, 2)
else:
    m6_perc_criticos = 0.0

# 7) taxa chamados por 1000 transações (por modelo)
tx_term = tx_f.groupby("id_terminal").size().reset_index(name="tx")
ch_term = ch_f.groupby("id_terminal").size().reset_index(name="ch")
m7_taxa = (
    tx_term.merge(ch_term, on="id_terminal", how="left")
    .fillna({"ch": 0})
    .merge(parque_f[["id_terminal", "modelo"]], on="id_terminal", how="inner")
    .groupby("modelo")
    .agg(total_tx=("tx", "sum"), total_chamados=("ch", "sum"))
    .assign(taxa_chamados_1000=lambda x: (x["total_chamados"] / x["total_tx"]) * 1000)
    .reset_index()
)
m7_taxa["taxa_chamados_1000"] = m7_taxa["taxa_chamados_1000"].replace([np.inf, -np.inf], np.nan).fillna(0).round(2)

# 8) % chamados reabertos por modelo
m8_reabertura = (
    ch_f.merge(parque_f[["id_terminal", "modelo"]], on="id_terminal", how="inner")
    .groupby("modelo")
    .agg(total_chamados=("id_chamado", "count"), chamados_reabertos=("reaberto", "sum"))
    .assign(percentual_reabertura=lambda x: 100 * x["chamados_reabertos"] / x["total_chamados"])
    .reset_index()
)
m8_reabertura["percentual_reabertura"] = m8_reabertura["percentual_reabertura"].round(2)

# 9) % SLA violado por modelo
m9_sla = (
    ch_f.merge(parque_f[["id_terminal", "modelo"]], on="id_terminal", how="inner")
    .assign(sla_violado=lambda x: x["tempo_resolucao_dias"] > sla_dias)
    .groupby("modelo")["sla_violado"]
    .mean()
    .mul(100)
    .reset_index(name="percentual_sla_violado")
)
m9_sla["percentual_sla_violado"] = m9_sla["percentual_sla_violado"].round(2)

# 10) índice de criticidade (score composto) — sem sklearn (para evitar dependência)
# normalização min-max manual
score_df = base_corr_mes.merge(
    ch_f.groupby("id_terminal")["tempo_resolucao_dias"].mean().reset_index(name="tempo_medio_res"),
    on="id_terminal",
    how="left"
).fillna({"tempo_medio_res": 0})

def minmax(s: pd.Series):
    s = s.astype(float)
    if s.max() == s.min():
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

score_df["tx_norm"] = minmax(score_df["tx_por_mes"])
score_df["ch_norm"] = minmax(score_df["qtd_chamados"])
score_df["tr_norm"] = minmax(score_df["tempo_medio_res"])

# pesos (ajustáveis)
w1, w2, w3 = 0.5, 0.3, 0.2
score_df["criticidade_score"] = (w1 * score_df["tx_norm"] + w2 * score_df["ch_norm"] + w3 * score_df["tr_norm"]).round(4)

top_criticos = score_df.sort_values("criticidade_score", ascending=False).head(20)

# =========================
# Abas (Views)
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Visão Geral", "Uso", "Confiabilidade", "Priorização", "Observabilidade"])

# -------------------------
# Tab 1: Visão Geral
# -------------------------
with tab1:
    colA, colB, colC, colD, colE = st.columns(5)

    total_terminais = int(parque_f["id_terminal"].nunique())
    total_tx = int(len(tx_f))
    total_ch = int(len(ch_f))

    colA.metric("Terminais (filtrados)", f"{total_terminais:,}".replace(",", "."))
    colB.metric("Transações (período)", f"{total_tx:,}".replace(",", "."))
    colC.metric("Chamados (período)", f"{total_ch:,}".replace(",", "."))
    colD.metric("% Inativos (>X dias)", f"{m5_perc_inativos:.2f}%")
    colE.metric("% Terminais críticos (80% tx)", f"{m6_perc_criticos:.2f}%")

    st.subheader("Distribuição de Modelos por Setor (Métrica 1)")
    fig = px.bar(
        dist_modelo_setor,
        x="setor_cliente",
        y="percentual",
        color="modelo",
        barmode="stack",
        text="percentual",
        title="Percentual de Modelos por Setor do Cliente"
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="inside")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Tab 2: Uso
# -------------------------
with tab2:
    st.subheader("Média de Transações Mensais por Modelo (Métrica 2)")
    fig2 = px.bar(
        m2_media_tx_modelo.sort_values("media_transacoes_mensais", ascending=False),
        x="modelo",
        y="media_transacoes_mensais",
        text="media_transacoes_mensais",
        title="Média de Transações por Mês Ativo — por Modelo"
    )
    fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Concentração de Transações (Pareto) — Métrica 6")
    if len(tx_por_terminal_sorted) > 0:
        tx_cumsum_vals = (tx_por_terminal_sorted.cumsum() / tx_por_terminal_sorted.sum()).values
        figp = px.line(
            x=np.arange(1, len(tx_cumsum_vals) + 1),
            y=tx_cumsum_vals,
            title="Curva de Concentração do Volume de Transações (Pareto)",
            labels={"x": "Terminais (ordenados por volume)", "y": "Proporção acumulada de transações"}
        )
        figp.add_hline(y=0.8, line_dash="dash")
        st.plotly_chart(figp, use_container_width=True)
    else:
        st.info("Sem transações no período filtrado para exibir Pareto.")

    st.subheader("Inatividade — Distribuição de Dias sem Transação (Métrica 5)")
    figi = px.histogram(
        parque_atividade.dropna(subset=["dias_sem_transacao"]),
        x="dias_sem_transacao",
        nbins=30,
        title="Distribuição de Dias sem Transações por Terminal",
        labels={"dias_sem_transacao": "Dias sem transações"}
    )
    st.plotly_chart(figi, use_container_width=True)

# -------------------------
# Tab 3: Confiabilidade
# -------------------------
with tab3:
    st.subheader("Modelos com Mais Chamados (Últimos 3 meses) — Métrica 3")
    fig3 = px.bar(
        m3_chamados_3m,
        x="modelo",
        y="qtd_chamados_3m",
        text="qtd_chamados_3m",
        title="Chamados por Modelo — Últimos 3 Meses"
    )
    fig3.update_traces(textposition="outside")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Tempo Médio de Resolução (dias) — por Modelo (Métrica 4)")
    fig4 = px.bar(
        m4_tempo_medio.sort_values("tempo_medio_resolucao_dias", ascending=False),
        x="modelo",
        y="tempo_medio_resolucao_dias",
        text="tempo_medio_resolucao_dias",
        title="Tempo Médio de Resolução de Chamados — por Modelo"
    )
    fig4.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Taxa de Chamados por 1.000 Transações — por Modelo (Métrica 7)")
    fig7 = px.bar(
        m7_taxa.sort_values("taxa_chamados_1000", ascending=False),
        x="modelo",
        y="taxa_chamados_1000",
        text="taxa_chamados_1000",
        title="Taxa de Chamados por 1.000 Transações"
    )
    fig7.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig7, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("% de Reabertura — por Modelo (Métrica 8)")
        fig8 = px.bar(
            m8_reabertura.sort_values("percentual_reabertura", ascending=False),
            x="modelo",
            y="percentual_reabertura",
            text="percentual_reabertura",
            title="Percentual de Chamados Reabertos — por Modelo"
        )
        fig8.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig8, use_container_width=True)

    with col2:
        st.subheader("% SLA Violado — por Modelo (Métrica 9)")
        fig9 = px.bar(
            m9_sla.sort_values("percentual_sla_violado", ascending=False),
            x="modelo",
            y="percentual_sla_violado",
            text="percentual_sla_violado",
            title="Percentual de Chamados com SLA Violado — por Modelo"
        )
        fig9.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig9, use_container_width=True)

    st.subheader("Correlação (Spearman) — Chamados vs Transações Mensais (Heatmap)")
    corr_ch_tx = base_corr_mes[["qtd_chamados", "tx_por_mes"]].corr(method="spearman")
    fig_hm = px.imshow(
        corr_ch_tx,
        text_auto=".2f",
        aspect="auto",
        title="Heatmap da Correlação (Spearman)"
    )
    st.plotly_chart(fig_hm, use_container_width=True)

# -------------------------
# Tab 4: Priorização
# -------------------------
with tab4:
    st.subheader("Top 20 Terminais por Índice de Criticalidade (Métrica 10)")
    st.dataframe(
        top_criticos[[
            "id_terminal", "modelo", "segmento_cliente", "setor_cliente",
            "tx_por_mes", "qtd_chamados", "criticidade_score"
        ]].reset_index(drop=True),
        use_container_width=True
    )

    st.subheader("Chamados vs Transações Mensais (Dispersão) — suporte à priorização")
    fig_sc = px.scatter(
        score_df,
        x="qtd_chamados",
        y="tx_por_mes",
        color="modelo",
        hover_data=["id_terminal", "segmento_cliente", "setor_cliente", "criticidade_score"],
        title="Relação entre Chamados Técnicos e Transações por Mês Ativo (por Terminal)"
    )
    fig_sc.update_xaxes(dtick=1, range=[-0.5, 10.5])
    st.plotly_chart(fig_sc, use_container_width=True)

st.divider()
st.caption("Protótipo alinhado às etapas do desafio: EDA → Métricas Estratégicas → Dashboard → (Etapa 4) Observabilidade/Alertas documentados.")

# -------------------------
# Tab 5: Observabilidade
# -------------------------
with tab5:
    st.subheader("Observabilidade — Regras de Alerta (documentadas)")

    st.markdown(
        """
        Esta aba documenta regras de observabilidade sugeridas a partir da EDA e das métricas estratégicas.
        Os alertas não são implementados (disparo automático e envio de notificações), conforme o escopo do desafio,
        mas são apresentados com critérios objetivos e canais recomendados, além de uma lista de entidades em violação
        no período/recorte selecionado.
        """
    )

    # ==========================================================
    # Regra 1 — Inatividade (exemplo: > 48h)
    # ==========================================================
    st.markdown("### Regra 1 — Inatividade Prolongada do Terminal")
    st.markdown(
        f"""
        **Critério de disparo:** gerar alerta se um terminal não registrar transações por mais de **48 horas**  
        (equivalente a **2 dias**) — ou, alternativamente, usar o limiar configurado no dashboard (**{inatividade_dias} dias**).

        **Canal de notificação sugerido:** Microsoft Teams (canal de Operações) e e-mail para equipe responsável,
        com abertura automática de ticket (ServiceNow/Jira) em cenários de criticidade.
        """
    )

    # Para o protótipo: exibir duas visões:
    # (A) > 48h (2 dias) e (B) > inatividade_dias (parâmetro do KPI)
    inativos_48h = parque_atividade.copy()
    inativos_48h["dias_sem_transacao"] = inativos_48h["dias_sem_transacao"].fillna(np.inf)
    alerta_inativos_48h = inativos_48h[inativos_48h["dias_sem_transacao"] > 2].copy()

    inativos_param = parque_atividade.copy()
    inativos_param["dias_sem_transacao"] = inativos_param["dias_sem_transacao"].fillna(np.inf)
    alerta_inativos_param = inativos_param[inativos_param["dias_sem_transacao"] > inatividade_dias].copy()

    # ==========================================================
    # Regra 2 — Violação de SLA
    # ==========================================================
    st.markdown("### Regra 2 — Violação de SLA em Chamados Técnicos")
    st.markdown(
        f"""
        **Critério de disparo:** gerar alerta se o tempo de resolução de um chamado técnico exceder o SLA definido.
        Neste protótipo, o SLA é parametrizável e está definido como **{sla_dias} dias**.

        **Canal de notificação sugerido:** e-mail para coordenação de suporte + destaque visual no dashboard,
        com notificação em Teams para chamados críticos e abertura de ticket para priorização.
        """
    )

    # Chamados fora do SLA (no recorte atual)
    # Considera chamados com tempo_resolucao_dias já calculado
    ch_sla = ch_f.copy()
    ch_sla["tempo_resolucao_dias"] = ch_sla["tempo_resolucao_dias"].fillna(
        (ch_sla["data_fechamento_chamado"] - ch_sla["data_abertura_chamado"]).dt.days
    )
    ch_fora_sla = ch_sla[ch_sla["tempo_resolucao_dias"] > sla_dias].copy()

    # Agregar por terminal: quantos chamados fora do SLA e qual maior tempo
    term_sla = (
        ch_fora_sla.groupby("id_terminal")
        .agg(
            chamados_fora_sla=("id_chamado", "count"),
            max_tempo_resolucao=("tempo_resolucao_dias", "max")
        )
        .reset_index()
        .merge(parque_f[["id_terminal", "modelo", "segmento_cliente", "setor_cliente"]], on="id_terminal", how="left")
        .sort_values(["chamados_fora_sla", "max_tempo_resolucao"], ascending=False)
    )

    st.markdown("### Top Terminais Mais Atrasados (SLA) — Prioridade de Ação")

    # Top N terminais com maior atraso (dias acima do SLA)
    TOP_N = 15

    # 1) Garante a métrica "dias acima do SLA"
    term_sla_plot = term_sla.copy()
    term_sla_plot["dias_acima_sla"] = term_sla_plot["max_tempo_resolucao"] - sla_dias

    # 2) Filtra apenas violações (>0)
    term_sla_plot = term_sla_plot[term_sla_plot["dias_acima_sla"] > 0].copy()

    # 3) Cria rótulo curto para o eixo Y
    term_sla_plot["terminal_curto"] = term_sla_plot["id_terminal"].astype(str).str.slice(0, 8) + "…"

    # 4) Seleciona Top N (maior atraso; em empate, mais chamados fora do SLA)
    top_atrasados = (
        term_sla_plot
        .sort_values(["dias_acima_sla", "chamados_fora_sla"], ascending=False)
        .head(TOP_N)
        .copy()
    )

    # 5) Plot (horizontal; ordenar para ficar “crescente” no eixo visual)
    fig_top = px.bar(
        top_atrasados.sort_values("dias_acima_sla", ascending=True),
        x="dias_acima_sla",
        y="terminal_curto",
        orientation="h",
        color="modelo",
        text="dias_acima_sla",
        hover_data=["id_terminal", "segmento_cliente", "setor_cliente", "chamados_fora_sla"],
        title=f"Top {TOP_N} Terminais — Dias Acima do SLA"
    )
    fig_top.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_top.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    fig_top.update_layout(
        xaxis_title="Dias acima do SLA",
        yaxis_title="Terminal",
        legend_title_text="Modelo do Terminal",
        bargap=0.35
    )

    st.plotly_chart(fig_top, use_container_width=True)



    st.markdown("### Heatmap por Modelo — Percentual de Violação")

    # -------------------------
    # 2.1 SLA: % de chamados fora do SLA por modelo
    # -------------------------
    # total de chamados por modelo (no recorte)
    ch_total_modelo = (
        ch_f.merge(parque_f[["id_terminal", "modelo"]], on="id_terminal", how="inner")
        .groupby("modelo")["id_chamado"]
        .count()
        .reset_index(name="total_chamados")
    )

    # chamados fora do SLA por modelo
    ch_sla_modelo = (
        ch_fora_sla.merge(parque_f[["id_terminal", "modelo"]], on="id_terminal", how="inner")
        .groupby("modelo")["id_chamado"]
        .count()
        .reset_index(name="chamados_fora_sla")
    )

    sla_heat = (
        ch_total_modelo.merge(ch_sla_modelo, on="modelo", how="left")
        .fillna({"chamados_fora_sla": 0})
    )
    sla_heat["perc_violacao_sla"] = (100 * sla_heat["chamados_fora_sla"] / sla_heat["total_chamados"]).round(2)

    # Transformar em matriz 1xN para heatmap
    sla_mat = sla_heat.set_index("modelo")[["perc_violacao_sla"]].T
    sla_mat.index = [f"% SLA violado (> {sla_dias}d)"]

    fig_hm_sla = px.imshow(
        sla_mat,
        text_auto=".2f",
        aspect="auto",
        title="Heatmap — Percentual de Violação de SLA por Modelo"
    )
    st.plotly_chart(fig_hm_sla, use_container_width=True)

    # -------------------------
    # 2.2 Inatividade: % de terminais inativos por modelo
    # -------------------------
    inatividade_por_modelo = (
        alerta_inativos_param
        .assign(em_alerta_inatividade=True)
        .groupby("modelo")["id_terminal"]
        .nunique()
        .reset_index(name="terminais_em_alerta")
        .merge(
            parque_f.groupby("modelo")["id_terminal"].nunique().reset_index(name="total_terminais"),
            on="modelo",
            how="right"
        )
        .fillna({"terminais_em_alerta": 0})
    )
    inatividade_por_modelo["perc_inatividade"] = (
        100 * inatividade_por_modelo["terminais_em_alerta"] / inatividade_por_modelo["total_terminais"]
    ).round(2)

    inatividade_mat = inatividade_por_modelo.set_index("modelo")[["perc_inatividade"]].T
    inatividade_mat.index = [f"% Inatividade (> {inatividade_dias}d)"]

    fig_hm_inat = px.imshow(
        inatividade_mat,
        text_auto=".2f",
        aspect="auto",
        title="Heatmap — Percentual de Terminais em Alerta de Inatividade por Modelo"
    )
    st.plotly_chart(fig_hm_inat, use_container_width=True)


    # ==========================================================
    # KPIs de Observabilidade
    # ==========================================================
    st.markdown("### Indicadores (estado atual do recorte)")

    colA, colB, colC = st.columns(3)
    colA.metric("Terminais sem transações > 48h", f"{alerta_inativos_48h['id_terminal'].nunique():,}".replace(",", "."))
    colB.metric(f"Terminais sem transações > {inatividade_dias} dias", f"{alerta_inativos_param['id_terminal'].nunique():,}".replace(",", "."))
    colC.metric("Terminais com chamados fora do SLA", f"{term_sla['id_terminal'].nunique():,}".replace(",", "."))

    # ==========================================================
    # Tabelas de violação
    # ==========================================================
    st.markdown("### Lista de Terminais em Alerta — Inatividade (>48h)")
    cols_inatividade = ["id_terminal", "modelo", "segmento_cliente", "setor_cliente", "ultima_transacao", "dias_sem_transacao"]
    st.dataframe(
        alerta_inativos_48h[cols_inatividade]
        .sort_values("dias_sem_transacao", ascending=False)
        .reset_index(drop=True),
        use_container_width=True
    )

    st.markdown(f"### Lista de Terminais em Alerta — Inatividade (>{inatividade_dias} dias)")
    st.dataframe(
        alerta_inativos_param[cols_inatividade]
        .sort_values("dias_sem_transacao", ascending=False)
        .reset_index(drop=True),
        use_container_width=True
    )

    st.markdown("### Lista de Terminais em Alerta — Chamados fora do SLA")
    st.dataframe(
        term_sla.reset_index(drop=True),
        use_container_width=True
    )

    st.info(
        "Observação: estes alertas estão documentados conforme o desafio. "
        "Em ambiente produtivo, a automação de notificações seria integrada a um orquestrador (ex.: jobs agendados) "
        "e a canais corporativos (Teams/e-mail/ticketing)."
    )
