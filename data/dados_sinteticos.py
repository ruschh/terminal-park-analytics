#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geração de dados sintéticos para três tabelas:
- tb_terminais.csv   (parque de terminais)
- tb_transacoes.csv  (transações por terminal)
- tb_chamados.csv    (chamados/ordens por terminal)

Ajustado para gerar exatamente:
- tb_transacoes.csv: 100000 linhas
- tb_chamados.csv:   10000 linhas
- tb_terminais.csv:   5000 linhas

As colunas e a ordem seguem os exemplos fornecidos nos CSVs anexados.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from faker import Faker
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Dependência ausente: Faker. Instale com:\n\n"
        "  pip install Faker\n"
    ) from e


# ---------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------
SEED = 42

N_TERMINAIS = 5_000
N_TRANSACOES = 100_000
N_CHAMADOS = 10_000

OUT_DIR = "outputs"
OUT_TERMINAIS = f"{OUT_DIR}/tb_terminais.csv"
OUT_TRANSACOES = f"{OUT_DIR}/tb_transacoes.csv"
OUT_CHAMADOS = f"{OUT_DIR}/tb_chamados.csv"


# Pools de valores (derivados dos exemplos anexados)
SEGMENTOS = ["Empreendedores", "Varejistas", "Grandes contas"]
SETORES = [
    "Tecnologia",
    "Postos de Gasolina",
    "Automotivos",
    "Padarias",
    "Mercados",
    "Farmacias",
    "Shoppings",
    "Restaurantes",
]
MODELOS = ["ME60", "L400", "LIO V3", "S920", "SP930", "L300", "D195", "S920C"]

STATUS_TRANSACAO = ["OK", "CANCELADA", "TIMEOUT"]
STATUS_WEIGHTS = [0.70, 0.20, 0.10]

TIPO_TRANSACAO = ["DEBITO", "CREDITO", "PIX"]
TIPO_WEIGHTS = [0.55, 0.35, 0.10]

TIPO_CHAMADO = ["MANUTENCAO", "SUPORTE"]
TIPO_CHAMADO_WEIGHTS = [0.72, 0.28]

MOTIVOS_CHAMADO = [
    "PROBLEMA COM CONEXAO",
    "PROBLEMA NA BATERIA",
    "PROBLEMA NA TELA",
    "PROBLEMA COM TECLADO",
    "MAQUINA NAO LIGA",
    "PROBLEMA NO LEITOR",
    "OUTROS",
]


def _rng_setup(seed: int = SEED) -> Faker:
    np.random.seed(seed)
    random.seed(seed)
    fake = Faker("pt_BR")
    Faker.seed(seed)
    return fake


def _unique_terminal_ids(n: int) -> List[str]:
    """
    Gera IDs no formato observado nos exemplos: '########-#'
    (8 dígitos, hífen, 1 dígito), garantindo unicidade.
    """
    ids = set()
    while len(ids) < n:
        ids.add(f"{random.randint(10_000_000, 99_999_999)}-{random.randint(0, 9)}")
    return list(ids)


def _unique_cliente_ids(n: int) -> List[str]:
    """Gera IDs de cliente como string numérica de 16 dígitos (ex.: '6053320103439738')."""
    ids = set()
    while len(ids) < n:
        ids.add("".join(str(random.randint(0, 9)) for _ in range(16)))
    return list(ids)


def generate_tb_terminais(n_terminais: int, fake: Faker) -> pd.DataFrame:
    # clientes com múltiplos terminais (média ~2 terminais/cliente)
    n_clientes = max(1, int(round(n_terminais / 2)))
    clientes = _unique_cliente_ids(n_clientes)

    terminais = _unique_terminal_ids(n_terminais)

    # data_registro distribuída nos últimos ~5 anos, em formato DATE (YYYY-MM-DD)
    start = date.today() - timedelta(days=5 * 365)
    end = date.today() - timedelta(days=30)

    data = []
    for tid in terminais:
        cid = random.choice(clientes)
        segmento = random.choice(SEGMENTOS)
        setor = random.choice(SETORES)
        modelo = random.choice(MODELOS)
        dt_reg = fake.date_between(start_date=start, end_date=end)
        data.append([tid, cid, segmento, setor, modelo, dt_reg.isoformat()])

    df = pd.DataFrame(
        data,
        columns=[
            "id_terminal",
            "id_cliente",
            "segmento_cliente",
            "setor_cliente",
            "modelo",
            "data_registro",
        ],
    )
    return df


def _sample_terminal_rows(tb_terminais: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return tb_terminais.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)


def _valor_transacao_lognormal(n: int) -> np.ndarray:
    """
    Aproxima o perfil do exemplo:
      - mediana ~ 1k
      - cauda longa até ~100k
    Usamos lognormal e aplicamos truncamentos mínimos/máximos.
    """
    mu = 6.9   # mediana ~ exp(6.9) ≈ 992
    sigma = 3.0
    vals = np.random.lognormal(mean=mu, sigma=sigma, size=n)

    # truncamentos observados no exemplo
    vals = np.clip(vals, 10.0, 99_989.55)
    return np.round(vals, 2)


def generate_tb_transacoes(n_transacoes: int, tb_terminais: pd.DataFrame, fake: Faker) -> pd.DataFrame:
    sampled = _sample_terminal_rows(tb_terminais, n_transacoes, seed=SEED + 1)

    valores = _valor_transacao_lognormal(n_transacoes)
    status = random.choices(STATUS_TRANSACAO, weights=STATUS_WEIGHTS, k=n_transacoes)
    tipos = random.choices(TIPO_TRANSACAO, weights=TIPO_WEIGHTS, k=n_transacoes)

    datas = []
    for i in range(n_transacoes):
        dt_reg = date.fromisoformat(str(sampled.loc[i, "data_registro"]))
        dt_tx = fake.date_between(start_date=dt_reg, end_date=date.today())
        datas.append(dt_tx.isoformat())

    df = pd.DataFrame(
        {
            "id_terminal": sampled["id_terminal"].values,
            "id_cliente": sampled["id_cliente"].values,
            "data_transacao": datas,
            "status_transacao": status,
            "valor_transacao": valores,
            "tipo_transacao": tipos,
        }
    )

    # Ordem das colunas conforme o CSV exemplo
    df = df[
        [
            "id_terminal",
            "id_cliente",
            "data_transacao",
            "status_transacao",
            "valor_transacao",
            "tipo_transacao",
        ]
    ]
    return df


def generate_tb_chamados(n_chamados: int, tb_terminais: pd.DataFrame, fake: Faker) -> pd.DataFrame:
    sampled = _sample_terminal_rows(tb_terminais, n_chamados, seed=SEED + 2)

    tipos = random.choices(TIPO_CHAMADO, weights=TIPO_CHAMADO_WEIGHTS, k=n_chamados)
    motivos = random.choices(MOTIVOS_CHAMADO, k=n_chamados)

    aberturas = []
    fechamentos = []
    for i in range(n_chamados):
        dt_reg = date.fromisoformat(str(sampled.loc[i, "data_registro"]))
        abertura = fake.date_between(start_date=dt_reg, end_date=date.today())
        # tempo de resolução em dias (cauda curta + alguns casos longos)
        dias = int(max(1, np.random.lognormal(mean=1.2, sigma=0.7)))  # ~1 a ~10+ dias
        fechamento = abertura + timedelta(days=dias)
        if fechamento > date.today():
            fechamento = date.today()
        aberturas.append(abertura.isoformat())
        fechamentos.append(fechamento.isoformat())

    # id_chamado como inteiro de 9 dígitos (como no exemplo)
    id_chamado = [str(random.randint(100_000_000, 999_999_999)) for _ in range(n_chamados)]

    df = pd.DataFrame(
        {
            "id_chamado": id_chamado,
            "id_terminal": sampled["id_terminal"].values,
            "id_cliente": sampled["id_cliente"].values,
            "tipo_chamado": tipos,
            "motivo_chamado": motivos,
            "data_abertura_chamado": aberturas,
            "data_fechamento_chamado": fechamentos,
        }
    )

    # Ordem das colunas conforme o CSV exemplo
    df = df[
        [
            "id_chamado",
            "id_terminal",
            "id_cliente",
            "tipo_chamado",
            "motivo_chamado",
            "data_abertura_chamado",
            "data_fechamento_chamado",
        ]
    ]
    return df


def main() -> None:
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    fake = _rng_setup(SEED)

    tb_terminais = generate_tb_terminais(N_TERMINAIS, fake)
    tb_transacoes = generate_tb_transacoes(N_TRANSACOES, tb_terminais, fake)
    tb_chamados = generate_tb_chamados(N_CHAMADOS, tb_terminais, fake)

    tb_terminais.to_csv(OUT_TERMINAIS, index=False)
    tb_transacoes.to_csv(OUT_TRANSACOES, index=False)
    tb_chamados.to_csv(OUT_CHAMADOS, index=False)

    print("Dados sintéticos gerados e exportados com sucesso!")
    print("Linhas:")
    print(f" - tb_terminais:   {len(tb_terminais)}  -> {OUT_TERMINAIS}")
    print(f" - tb_transacoes:  {len(tb_transacoes)} -> {OUT_TRANSACOES}")
    print(f" - tb_chamados:    {len(tb_chamados)}   -> {OUT_CHAMADOS}")


if __name__ == "__main__":
    main()
