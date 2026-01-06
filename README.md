# Painel de monitoramento do Terminal Park

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Pandas](https://img.shields.io/badge/pandas-data%20analysis-green)
![Streamlit](https://img.shields.io/badge/streamlit-dashboard-red)
![Status](https://img.shields.io/badge/status-completed-success)

## Breve descrição
End-to-end Análise exploratória de dados e painel interativo para monitoramento do uso do terminal, confiabilidade operacional e priorização dentro do parque de terminais.

# Estrutura de diretórios

root/
│
├── data/
│   ├── dados_siteticos.py                  # Código que gera a base de dados sintética
│   ├── tb_parque_terminais.csv             # Data frame contendo o parque de terminais da empresa
│   ├── tb_chamados.csv                     # Data frame contendo os chamados técnicos    
│   └── tb_transacoes.csv                   # Data frame contendo as transações 
│
├── notebooks/
│   └── 01_eda.ipynb                        # Análise Exploratória de Dados (EDA)
│
├── src/
│   ├── __init__.py
│   └── 02_eda.py                           # Análise Exploratória de Dados (EDA)
│
├── dashboard/
│   ├── app.py               # Aplicação Streamlit
│   └── components/          # Componentes reutilizáveis do dashboard
│
├── reports/
│   └── figures/             # Gráficos exportados
│
├── requirements.txt         # Dependências do projeto
├── README.md                # Documentação principal (este arquivo)
└── .gitignore



# Desafio 2 — Monitoramento do Parque de Terminais

## 1. Visão Geral do Projeto

O objetivo central é transformar dados operacionais brutos em **insights acionáveis**, capazes de apoiar decisões estratégicas relacionadas a:
- Uso e disponibilidade dos terminais,
- Confiabilidade operacional,
- Priorização de ações corretivas,
- Observabilidade contínua do parque instalado.

A solução foi concebida seguindo princípios de **Data Analytics end-to-end**, integrando EDA, definição de métricas, visualização e narrativa de negócio.

---

## 2. Objetivos Analíticos

Os principais objetivos deste projeto são:

- Avaliar o **nível de utilização dos terminais** ao longo do tempo;
- Identificar **terminais inativos ou com baixa recorrência de transações**;
- Medir **indicadores de confiabilidade operacional**, como atrasos e indisponibilidades;
- Fornecer **subsídios analíticos para priorização operacional**, orientando ações de manutenção, substituição ou reconfiguração;
- Criar um **dashboard executivo e operacional**, com filtros globais e métricas claras.

---

## 3. Dados Utilizados

Os dados analisados contemplam, de forma integrada:

- **Parque de terminais**: identificação do terminal, modelo, segmento e setor do cliente;
- **Transações**: registros de transações com timestamp, terminal associado e volume;
- **Chamados técnicos**: datas de abertura e fechamento, permitindo cálculo de tempos de resolução.

Todos os dados passaram por etapas de:
- Padronização de tipos,
- Tratamento de datas,
- Validação de chaves,
- Remoção de inconsistências lógicas.

---

## 4. Análise Exploratória de Dados (EDA)

A EDA foi estruturada para responder perguntas de negócio concretas, indo além de estatísticas descritivas básicas.

### 4.1 Utilização dos Terminais

Foi identificada uma **distribuição altamente heterogênea de uso**, com:
- Um pequeno grupo de terminais concentrando grande parte das transações;
- Uma fração relevante do parque apresentando **baixa ou nenhuma atividade** em janelas prolongadas de tempo.

Esse padrão sugere oportunidades claras de:
- Otimização do parque instalado,
- Realocação de ativos,
- Revisão de contratos e estratégias comerciais.

---

### 4.2 Inatividade e Atrasos Operacionais

A análise temporal revelou:
- Terminais sem registro de transações por períodos superiores a 48 horas;
- Padrões recorrentes de inatividade associados a determinados modelos ou segmentos.

Esses achados indicam potenciais falhas de:
- Conectividade,
- Configuração,
- Aderência do terminal ao perfil do cliente.

---

### 4.3 Confiabilidade e Chamados Técnicos

A partir dos dados de chamados, foi calculado o **tempo de resolução em dias**, revelando que:
- A maioria dos chamados é resolvida em curto prazo;
- Existe uma cauda longa de casos com **resolução significativamente mais lenta**, impactando diretamente a disponibilidade dos terminais.

Esse comportamento reforça a importância de **monitoramento contínuo** e **priorizaçãoHook dinâmica de atendimento**.

---

## 5. Principais Insights Extraídos

Os principais insights consolidados da EDA são:

1. **Concentração de uso**: grande parte do volume transacional está concentrada em poucos terminais.
2. **Ociosidade estrutural**: há um percentual relevante do parque subutilizado.
3. **Risco operacional silencioso**: terminais inativos nem sempre geram chamados, mas impactam receita.
4. **Assimetria na resolução de chamados**: poucos casos críticos afetam significativamente a média.
5. **Potencial de priorização baseada em dados**: combinação de uso, inatividade e chamados permite rankeamento operacional.

---

## 6. Dashboard Interativo

O dashboard foi desenvolvido em **Streamlit**, estruturado em cinco abas principais:

- **Visão Geral**: KPIs globais do parque;
- **Uso**: métricas de transações e atividade;
- **Confiabilidade**: análise de chamados e tempos de resolução;
- **Priorização**: ranking de terminais críticos;
- **Observabilidade**: visão integrada para acompanhamento contínuo.

Filtros globais permitem segmentação por data, modelo, setor e segmento de cliente.

---

## 7. Tecnologias Utilizadas

- Python 3.x
- Pandas, NumPy
- Plotly
- Streamlit
- Jupyter Notebook

---

## 8. Próximos Passos

Como extensões naturais do projeto, destacam-se:
- Aplicação de modelos de **anomaly detection** para identificar comportamentos atípicos;
- Construção de **scores de risco operacional**;
- Integração com pipelines automatizados (ETL/ELT);
- Monitoramento em tempo real com alertas.

---

## 9. Autor

Projeto desenvolvido por **Dr. Flavio Rusch**, no contexto de um desafio técnico voltado à análise de dados e monitoramento operacional.

