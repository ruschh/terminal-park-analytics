# Terminal Park Monitoring Dashboard

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Pandas](https://img.shields.io/badge/pandas-data%20analysis-green)
![Streamlit](https://img.shields.io/badge/streamlit-dashboard-red)
![Status](https://img.shields.io/badge/status-completed-success)

## Short Description
End-to-end exploratory data analysis and interactive dashboard for monitoring terminal usage, operational reliability, and prioritization within the terminal park.

# Directories strutucters
```text
root/
│
├── data/
│   ├── dados_siteticos.py                  # Code that generates the synthetic database
│   ├── tb_parque_terminais.csv             # Data frame containing the company's terminal park
│   ├── tb_chamados.csv                     # Data frame containing the technical support calls
│   └── tb_transacoes.csv                   # Data frame containing the transactions 
│
├── notebooks/
│   └── 01_eda.ipynb                        # Exploratory Data Analisys (EDA)
│
├── src/
│   ├── __init__.py
│   └── 01_eda.py                           # Exploratory Data Analisys (EDA)
│
├── dashboard/
│   └── app.py               # Streamlit application
│
├── reports/
│   └── figures/             # Exported figures
│
├── requirements.txt         
├── README.md                
└── .gitignore
```

---

## 1. Project Overview

This project was developed as part of **Challenge 2** in a technical assessment context related to **:contentReference[oaicite:0]{index=0}**, focusing on data-driven monitoring of the installed terminal park.

The primary objective is to transform raw operational data into **actionable insights** that support strategic and operational decision-making, particularly in areas such as:
- Terminal usage and availability
- Operational reliability
- Incident resolution efficiency
- Data-driven prioritization of corrective actions
- Continuous observability through dashboards

The solution follows an **end-to-end analytics approach**, integrating Exploratory Data Analysis (EDA), KPI definition, and interactive visualization.

---

## 2. Analytical Objectives

The main analytical goals of this project are:

- Assess terminal usage patterns over time
- Identify inactive or underutilized terminals
- Measure operational reliability through incident and resolution metrics
- Support operational prioritization using data-driven rankings
- Deliver an executive-ready and operational dashboard with global filters

---

## 3. Data Sources

The analysis integrates multiple operational datasets, including:

- **Terminal Park Data**: terminal identifiers, models, client segments, and business sectors
- **Transaction Data**: transactional activity with timestamps and terminal associations
- **Technical Incidents**: opening and closing dates of service calls, enabling resolution-time analysis

All datasets undergo:
- Data type standardization
- Date parsing and validation
- Key consistency checks
- Logical integrity validation

---

## 4. Exploratory Data Analysis (EDA)

The EDA was designed to answer concrete business questions rather than purely descriptive statistics.

### 4.1 Terminal Usage Patterns

The analysis reveals a **highly heterogeneous usage distribution**, where:
- A relatively small subset of terminals concentrates a significant share of transaction volume
- A non-negligible portion of the terminal park shows **low or no activity** over extended periods

This pattern highlights clear opportunities for:
- Asset optimization
- Terminal reallocation
- Contract and commercial strategy review

---

### 4.2 Inactivity and Operational Delays

Temporal analysis identified:
- Terminals without recorded transactions for periods exceeding 48 hours
- Recurrent inactivity patterns associated with specific terminal models or client segments

These findings may indicate issues related to:
- Connectivity
- Configuration
- Mismatch between terminal profile and client needs

---

### 4.3 Operational Reliability and Incident Resolution

Using incident data, the **resolution time (in days)** was computed, revealing:
- Most incidents are resolved within short timeframes
- A long-tail distribution of cases with significantly delayed resolution

Although infrequent, these critical cases have a disproportionate impact on terminal availability and operational efficiency.

---

## 5. Key Insights

The main insights consolidated from the EDA are:

1. **Usage concentration**: Transaction volume is heavily concentrated in a limited subset of terminals
2. **Structural underutilization**: A relevant fraction of the terminal park is persistently underused
3. **Silent operational risk**: Inactive terminals do not always trigger incidents but may directly impact revenue
4. **Asymmetric incident resolution**: Few critical incidents dominate average resolution metrics
5. **Data-driven prioritization potential**: Combining usage, inactivity, and incident data enables effective operational ranking

---

## 6. Interactive Dashboard

The dashboard was developed using **Streamlit** and is structured into five main sections:

- **Overview**: Global KPIs of the terminal park
- **Usage**: Transactional activity and usage metrics
- **Reliability**: Incident analysis and resolution times
- **Prioritization**: Ranking of critical terminals
- **Observability**: Integrated monitoring view for continuous follow-up

Global filters allow segmentation by date, terminal model, client sector, and segment.

---

## 7. Technology Stack

- Python 3.9+
- Pandas
- NumPy
- Plotly
- Streamlit
- Jupyter Notebook

---

## 8. Future Enhancements

Potential next steps include:
- Anomaly detection models for early identification of abnormal behavior
- Operational risk scoring frameworks
- Integration with automated ETL/ELT pipelines
- Real-time monitoring and alerting mechanisms

---

## 9. Author

Developed by **Flavio Rusch** as part of a technical challenge focused on data analytics and operational monitoring.

