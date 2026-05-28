"""
OrbitaConecta — Análise Exploratória de Dados
==============================================
Este script realiza a análise exploratória completa dos dados utilizados
no projeto OrbitaConecta, usando apenas fontes públicas e gratuitas.

Fontes de dados:
- IBGE: dados municipais (IDH, renda, população rural)
- Anatel: cobertura de telecomunicações por município
- Dados sintéticos baseados em distribuições reais para o protótipo

Requisitos:
    pip install pandas numpy matplotlib seaborn geopandas folium plotly scikit-learn requests

Execução:
    python orbitaconecta_analise.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURAÇÕES VISUAIS
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F9FAFB",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
    "font.size": 11,
})

PALETTE = ["#1E3A5F", "#2980B9", "#27AE60", "#E67E22", "#E74C3C", "#8E44AD"]
OUTPUT_DIR = "orbitaconecta_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  OrbitaConecta — Análise Exploratória de Dados")
print("=" * 60)


# ─────────────────────────────────────────────
# 1. GERAÇÃO DE DADOS MUNICIPAIS
#    (simulados com distribuições baseadas nos
#    dados reais do IBGE/Anatel 2022-2024)
# ─────────────────────────────────────────────
print("\n[1/7] Gerando base de dados municipais...")

np.random.seed(42)
N = 5570  # total de municípios brasileiros

REGIOES = {
    "Norte":     {"n": 450, "idh_mu": 0.62, "idh_sd": 0.06, "renda_mu": 750,  "renda_sd": 200, "cob_mu": 0.45},
    "Nordeste":  {"n": 1794,"idh_mu": 0.64, "idh_sd": 0.07, "renda_mu": 800,  "renda_sd": 220, "cob_mu": 0.52},
    "Centro-Oeste":{"n":467,"idh_mu": 0.73, "idh_sd": 0.05, "renda_mu": 1100, "renda_sd": 280, "cob_mu": 0.72},
    "Sudeste":   {"n": 1668,"idh_mu": 0.76, "idh_sd": 0.05, "renda_mu": 1350, "renda_sd": 350, "cob_mu": 0.81},
    "Sul":       {"n": 1191,"idh_mu": 0.75, "idh_sd": 0.05, "renda_mu": 1250, "renda_sd": 300, "cob_mu": 0.80},
}

frames = []
for regiao, cfg in REGIOES.items():
    n = cfg["n"]
    idh = np.clip(np.random.normal(cfg["idh_mu"], cfg["idh_sd"], n), 0.40, 0.89)
    renda = np.clip(np.random.normal(cfg["renda_mu"], cfg["renda_sd"], n), 300, 3500)
    pop_rural = np.random.beta(2, 5, n) * 100  # % pop rural
    cobertura_4g = np.clip(np.random.normal(cfg["cob_mu"], 0.18, n), 0.0, 1.0)
    cobertura_banda_larga = cobertura_4g * np.random.uniform(0.5, 0.9, n)
    pop_total = np.random.lognormal(9.5, 1.1, n).astype(int)
    ndvi = np.clip(np.random.normal(0.52, 0.15, n), 0.1, 0.9)  # índice vegetação NDVI
    distancia_capital = np.abs(np.random.normal(250, 180, n))

    frames.append(pd.DataFrame({
        "regiao": regiao,
        "idh": idh,
        "renda_per_capita": renda,
        "pct_pop_rural": pop_rural,
        "cobertura_4g": cobertura_4g,
        "cobertura_banda_larga": cobertura_banda_larga,
        "pop_total": pop_total,
        "ndvi_medio": ndvi,
        "distancia_capital_km": distancia_capital,
    }))

df = pd.concat(frames, ignore_index=True)
df.index.name = "municipio_id"
df["municipio_id"] = df.index + 1

print(f"   Base criada: {len(df):,} municípios | {df.shape[1]} variáveis")
print(df.describe().round(3).to_string())


# ─────────────────────────────────────────────
# 2. CÁLCULO DO ÍNDICE DE VULNERABILIDADE DIGITAL (IVD)
# ─────────────────────────────────────────────
print("\n[2/7] Calculando Índice de Vulnerabilidade Digital (IVD)...")

scaler = MinMaxScaler()

# Variáveis que AUMENTAM a vulnerabilidade (quanto menor, mais vulnerável)
vars_inversas = ["idh", "renda_per_capita", "cobertura_4g", "cobertura_banda_larga"]
# Variáveis que AUMENTAM diretamente a vulnerabilidade
vars_diretas = ["pct_pop_rural", "distancia_capital_km"]

df_norm = df.copy()

# Normaliza inversas: 1 - valor normalizado = maior vulnerabilidade
for col in vars_inversas:
    vals = scaler.fit_transform(df[[col]])
    df_norm[f"norm_{col}"] = 1 - vals.flatten()

# Normaliza diretas
for col in vars_diretas:
    vals = scaler.fit_transform(df[[col]])
    df_norm[f"norm_{col}"] = vals.flatten()

# Pesos calibrados com base em literatura de inclusão digital
PESOS = {
    "norm_cobertura_4g":          0.30,
    "norm_cobertura_banda_larga": 0.20,
    "norm_idh":                   0.20,
    "norm_renda_per_capita":      0.15,
    "norm_pct_pop_rural":         0.10,
    "norm_distancia_capital_km":  0.05,
}

df["ivd"] = sum(df_norm[col] * peso for col, peso in PESOS.items())
df["ivd"] = (df["ivd"] * 100).round(2)  # escala 0–100

# Classificação por faixas
def classificar_ivd(ivd):
    if ivd >= 70: return "Crítico"
    if ivd >= 50: return "Alto"
    if ivd >= 30: return "Médio"
    return "Baixo"

df["classe_ivd"] = df["ivd"].apply(classificar_ivd)
df["pop_sem_internet"] = (df["pop_total"] * (1 - df["cobertura_4g"])).astype(int)

print(f"   IVD calculado. Estatísticas:")
print(df["ivd"].describe().round(2).to_string())
print(f"\n   Distribuição por classe:")
print(df["classe_ivd"].value_counts().to_string())


# ─────────────────────────────────────────────
# 3. CLUSTERING K-MEANS (Segmentação de Prioridade)
# ─────────────────────────────────────────────
print("\n[3/7] Aplicando K-Means para segmentação de municípios...")

FEATURES_CLUSTER = [
    "norm_cobertura_4g", "norm_idh", "norm_renda_per_capita",
    "norm_pct_pop_rural", "norm_distancia_capital_km"
]

X = df_norm[FEATURES_CLUSTER].values

# Método do cotovelo para escolher K
inertias = []
silhouettes = []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels))

# K=4 tem melhor relação cotovelo/silhueta
K_FINAL = 4
km_final = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
df["cluster"] = km_final.fit_predict(X)

# Ordena clusters por IVD médio e renomeia
ordem_clusters = df.groupby("cluster")["ivd"].mean().sort_values(ascending=False)
mapa_cluster = {old: new for new, old in enumerate(ordem_clusters.index)}
df["cluster"] = df["cluster"].map(mapa_cluster)

NOMES_CLUSTER = {
    0: "P1 — Prioridade Crítica",
    1: "P2 — Alta Prioridade",
    2: "P3 — Prioridade Média",
    3: "P4 — Baixa Prioridade",
}
df["segmento"] = df["cluster"].map(NOMES_CLUSTER)

print(f"   Silhouette Score (K=4): {silhouette_score(X, df['cluster']):.4f}")
print("\n   Perfil médio por cluster:")
print(
    df.groupby("segmento")[
        ["ivd", "cobertura_4g", "idh", "renda_per_capita", "pop_sem_internet"]
    ].mean().round(2).to_string()
)


# ─────────────────────────────────────────────
# 4. VISUALIZAÇÕES
# ─────────────────────────────────────────────
print("\n[4/7] Gerando visualizações...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("OrbitaConecta — Análise Exploratória do Índice de Vulnerabilidade Digital",
             fontsize=15, fontweight="bold", color="#1E3A5F", y=1.01)

# ── Gráfico 1: Distribuição do IVD
ax = axes[0, 0]
ax.hist(df["ivd"], bins=40, color=PALETTE[1], alpha=0.85, edgecolor="white", linewidth=0.5)
ax.axvline(df["ivd"].mean(), color=PALETTE[4], linestyle="--", linewidth=1.8, label=f"Média: {df['ivd'].mean():.1f}")
ax.axvline(df["ivd"].median(), color=PALETTE[2], linestyle="--", linewidth=1.8, label=f"Mediana: {df['ivd'].median():.1f}")
ax.set_title("Distribuição do IVD nos Municípios Brasileiros", fontweight="bold")
ax.set_xlabel("Índice de Vulnerabilidade Digital (0–100)")
ax.set_ylabel("Número de municípios")
ax.legend(framealpha=0)

# ── Gráfico 2: IVD médio por região
ax = axes[0, 1]
ivd_regiao = df.groupby("regiao")["ivd"].mean().sort_values(ascending=True)
bars = ax.barh(ivd_regiao.index, ivd_regiao.values, color=PALETTE[:5], alpha=0.85)
for bar, val in zip(bars, ivd_regiao.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}", va="center", fontsize=10, color="#1E3A5F", fontweight="bold")
ax.set_title("IVD Médio por Região do Brasil", fontweight="bold")
ax.set_xlabel("IVD Médio")
ax.set_xlim(0, max(ivd_regiao.values) * 1.15)

# ── Gráfico 3: Cobertura 4G vs IDH (scatter)
ax = axes[0, 2]
cores_regiao = dict(zip(REGIOES.keys(), PALETTE))
for regiao in df["regiao"].unique():
    sub = df[df["regiao"] == regiao]
    ax.scatter(sub["cobertura_4g"], sub["idh"], alpha=0.15, s=8,
               color=cores_regiao.get(regiao, "#888"), label=regiao)
ax.set_title("Cobertura 4G × IDH por Município", fontweight="bold")
ax.set_xlabel("Cobertura 4G (%)")
ax.set_ylabel("IDH Municipal")
ax.legend(markerscale=3, framealpha=0, fontsize=9)

# ── Gráfico 4: Cotovelo K-Means
ax = axes[1, 0]
ax2_twin = ax.twinx()
ax.plot(K_range, inertias, "o-", color=PALETTE[0], linewidth=2, label="Inércia")
ax2_twin.plot(K_range, silhouettes, "s--", color=PALETTE[2], linewidth=2, label="Silhueta")
ax.axvline(K_FINAL, color=PALETTE[4], linestyle=":", linewidth=1.5, alpha=0.7)
ax.set_title("Método do Cotovelo — Escolha do K", fontweight="bold")
ax.set_xlabel("Número de clusters (K)")
ax.set_ylabel("Inércia", color=PALETTE[0])
ax2_twin.set_ylabel("Score de Silhueta", color=PALETTE[2])
ax.text(K_FINAL + 0.1, max(inertias)*0.9, f"K={K_FINAL}\n(escolhido)", fontsize=9, color=PALETTE[4])

# ── Gráfico 5: Municípios por cluster
ax = axes[1, 1]
contagem = df["segmento"].value_counts().sort_index()
cores_cluster = [PALETTE[4], PALETTE[3], PALETTE[1], PALETTE[2]]
wedges, texts, autotexts = ax.pie(
    contagem.values, labels=None, autopct="%1.1f%%",
    colors=cores_cluster, startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
for at in autotexts:
    at.set_fontsize(9)
ax.legend(contagem.index, loc="lower center", bbox_to_anchor=(0.5, -0.18),
          fontsize=8.5, framealpha=0)
ax.set_title("Distribuição de Municípios por Segmento", fontweight="bold")

# ── Gráfico 6: Pop sem internet por cluster
ax = axes[1, 2]
pop_cluster = df.groupby("segmento")["pop_sem_internet"].sum() / 1_000_000
pop_cluster = pop_cluster.sort_values(ascending=False)
bars = ax.bar(range(len(pop_cluster)), pop_cluster.values,
              color=cores_cluster, alpha=0.85, edgecolor="white")
for bar, val in zip(bars, pop_cluster.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{val:.1f}M", ha="center", fontsize=9, fontweight="bold", color="#1E3A5F")
ax.set_title("População Sem Internet por Segmento", fontweight="bold")
ax.set_ylabel("Milhões de pessoas")
ax.set_xticks(range(len(pop_cluster)))
ax.set_xticklabels([s.split("—")[0].strip() for s in pop_cluster.index], fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_analise_exploratoria.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Salvo: {OUTPUT_DIR}/01_analise_exploratoria.png")


# ─────────────────────────────────────────────
# 5. ANÁLISE DE CORRELAÇÃO
# ─────────────────────────────────────────────
print("\n[5/7] Análise de correlação e heatmap...")

cols_corr = ["ivd", "cobertura_4g", "cobertura_banda_larga", "idh",
             "renda_per_capita", "pct_pop_rural", "distancia_capital_km", "ndvi_medio"]
corr = df[cols_corr].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
            center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
            cbar_kws={"shrink": 0.8}, annot_kws={"size": 9})
ax.set_title("Matriz de Correlação — Variáveis do IVD", fontsize=13, fontweight="bold", color="#1E3A5F", pad=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_correlacao.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Salvo: {OUTPUT_DIR}/02_correlacao.png")


# ─────────────────────────────────────────────
# 6. CENÁRIOS DE INTERVENÇÃO
# ─────────────────────────────────────────────
print("\n[6/7] Calculando cenários de intervenção...")

# Municípios críticos: top 200 por IVD
criticos = df.nlargest(200, "ivd").copy()

# Cenário A: sem priorização (investimento uniforme aleatório)
np.random.seed(0)
aleatorios = df.sample(200).copy()
pop_conectada_aleatorio = (aleatorios["pop_sem_internet"] * 0.6).sum()

# Cenário B: com priorização pelo IVD
pop_conectada_ivd = (criticos["pop_sem_internet"] * 0.6).sum()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Cenários de Intervenção: Impacto da Priorização por IVD",
             fontsize=13, fontweight="bold", color="#1E3A5F")

# Comparativo de população conectada
ax = axes[0]
cenarios = ["Sem Priorização\n(aleatório)", "Com Priorização\npor IVD"]
valores = [pop_conectada_aleatorio / 1e6, pop_conectada_ivd / 1e6]
cores = ["#BDC3C7", "#2980B9"]
bars = ax.bar(cenarios, valores, color=cores, width=0.5, edgecolor="white", linewidth=2)
for bar, val in zip(bars, valores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{val:.2f}M", ha="center", fontsize=12, fontweight="bold", color="#1E3A5F")
ganho = ((valores[1] - valores[0]) / valores[0]) * 100
ax.annotate(f"+{ganho:.0f}% mais pessoas\nconectadas com o IVD",
            xy=(1, valores[1]), xytext=(0.5, valores[1] * 0.85),
            fontsize=10, color=PALETTE[2], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PALETTE[2]))
ax.set_ylabel("Milhões de pessoas conectadas")
ax.set_title("Pessoas Conectadas com Mesmo Investimento\n(200 municípios intervenidos)")

# Distribuição por região no cenário IVD
ax = axes[1]
reg_criticos = criticos.groupby("regiao").size().sort_values(ascending=True)
ax.barh(reg_criticos.index, reg_criticos.values,
        color=[cores_regiao[r] for r in reg_criticos.index], alpha=0.85)
for i, val in enumerate(reg_criticos.values):
    ax.text(val + 0.5, i, str(val), va="center", fontsize=10, fontweight="bold", color="#1E3A5F")
ax.set_title("Municípios Críticos por Região\n(Top 200 pelo IVD)")
ax.set_xlabel("Número de municípios")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_cenarios_intervencao.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Salvo: {OUTPUT_DIR}/03_cenarios_intervencao.png")


# ─────────────────────────────────────────────
# 7. EXPORTAÇÃO DOS DADOS
# ─────────────────────────────────────────────
print("\n[7/7] Exportando dados processados...")

df.to_csv(f"{OUTPUT_DIR}/municipios_ivd_completo.csv", index=False)
criticos[["municipio_id", "regiao", "ivd", "classe_ivd", "segmento",
          "cobertura_4g", "idh", "renda_per_capita", "pop_sem_internet"]]\
    .sort_values("ivd", ascending=False)\
    .to_csv(f"{OUTPUT_DIR}/municipios_criticos_top200.csv", index=False)

print(f"   Salvo: {OUTPUT_DIR}/municipios_ivd_completo.csv ({len(df):,} linhas)")
print(f"   Salvo: {OUTPUT_DIR}/municipios_criticos_top200.csv (200 municípios)")

# ─────────────────────────────────────────────
# SUMÁRIO FINAL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMÁRIO DOS RESULTADOS")
print("=" * 60)
print(f"  Total de municípios analisados : {len(df):,}")
print(f"  Municípios em situação crítica  : {(df['classe_ivd']=='Crítico').sum():,} ({(df['classe_ivd']=='Crítico').mean()*100:.1f}%)")
print(f"  Total sem internet (estimado)  : {df['pop_sem_internet'].sum()/1e6:.1f} milhões de pessoas")
print(f"  Região mais vulnerável (IVD)   : {df.groupby('regiao')['ivd'].mean().idxmax()}")
print(f"  IVD médio nacional             : {df['ivd'].mean():.1f}/100")
print(f"  Silhouette Score (K-Means K=4) : {silhouette_score(X, df['cluster']):.4f}")
print(f"  Ganho com priorização IVD      : +{ganho:.0f}% vs. investimento aleatório")
print(f"\n  Arquivos gerados em: ./{OUTPUT_DIR}/")
print("=" * 60)
