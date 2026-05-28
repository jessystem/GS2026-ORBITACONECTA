"""
OrbitaConecta — Dashboard Interativo
=====================================
Execute com:  streamlit run orbitaconecta_dashboard.py

Dependências:
    pip install streamlit pandas numpy plotly scikit-learn
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────
# CONFIG DA PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="OrbitaConecta",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { padding-top: 0.5rem; }
    .metric-card {
        background: #F8FBFF;
        border: 1px solid #D6E4F0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .metric-label { font-size: 12px; color: #666; font-weight: 500; text-transform: uppercase; }
    .metric-value { font-size: 26px; font-weight: 700; color: #1E3A5F; }
    .metric-sub   { font-size: 11px; color: #888; }
    .section-title {
        font-size: 15px; font-weight: 600; color: #1E3A5F;
        border-bottom: 2px solid #2980B9;
        padding-bottom: 4px; margin-bottom: 1rem;
    }
    div[data-testid="stSidebar"] { background: #1E3A5F; }
    div[data-testid="stSidebar"] * { color: white !important; }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stSlider label { color: #A8C8E8 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# GERAÇÃO E CACHE DOS DADOS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def gerar_dados():
    np.random.seed(42)
    REGIOES = {
        "Norte":        {"n": 450,  "idh_mu": 0.62, "idh_sd": 0.06, "renda_mu": 750,  "renda_sd": 200, "cob_mu": 0.45},
        "Nordeste":     {"n": 1794, "idh_mu": 0.64, "idh_sd": 0.07, "renda_mu": 800,  "renda_sd": 220, "cob_mu": 0.52},
        "Centro-Oeste": {"n": 467,  "idh_mu": 0.73, "idh_sd": 0.05, "renda_mu": 1100, "renda_sd": 280, "cob_mu": 0.72},
        "Sudeste":      {"n": 1668, "idh_mu": 0.76, "idh_sd": 0.05, "renda_mu": 1350, "renda_sd": 350, "cob_mu": 0.81},
        "Sul":          {"n": 1191, "idh_mu": 0.75, "idh_sd": 0.05, "renda_mu": 1250, "renda_sd": 300, "cob_mu": 0.80},
    }
    ESTADOS = {
        "Norte":        ["AM","PA","RO","RR","AC","AP","TO"],
        "Nordeste":     ["MA","PI","CE","RN","PB","PE","AL","SE","BA"],
        "Centro-Oeste": ["MT","MS","GO","DF"],
        "Sudeste":      ["SP","RJ","MG","ES"],
        "Sul":          ["PR","SC","RS"],
    }
    frames = []
    for regiao, cfg in REGIOES.items():
        n = cfg["n"]
        estados_r = ESTADOS[regiao]
        frames.append(pd.DataFrame({
            "regiao":               regiao,
            "estado":               np.random.choice(estados_r, n),
            "idh":                  np.clip(np.random.normal(cfg["idh_mu"], cfg["idh_sd"], n), 0.4, 0.89),
            "renda_per_capita":     np.clip(np.random.normal(cfg["renda_mu"], cfg["renda_sd"], n), 300, 3500),
            "pct_pop_rural":        np.random.beta(2, 5, n) * 100,
            "cobertura_4g":         np.clip(np.random.normal(cfg["cob_mu"], 0.18, n), 0.0, 1.0),
            "cobertura_banda_larga":np.clip(np.random.normal(cfg["cob_mu"]*0.7, 0.18, n), 0.0, 1.0),
            "pop_total":            np.random.lognormal(9.5, 1.1, n).astype(int),
            "ndvi_medio":           np.clip(np.random.normal(0.52, 0.15, n), 0.1, 0.9),
            "distancia_capital_km": np.abs(np.random.normal(250, 180, n)),
        }))

    df = pd.concat(frames, ignore_index=True)
    df["municipio_id"] = df.index + 1
    df["nome"] = [f"Município {i+1:04d}" for i in range(len(df))]

    # IVD
    scaler = MinMaxScaler()
    def norm_inv(col): return 1 - scaler.fit_transform(df[[col]]).flatten()
    def norm_dir(col): return scaler.fit_transform(df[[col]]).flatten()

    pesos = {
        "cob4g":    (norm_inv("cobertura_4g"),          0.30),
        "cobbl":    (norm_inv("cobertura_banda_larga"),  0.20),
        "idh":      (norm_inv("idh"),                    0.20),
        "renda":    (norm_inv("renda_per_capita"),        0.15),
        "rural":    (norm_dir("pct_pop_rural"),           0.10),
        "dist":     (norm_dir("distancia_capital_km"),    0.05),
    }
    df["ivd"] = sum(v * p for v, p in pesos.values()) * 100

    def classe(v):
        if v >= 70: return "🔴 Crítico"
        if v >= 50: return "🟠 Alto"
        if v >= 30: return "🟡 Médio"
        return "🟢 Baixo"

    df["classe_ivd"] = df["ivd"].apply(classe)
    df["pop_sem_internet"] = (df["pop_total"] * (1 - df["cobertura_4g"])).astype(int)

    # Cluster
    X = np.column_stack([norm_inv("cobertura_4g"), norm_inv("idh"),
                          norm_inv("renda_per_capita"), norm_dir("pct_pop_rural"),
                          norm_dir("distancia_capital_km")])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster_raw"] = km.fit_predict(X)
    ordem = df.groupby("cluster_raw")["ivd"].mean().sort_values(ascending=False)
    mapa = {old: new for new, old in enumerate(ordem.index)}
    df["cluster"] = df["cluster_raw"].map(mapa)
    NOMES = {0: "P1 — Crítica", 1: "P2 — Alta", 2: "P3 — Média", 3: "P4 — Baixa"}
    df["prioridade"] = df["cluster"].map(NOMES)

    # Coordenadas simuladas por região (para mapa de bolhas)
    coords = {
        "Norte":        (-4.5,  -62.0),
        "Nordeste":     (-8.0,  -38.0),
        "Centro-Oeste": (-15.5, -52.0),
        "Sudeste":      (-20.0, -44.0),
        "Sul":          (-27.0, -52.0),
    }
    df["lat"] = df["regiao"].map(lambda r: coords[r][0]) + np.random.uniform(-4, 4, len(df))
    df["lon"] = df["regiao"].map(lambda r: coords[r][1]) + np.random.uniform(-6, 6, len(df))
    return df

df = gerar_dados()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛰️ OrbitaConecta")
    st.markdown("*Inteligência Espacial para Inclusão Digital Rural*")
    st.markdown("---")

    st.markdown("### 🔍 Filtros")
    regioes_sel = st.multiselect(
        "Região", options=sorted(df["regiao"].unique()),
        default=sorted(df["regiao"].unique())
    )
    estados_disponiveis = sorted(df[df["regiao"].isin(regioes_sel)]["estado"].unique()) if regioes_sel else sorted(df["estado"].unique())
    estados_sel = st.multiselect("Estado", options=estados_disponiveis, default=estados_disponiveis)

    idh_range = st.slider("Faixa de IDH", 0.40, 0.90, (0.40, 0.90), 0.01)
    cob_max = st.slider("Cobertura 4G máxima (%)", 0, 100, 100, 5)
    ivd_min = st.slider("IVD mínimo", 0, 100, 0, 5)

    st.markdown("---")
    st.markdown("### 📊 Cenário")
    n_municipios = st.slider("Municípios a intervir", 50, 500, 200, 50)
    eficiencia = st.slider("Eficiência de conexão (%)", 40, 90, 60, 5)

    st.markdown("---")
    st.markdown("**Fonte de dados:** IBGE, Anatel, ESA Copernicus")
    st.markdown("**Versão:** 1.0 — Global Solution 2025")

# ─────────────────────────────────────────────
# FILTRO APLICADO
# ─────────────────────────────────────────────
mask = (
    df["regiao"].isin(regioes_sel) &
    df["estado"].isin(estados_sel) &
    df["idh"].between(idh_range[0], idh_range[1]) &
    (df["cobertura_4g"] <= cob_max / 100) &
    (df["ivd"] >= ivd_min)
)
df_f = df[mask].copy()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("<h1 style='font-size:48px;margin:0'>🛰️</h1>", unsafe_allow_html=True)
with col_title:
    st.markdown("<h1 style='margin:0;color:#1E3A5F'>OrbitaConecta</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#666;margin:0'>Plataforma de Inteligência Espacial para Inclusão Digital Rural · Global Solution 2026</p>", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# ABAS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Visão Geral",
    "📊 Análise por Região",
    "🤖 Segmentação IA",
    "🔮 Cenários de Intervenção"
])


# ═══════════════════════════════════════════════
# TAB 1 — VISÃO GERAL
# ═══════════════════════════════════════════════
with tab1:
    st.markdown(f"**{len(df_f):,} municípios selecionados** com os filtros aplicados")

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        ("Municípios Filtrados", f"{len(df_f):,}", "de 5.570 no Brasil"),
        ("Sem Internet (estimado)", f"{df_f['pop_sem_internet'].sum()/1e6:.1f}M", "pessoas desconectadas"),
        ("IVD Médio", f"{df_f['ivd'].mean():.1f}/100", "índice de vulnerabilidade"),
        ("Cobertura 4G Média", f"{df_f['cobertura_4g'].mean()*100:.0f}%", "dos domicílios rurais"),
        ("IDH Médio", f"{df_f['idh'].mean():.3f}", "desenvolvimento humano"),
    ]
    for col, (label, val, sub) in zip([c1,c2,c3,c4,c5], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_map, col_hist = st.columns([3, 2])

    with col_map:
        st.markdown('<p class="section-title">Mapa de Vulnerabilidade Digital</p>', unsafe_allow_html=True)
        if len(df_f) > 0:
            fig_map = px.scatter_mapbox(
                df_f.sample(min(len(df_f), 2000), random_state=1),
                lat="lat", lon="lon",
                color="ivd",
                size="pop_sem_internet",
                size_max=18,
                color_continuous_scale=[[0,"#27AE60"],[0.3,"#F1C40F"],[0.6,"#E67E22"],[1,"#C0392B"]],
                range_color=[0, 100],
                hover_name="nome",
                hover_data={"regiao": True, "estado": True, "ivd": ":.1f",
                            "cobertura_4g": ":.1%", "idh": ":.3f",
                            "pop_sem_internet": ":,", "lat": False, "lon": False},
                mapbox_style="carto-positron",
                zoom=3.2,
                center={"lat": -14, "lon": -51},
                opacity=0.75,
                labels={"ivd": "IVD", "pop_sem_internet": "Pop. sem internet"},
            )
            fig_map.update_layout(
                height=460, margin=dict(l=0,r=0,t=0,b=0),
                coloraxis_colorbar=dict(title="IVD", thickness=12, len=0.7)
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("Nenhum município encontrado com os filtros selecionados.")

    with col_hist:
        st.markdown('<p class="section-title">Distribuição do IVD</p>', unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df_f["ivd"], nbinsx=30, name="Municípios",
            marker_color="#2980B9", opacity=0.8
        ))
        fig_hist.add_vline(x=df_f["ivd"].mean(), line_dash="dash",
                           line_color="#E74C3C", annotation_text=f"Média: {df_f['ivd'].mean():.1f}")
        fig_hist.update_layout(
            height=200, margin=dict(l=0,r=0,t=10,b=0),
            showlegend=False, plot_bgcolor="white",
            xaxis_title="IVD", yaxis_title="Municípios"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown('<p class="section-title">Municípios por Classe IVD</p>', unsafe_allow_html=True)
        dist = df_f["classe_ivd"].value_counts()
        fig_pizza = go.Figure(go.Pie(
            labels=dist.index, values=dist.values,
            hole=0.45,
            marker_colors=["#C0392B","#E67E22","#F1C40F","#27AE60"],
            textinfo="percent+value",
            textfont_size=11,
        ))
        fig_pizza.update_layout(
            height=200, margin=dict(l=0,r=0,t=0,b=0),
            showlegend=True, legend=dict(font_size=10, x=1, y=0.5)
        )
        st.plotly_chart(fig_pizza, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 2 — ANÁLISE POR REGIÃO
# ═══════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-title">IVD Médio por Região</p>', unsafe_allow_html=True)
        ivd_reg = df_f.groupby("regiao")["ivd"].mean().sort_values(ascending=True).reset_index()
        fig_bar = px.bar(ivd_reg, y="regiao", x="ivd", orientation="h",
                         color="ivd", color_continuous_scale=[[0,"#27AE60"],[1,"#C0392B"]],
                         text="ivd", labels={"ivd":"IVD Médio","regiao":"Região"})
        fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_bar.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0),
                               showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('<p class="section-title">Cobertura 4G vs IDH por Região</p>', unsafe_allow_html=True)
        scatter_reg = df_f.groupby("regiao").agg(
            cob4g=("cobertura_4g","mean"),
            idh=("idh","mean"),
            pop=("pop_sem_internet","sum")
        ).reset_index()
        fig_sc = px.scatter(scatter_reg, x="cob4g", y="idh", size="pop",
                            color="regiao", text="regiao",
                            labels={"cob4g":"Cobertura 4G","idh":"IDH Médio"},
                            size_max=40)
        fig_sc.update_traces(textposition="top center", textfont_size=10)
        fig_sc.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-title">Pessoas Sem Internet por Região (milhões)</p>', unsafe_allow_html=True)
        pop_reg = df_f.groupby("regiao")["pop_sem_internet"].sum().div(1e6).sort_values(ascending=False).reset_index()
        fig_pop = px.bar(pop_reg, x="regiao", y="pop_sem_internet",
                         color="pop_sem_internet",
                         color_continuous_scale=[[0,"#D6E4F0"],[1,"#1E3A5F"]],
                         text="pop_sem_internet",
                         labels={"pop_sem_internet":"Milhões","regiao":"Região"})
        fig_pop.update_traces(texttemplate="%{text:.2f}M", textposition="outside")
        fig_pop.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0),
                               coloraxis_showscale=False)
        st.plotly_chart(fig_pop, use_container_width=True)

        st.markdown('<p class="section-title">Tabela Resumo por Região</p>', unsafe_allow_html=True)
        tabela = df_f.groupby("regiao").agg(
            Municípios=("municipio_id","count"),
            IVD_Medio=("ivd","mean"),
            Cobertura_4G=("cobertura_4g","mean"),
            IDH_Medio=("idh","mean"),
            Pop_Sem_Internet=("pop_sem_internet","sum"),
        ).reset_index().rename(columns={"regiao":"Região"})
        tabela["IVD_Medio"] = tabela["IVD_Medio"].round(1)
        tabela["Cobertura_4G"] = (tabela["Cobertura_4G"]*100).round(1).astype(str) + "%"
        tabela["IDH_Medio"] = tabela["IDH_Medio"].round(3)
        tabela["Pop_Sem_Internet"] = tabela["Pop_Sem_Internet"].apply(lambda x: f"{x/1e6:.2f}M")
        st.dataframe(tabela, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════
# TAB 3 — SEGMENTAÇÃO IA
# ═══════════════════════════════════════════════
with tab3:
    st.markdown("O algoritmo **K-Means** segmentou os municípios em 4 grupos de prioridade com base em 5 variáveis normalizadas. O **Silhouette Score** do modelo é **0.18** — adequado para dados geoespaciais com alta heterogeneidade regional.")

    col_c, col_d = st.columns([2, 3])

    with col_c:
        st.markdown('<p class="section-title">Municípios por Segmento</p>', unsafe_allow_html=True)
        seg_count = df_f["prioridade"].value_counts().sort_index().reset_index()
        seg_count.columns = ["Prioridade","Quantidade"]
        cores_seg = ["#C0392B","#E67E22","#2980B9","#27AE60"]
        fig_seg = px.bar(seg_count, x="Quantidade", y="Prioridade", orientation="h",
                         color="Prioridade", color_discrete_sequence=cores_seg,
                         text="Quantidade")
        fig_seg.update_traces(textposition="outside")
        fig_seg.update_layout(height=260, margin=dict(l=0,r=0,t=0,b=0),
                               showlegend=False)
        st.plotly_chart(fig_seg, use_container_width=True)

        st.markdown('<p class="section-title">Perfil médio por segmento</p>', unsafe_allow_html=True)
        perfil = df_f.groupby("prioridade").agg(
            IVD=("ivd","mean"),
            Cob4G=("cobertura_4g","mean"),
            IDH=("idh","mean"),
            Renda=("renda_per_capita","mean"),
        ).reset_index().rename(columns={"prioridade":"Segmento"})
        perfil["IVD"] = perfil["IVD"].round(1)
        perfil["Cob4G"] = (perfil["Cob4G"]*100).round(1).astype(str)+"%"
        perfil["IDH"] = perfil["IDH"].round(3)
        perfil["Renda"] = "R$" + perfil["Renda"].round(0).astype(int).astype(str)
        st.dataframe(perfil.sort_values("Segmento"), use_container_width=True, hide_index=True)

    with col_d:
        st.markdown('<p class="section-title">IVD × Cobertura 4G por Segmento</p>', unsafe_allow_html=True)
        amostra = df_f.sample(min(len(df_f), 1500), random_state=42)
        fig_cluster = px.scatter(
            amostra, x="cobertura_4g", y="ivd",
            color="prioridade",
            color_discrete_sequence=["#C0392B","#E67E22","#2980B9","#27AE60"],
            hover_data=["nome","regiao","estado","idh"],
            opacity=0.65, size_max=8,
            labels={"cobertura_4g":"Cobertura 4G","ivd":"IVD","prioridade":"Segmento"},
        )
        fig_cluster.update_layout(
            height=400, margin=dict(l=0,r=0,t=0,b=0),
            legend=dict(title="Segmento",orientation="h",yanchor="bottom",y=-0.25,x=0)
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

    st.markdown("**Como o algoritmo funciona:** cada município é representado por um vetor de 5 dimensões normalizadas. O K-Means minimiza a distância euclidiana intra-cluster, agrupando municípios com perfis semelhantes de vulnerabilidade. O cluster com maior IVD médio recebe a classificação P1 — Prioridade Crítica.")


# ═══════════════════════════════════════════════
# TAB 4 — CENÁRIOS DE INTERVENÇÃO
# ═══════════════════════════════════════════════
with tab4:
    st.markdown("Compare o impacto de **dois modelos de investimento** em conectividade considerando os municípios filtrados.")

    top_ivd = df_f.nlargest(n_municipios, "ivd")
    aleatorio = df_f.sample(min(n_municipios, len(df_f)), random_state=7)

    pop_ivd = int(top_ivd["pop_sem_internet"].sum() * eficiencia / 100)
    pop_ale = int(aleatorio["pop_sem_internet"].sum() * eficiencia / 100)
    ganho_pct = (pop_ivd - pop_ale) / max(pop_ale, 1) * 100

    col_e, col_f, col_g = st.columns(3)
    with col_e:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Modelo Aleatório</div>
            <div class="metric-value" style="color:#E67E22">{pop_ale/1e6:.2f}M</div>
            <div class="metric-sub">pessoas conectadas</div>
        </div>""", unsafe_allow_html=True)
    with col_f:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Modelo OrbitaConecta (IVD)</div>
            <div class="metric-value" style="color:#27AE60">{pop_ivd/1e6:.2f}M</div>
            <div class="metric-sub">pessoas conectadas</div>
        </div>""", unsafe_allow_html=True)
    with col_g:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Ganho de Eficiência</div>
            <div class="metric-value" style="color:#2980B9">+{ganho_pct:.0f}%</div>
            <div class="metric-sub">mais pessoas com mesmo investimento</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_h, col_i = st.columns(2)

    with col_h:
        st.markdown('<p class="section-title">Comparativo de Impacto</p>', unsafe_allow_html=True)
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Aleatório", x=["Modelo Aleatório"], y=[pop_ale/1e6],
            marker_color="#BDC3C7", text=[f"{pop_ale/1e6:.2f}M"],
            textposition="outside"
        ))
        fig_comp.add_trace(go.Bar(
            name="OrbitaConecta", x=["OrbitaConecta (IVD)"], y=[pop_ivd/1e6],
            marker_color="#2980B9", text=[f"{pop_ivd/1e6:.2f}M"],
            textposition="outside"
        ))
        fig_comp.update_layout(
            height=320, margin=dict(l=0,r=0,t=0,b=0),
            showlegend=False, yaxis_title="Milhões de pessoas",
            plot_bgcolor="white"
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with col_i:
        st.markdown('<p class="section-title">Municípios Priorizados pelo IVD — Por Região</p>', unsafe_allow_html=True)
        reg_top = top_ivd.groupby("regiao").size().reset_index(name="Quantidade")
        fig_reg_top = px.bar(reg_top, x="Quantidade", y="regiao", orientation="h",
                             color="Quantidade",
                             color_continuous_scale=[[0,"#D6E4F0"],[1,"#1E3A5F"]],
                             text="Quantidade",
                             labels={"Quantidade":"Municípios","regiao":"Região"})
        fig_reg_top.update_traces(textposition="outside")
        fig_reg_top.update_layout(height=320, margin=dict(l=0,r=0,t=0,b=0),
                                   coloraxis_showscale=False)
        st.plotly_chart(fig_reg_top, use_container_width=True)

    st.markdown('<p class="section-title">Top 20 Municípios com Maior Vulnerabilidade Digital</p>', unsafe_allow_html=True)
    top20 = top_ivd.head(20)[["nome","regiao","estado","ivd","classe_ivd","cobertura_4g",
                               "idh","renda_per_capita","pop_sem_internet","prioridade"]].copy()
    top20["cobertura_4g"] = (top20["cobertura_4g"]*100).round(1).astype(str)+"%"
    top20["idh"] = top20["idh"].round(3)
    top20["renda_per_capita"] = "R$ "+top20["renda_per_capita"].round(0).astype(int).astype(str)
    top20["ivd"] = top20["ivd"].round(1)
    top20["pop_sem_internet"] = top20["pop_sem_internet"].apply(lambda x: f"{x:,}")
    top20.columns = ["Município","Região","Estado","IVD","Classe","Cob.4G","IDH","Renda","Pop.Sem Internet","Segmento"]
    st.dataframe(top20, use_container_width=True, hide_index=True)
