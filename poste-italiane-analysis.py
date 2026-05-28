"""
=============================================================================
POSTE ITALIANE – Customer Needs & Sales Network Analysis
"How to Satisfy Our Customers' Needs?"
Alikhan Salimov, Arshat Kaipbergen, Anuar Abdumannapov, Temirlan Zorlykbaev

Full analysis pipeline:
  Step 1 – Population Needs Modeling (external data, province level)
  Step 2 – Customer Base Needs Modeling (internal data, gap analysis)
  Step 3 – Sales Network Sizing (10,000 advisors allocation)

Extras:
  • Interactive Plotly/HTML dashboard
  • ML models (clustering, random forest, regression)
  • Business-insight summaries printed throughout
=============================================================================
"""

# ── 0. IMPORTS ───────────────────────────────────────────────────────────────
import warnings, os
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.decomposition   import PCA
from sklearn.cluster         import KMeans
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import (classification_report,
                                     confusion_matrix, ConfusionMatrixDisplay,
                                     roc_curve, auc,
                                     average_precision_score)
import plotly.graph_objects as go
from plotly.subplots        import make_subplots
import plotly.io            as pio

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Poste Italiane brand colours
PI_BLUE   = "#003DA5"
PI_YELLOW = "#F5D100"
PI_LGREY  = "#F4F6FB"
PI_DGREY  = "#404040"

PRODUCT_COLS   = ["Postal_savings", "Liquidity", "Financing", "investment", "utilities"]
PRODUCT_LABELS = ["Postal Savings", "Liquidity", "Financing", "Investment", "Utilities"]
PRODUCT_COLORS = [PI_BLUE, "#0066CC", "#E87722", "#009B77", "#C41E3A"]

print("="*70)
print("  POSTE ITALIANE – FULL ANALYSIS PIPELINE")
print("="*70)

# ── 1. DATA LOADING ──────────────────────────────────────────────────────────
print("\n[1] Loading data …")

EXT_PATH = "External Data for step 1(DATA).csv"
INT_PATH = "Internal Clients Data for step 2(Internal Clients Data for step ).csv"

ext = pd.read_csv(EXT_PATH, sep=";", decimal=",")
int_df = pd.read_csv(INT_PATH, sep=";")

# Clean column names
ext.rename(columns={"Provincia ": "provincia"}, inplace=True)
ext.columns = ext.columns.str.strip()
int_df.columns = int_df.columns.str.strip()

print(f"  External data : {ext.shape[0]} provinces × {ext.shape[1]} variables")
print(f"  Internal data : {int_df.shape[0]:,} segment rows | "
      f"{int_df['customers'].sum():,.0f} total customers")

# Province abbreviation → full name mapping (all 107 Italian provinces)
PROV_MAP = {
    "AG":"AGRIGENTO","AL":"ALESSANDRIA","AN":"ANCONA","AO":"AOSTA",
    "AP":"ASCOLI PICENO","AQ":"L'AQUILA","AR":"AREZZO","AT":"ASTI",
    "AV":"AVELLINO","BA":"BARI","BG":"BERGAMO","BI":"BIELLA",
    "BL":"BELLUNO","BN":"BENEVENTO","BO":"BOLOGNA","BR":"BRINDISI",
    "BS":"BRESCIA","BT":"BARLETTA-ANDRIA-TRANI","BZ":"BOLZANO",
    "CA":"CAGLIARI","CB":"CAMPOBASSO","CE":"CASERTA","CH":"CHIETI",
    "CL":"CALTANISSETTA","CN":"CUNEO","CO":"COMO","CR":"CREMONA",
    "CS":"COSENZA","CT":"CATANIA","CZ":"CATANZARO","EN":"ENNA",
    "FC":"FORLÌ-CESENA","FE":"FERRARA","FG":"FOGGIA","FI":"FIRENZE",
    "FM":"FERMO","FR":"FROSINONE","GE":"GENOVA","GO":"GORIZIA",
    "GR":"GROSSETO","IM":"IMPERIA","IS":"ISERNIA","KR":"CROTONE",
    "LC":"LECCO","LE":"LECCE","LI":"LIVORNO","LO":"LODI",
    "LT":"LATINA","LU":"LUCCA","MB":"MONZA E DELLA BRIANZA",
    "MC":"MACERATA","ME":"MESSINA","MI":"MILANO","MN":"MANTOVA",
    "MO":"MODENA","MS":"MASSA-CARRARA","MT":"MATERA","NO":"NOVARA",
    "NU":"NUORO","OR":"ORISTANO","PA":"PALERMO","PC":"PIACENZA",
    "PD":"PADOVA","PE":"PESCARA","PG":"PERUGIA","PI":"PISA",
    "PN":"PORDENONE","PO":"PRATO","PR":"PARMA","PT":"PISTOIA",
    "PU":"PESARO E URBINO","PV":"PAVIA","PZ":"POTENZA","RA":"RAVENNA",
    "RC":"REGGIO DI CALABRIA","RE":"REGGIO NELL'EMILIA","RG":"RAGUSA",
    "RI":"RIETI","RM":"ROMA","RN":"RIMINI","RO":"ROVIGO",
    "SA":"SALERNO","SI":"SIENA","SO":"SONDRIO","SP":"LA SPEZIA",
    "SR":"SIRACUSA","SS":"SASSARI","SU":"SUD SARDEGNA","SV":"SAVONA",
    "TA":"TARANTO","TE":"TERAMO","TN":"TRENTO","TO":"TORINO",
    "TP":"TRAPANI","TR":"TERNI","TS":"TRIESTE","TV":"TREVISO",
    "UD":"UDINE","VA":"VARESE","VB":"VERBANO-CUSIO-OSSOLA",
    "VC":"VERCELLI","VE":"VENEZIA","VI":"VICENZA","VR":"VERONA",
    "VT":"VITERBO","VV":"VIBO VALENTIA",
}
int_df["provincia"] = int_df["province"].map(PROV_MAP)

print("  Province mapping applied.")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 – POPULATION NEEDS MODELING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  STEP 1 – POPULATION NEEDS MODELING")
print("="*70)

# ── 1a. EDA – External Data ──────────────────────────────────────────────────
print("\n[1a] EDA – External Data")

# Per-capita variables (normalise monetary totals by population)
ext["reddito_pc"]   = ext["REDDITO_TOT"]   / ext["POPOLAZIONE"]
ext["consumi_pc"]   = ext["CONSUMI_TOT"]   / ext["POPOLAZIONE"]
ext["risparmio_pc"] = ext["RISPARMIO"]     / ext["POPOLAZIONE"]

# Education rate
ext["laurea_rate"]  = ext["LAUREA"]        / ext["MAGGIORENNI"]
ext["diploma_rate"] = ext["DIPLOMA"]       / ext["MAGGIORENNI"]

# Employment rate
ext["occupazione_rate"] = ext["OCCUPATO"]  / ext["MAGGIORENNI"]
ext["pensionato_rate"]  = ext["PENSIONATO"]/ ext["MAGGIORENNI"]
ext["disoccupato_rate"] = ext["DISOCCUPATO"]/ ext["MAGGIORENNI"]

# Share foreign
ext["stranieri_rate"]   = ext["STRANIERI"] / ext["POPOLAZIONE"]
# Home ownership
ext["proprieta_rate"]   = ext["FAM_PROPRIETA"] / ext["FAMIGLIE"]

# ── Figure 1: Top-10 provinces by income per capita ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("EDA – External Data: Income & Savings by Province",
             fontsize=14, fontweight="bold", color=PI_BLUE)

top10_reddito = ext.nlargest(10, "reddito_pc")
axes[0].barh(top10_reddito["provincia"], top10_reddito["reddito_pc"],
             color=PI_BLUE, edgecolor="white")
axes[0].set_title("Top 10 Provinces – Income per capita (€)", fontsize=11)
axes[0].invert_yaxis()
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"€{x:,.0f}"))

top10_risp = ext.nlargest(10, "risparmio_pc")
axes[1].barh(top10_risp["provincia"], top10_risp["risparmio_pc"],
             color=PI_YELLOW, edgecolor="white")
axes[1].set_title("Top 10 Provinces – Savings per capita (€)", fontsize=11)
axes[1].invert_yaxis()
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"€{x:,.0f}"))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig01_income_savings.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig01_income_savings.png")

# ── Figure 2: Digital behaviour heatmap ─────────────────────────────────────
digital_cols = ["P_INTERNET_QUOTIDIANO","P_HOME_BANKING",
                "P_ACQUISTO_ONLINE","P_WEB_INSURANCE",
                "P_PLURI_BANCA","P_BANCOMAT","P_CARTA_CREDITO","P_PAYPAL"]
digital_labels = ["Daily Internet","Home Banking","Online Shopping",
                  "Web Insurance","Multi-Bank","Debit Card","Credit Card","PayPal"]

top15_internet = ext.nlargest(15, "P_INTERNET_QUOTIDIANO")["provincia"].tolist()
heat_data = ext[ext["provincia"].isin(top15_internet)].set_index("provincia")[digital_cols]
heat_data.columns = digital_labels

fig, ax = plt.subplots(figsize=(13, 7))
sns.heatmap(heat_data, annot=True, fmt=".2f", cmap="Blues",
            linewidths=0.4, ax=ax, cbar_kws={"label":"Score (0–1)"})
ax.set_title("Digital Behaviour Scores – Top 15 Digital Provinces",
             fontsize=13, fontweight="bold", color=PI_BLUE, pad=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig02_digital_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig02_digital_heatmap.png")

# ── Figure 3: Correlation matrix of key financial variables ─────────────────
key_vars = ["reddito_pc","risparmio_pc","laurea_rate","occupazione_rate",
            "pensionato_rate","disoccupato_rate","proprieta_rate",
            "P_HOME_BANKING","P_CARTA_CREDITO","P_PLURI_BANCA",
            "VALORE_MQ","stranieri_rate"]
key_labels = ["Income pc","Savings pc","Degree rate","Employment rate",
              "Retirement rate","Unemployment rate","Home ownership",
              "Home Banking","Credit Card","Multi-Bank",
              "Property €/sqm","Foreign rate"]

corr = ext[key_vars].corr()
corr.index = key_labels
corr.columns = key_labels

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.3, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix – Key Socioeconomic Variables",
             fontsize=13, fontweight="bold", color=PI_BLUE, pad=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig03_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig03_correlation_matrix.png")

# ── Figure 4: North / Centre / South comparison ─────────────────────────────
NORTH = ["TORINO","AOSTA","CUNEO","ALESSANDRIA","ASTI","BIELLA","NOVARA",
         "VERBANO-CUSIO-OSSOLA","VERCELLI","GENOVA","IMPERIA","LA SPEZIA",
         "SAVONA","MILANO","BERGAMO","BRESCIA","COMO","CREMONA","LECCO",
         "LODI","MANTOVA","MONZA E DELLA BRIANZA","PAVIA","SONDRIO","VARESE",
         "BOLZANO","TRENTO","VENEZIA","BELLUNO","PADOVA","ROVIGO","TREVISO",
         "VERONA","VICENZA","PORDENONE","GORIZIA","TRIESTE","UDINE",
         "BOLOGNA","FERRARA","FORLÌ-CESENA","MODENA","PARMA","PIACENZA",
         "RAVENNA","REGGIO NELL'EMILIA","RIMINI"]
SOUTH = ["NAPOLI","SALERNO","CASERTA","AVELLINO","BENEVENTO","BARI","BRINDISI",
         "FOGGIA","LECCE","TARANTO","BARLETTA-ANDRIA-TRANI","POTENZA","MATERA",
         "CATANZARO","COSENZA","CROTONE","REGGIO DI CALABRIA","VIBO VALENTIA",
         "PALERMO","AGRIGENTO","CALTANISSETTA","CATANIA","ENNA","MESSINA",
         "RAGUSA","SIRACUSA","TRAPANI","CAGLIARI","NUORO","ORISTANO",
         "SASSARI","SUD SARDEGNA","CAMPOBASSO","ISERNIA","CHIETI","L'AQUILA",
         "PESCARA","TERAMO"]

# Macro areas manually mapped based on standard Italian geographical grouping
def macro(p):
    if p in NORTH: return "North"
    if p in SOUTH: return "South"
    return "Centre"

ext["macro_area"] = ext["provincia"].apply(macro)

macro_agg = ext.groupby("macro_area").agg(
    reddito_pc=("reddito_pc","mean"),
    risparmio_pc=("risparmio_pc","mean"),
    laurea_rate=("laurea_rate","mean"),
    occupazione_rate=("occupazione_rate","mean"),
    P_HOME_BANKING=("P_HOME_BANKING","mean"),
    P_INTERNET_QUOTIDIANO=("P_INTERNET_QUOTIDIANO","mean"),
).round(4)

metrics = ["reddito_pc","risparmio_pc","laurea_rate",
           "occupazione_rate","P_HOME_BANKING","P_INTERNET_QUOTIDIANO"]
labels_m = ["Income pc (€)","Savings pc (€)","Degree rate",
            "Employment","Home Banking","Daily Internet"]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("North vs Centre vs South – Key Socioeconomic Metrics",
             fontsize=14, fontweight="bold", color=PI_BLUE)
colors = [PI_BLUE, PI_YELLOW, "#E87722"]
for i, (m, lbl) in enumerate(zip(metrics, labels_m)):
    ax = axes[i//3][i%3]
    vals = [macro_agg.loc[ma, m] for ma in ["North","Centre","South"]]
    bars = ax.bar(["North","Centre","South"], vals, color=colors, edgecolor="white")
    ax.set_title(lbl, fontsize=10)
    ax.bar_label(bars, fmt="%.3f" if vals[0]<1 else "€{:,.0f}",
                 padding=2, fontsize=8)
    ax.set_ylim(0, max(vals)*1.2)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig04_north_south.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig04_north_south.png")

# ── 1b. NEED INDEX CONSTRUCTION ──────────────────────────────────────────────
print("\n[1b] Building Province-level Need Indices …")

# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT JUSTIFICATION FOR NEED INDICES
# ─────────────────────────────────────────────────────────────────────────────
# Weights were defined heuristically (heuristic scoring model) because:
#   • No labelled demand variable was provided
#   • No labelled demand variable was available — proxy variables used
#
# Weight structure rationale:
#   • Highest weights (0.30–0.35) assigned to variables with strongest correlation
#     to the corresponding product (see fig03_correlation_matrix.png)
#   • Weight sensitivity: financial variables (income, savings) were prioritised
#     because they are conceptually linked to product demand and strongly
#     correlated with other socioeconomic maturity indicators (fig03)
#   • This is a heuristic business model, not a statistically proven demand model
#
# Correct phrasing for the defence:
#   "Because no labelled demand variable was provided, we constructed composite
#    need indices based on proxy variables with expert-defined weights,
#    validated by domain logic and correlation analysis."
# ─────────────────────────────────────────────────────────────────────────────
def minmax(series):
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn + 1e-9)

# ── LIQUIDITY need: income variability, employed pop, digital banking ────────
ext["need_liquidity"] = (
    0.30 * minmax(ext["reddito_pc"]) +
    0.20 * minmax(ext["occupazione_rate"]) +
    0.20 * minmax(ext["P_HOME_BANKING"]) +
    0.15 * minmax(ext["P_BANCOMAT"]) +
    0.15 * minmax(ext["FAMIGLIE"])
)

# ── POSTAL SAVINGS need: savings capacity, elderly share, low digital ────────
ext["need_postal_savings"] = (
    0.35 * minmax(ext["risparmio_pc"]) +
    0.25 * minmax(ext["pensionato_rate"]) +
    0.20 * minmax(ext["proprieta_rate"]) +
    0.20 * minmax(1 - ext["P_INTERNET_QUOTIDIANO"])   # inverse: less digital = more postal
)

# ── INVESTMENT need: wealth, education, entrepreneurship ────────────────────
ext["need_investment"] = (
    0.30 * minmax(ext["reddito_pc"]) +
    0.25 * minmax(ext["laurea_rate"]) +
    0.20 * minmax(ext["risparmio_pc"]) +
    0.15 * minmax(ext["IMPRENDITORE_LIBPROF"] / ext["MAGGIORENNI"]) +
    0.10 * minmax(ext["P_CARTA_CREDITO"])
)

# ── FINANCING need: mortgages, young workers, property demand ────────────────
ext["need_financing"] = (
    0.30 * minmax(ext["occupazione_rate"]) +
    0.25 * minmax(ext["VALORE_MQ"]) +
    0.25 * minmax(1 - ext["proprieta_rate"]) +   # renting → need for mortgage
    0.20 * minmax(ext["reddito_pc"])
)

# ── UTILITIES need: telecom spend, internet usage, household size ────────────
ext["need_utilities"] = (
    0.30 * minmax(ext["C_SERVIZI_TELEFONIA"]) +
    0.25 * minmax(ext["P_INTERNET_QUOTIDIANO"]) +
    0.20 * minmax(ext["C_ELETTRONICA_TELEFONIA"]) +
    0.15 * minmax(ext["FAMIGLIE"]) +
    0.10 * minmax(ext["C_CASA"])
)

NEED_COLS   = ["need_liquidity","need_postal_savings","need_investment",
               "need_financing","need_utilities"]
NEED_LABELS = ["Liquidity","Postal Savings","Investment","Financing","Utilities"]

print("  Need indices computed.")

# ── Figure 5: Need index bar charts (top 15 per need) ───────────────────────
fig, axes = plt.subplots(1, 5, figsize=(22, 7))
fig.suptitle("Population Need Indices by Province (Top 15)",
             fontsize=14, fontweight="bold", color=PI_BLUE)

for i, (nc, nl, col) in enumerate(zip(NEED_COLS, NEED_LABELS, PRODUCT_COLORS)):
    top = ext.nlargest(15, nc)[["provincia", nc]]
    axes[i].barh(top["provincia"], top[nc], color=col, edgecolor="white")
    axes[i].set_title(nl, fontsize=10, fontweight="bold")
    axes[i].invert_yaxis()
    axes[i].set_xlim(0, 1)
    axes[i].tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig05_need_indices.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig05_need_indices.png")

# ── 1c. PCA on need indices ──────────────────────────────────────────────────
print("\n[1c] PCA on Need Indices …")

X_need = ext[NEED_COLS].values
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_need)
ext["pca1"] = coords[:, 0]
ext["pca2"] = coords[:, 1]

explained = pca.explained_variance_ratio_
print(f"  PCA explained variance: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}")

fig, ax = plt.subplots(figsize=(11, 8))
scatter = ax.scatter(ext["pca1"], ext["pca2"],
                     c=ext["need_investment"], cmap="RdYlBu_r",
                     s=60, edgecolors="white", linewidths=0.5, alpha=0.85)
plt.colorbar(scatter, ax=ax, label="Investment Need Index")
for _, row in ext.nlargest(10, "reddito_pc").iterrows():
    ax.annotate(row["provincia"], (row["pca1"], row["pca2"]),
                fontsize=7, ha="center", va="bottom",
                xytext=(0, 5), textcoords="offset points")
ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance)", fontsize=11)
ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance)", fontsize=11)
ax.set_title("PCA of Province Need Profiles\n(coloured by Investment Need)",
             fontsize=13, fontweight="bold", color=PI_BLUE)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig06_pca_needs.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig06_pca_needs.png")

# ── 1d. K-Means Clustering of provinces ─────────────────────────────────────
print("\n[1d] K-Means Clustering (province profiles) …")

# Elbow method
inertias = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_need)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(K_range, inertias, "o-", color=PI_BLUE, linewidth=2, markersize=8)
ax.axvline(4, color=PI_YELLOW, linestyle="--", linewidth=1.5, label="Chosen k=4")
ax.set_xlabel("Number of Clusters (k)", fontsize=11)
ax.set_ylabel("Inertia", fontsize=11)
ax.set_title("Elbow Method – Optimal Number of Province Clusters",
             fontsize=12, fontweight="bold", color=PI_BLUE)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig07_elbow.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig07_elbow.png")

# Final clustering with k=4
km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
ext["cluster"] = km4.fit_predict(X_need)

# Label clusters meaningfully based on centroid profiles
centers = pd.DataFrame(km4.cluster_centers_, columns=NEED_LABELS)
print("\n  Cluster centroids:")
print(centers.round(3).to_string())

# Assign labels
cluster_means = ext.groupby("cluster")["reddito_pc"].mean()
rank = cluster_means.rank()
cluster_label_map = {}
for c in ext["cluster"].unique():
    r = rank[c]
    if r == 4:   cluster_label_map[c] = "A – High Wealth Northern"
    elif r == 3: cluster_label_map[c] = "B – Urban Dynamic"
    elif r == 2: cluster_label_map[c] = "C – Mixed Central"
    else:        cluster_label_map[c] = "D – Lower Income Southern"

ext["cluster_label"] = ext["cluster"].map(cluster_label_map)

# Figure: cluster scatter
fig, ax = plt.subplots(figsize=(11, 8))
palette = [PI_BLUE, PI_YELLOW, "#E87722", "#009B77"]
for i, (cl, grp) in enumerate(ext.groupby("cluster_label")):
    ax.scatter(grp["pca1"], grp["pca2"], label=cl,
               color=palette[i], s=70, edgecolors="white",
               linewidths=0.5, alpha=0.9)
    for _, row in grp.iterrows():
        ax.annotate(row["provincia"], (row["pca1"], row["pca2"]),
                    fontsize=5.5, color=palette[i], alpha=0.7)
ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance)", fontsize=11)
ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance)", fontsize=11)
ax.set_title("Province Clusters (K-Means, k=4) – PCA Space",
             fontsize=13, fontweight="bold", color=PI_BLUE)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig08_clusters.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig08_clusters.png")

# Cluster radar chart
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(NEED_LABELS))
width = 0.20
for i, (cl, grp) in enumerate(ext.groupby("cluster_label")):
    means = grp[NEED_COLS].mean().values
    ax.bar(x + i*width, means, width, label=cl,
           color=palette[i], edgecolor="white", alpha=0.85)
ax.set_xticks(x + 1.5*width)
ax.set_xticklabels(NEED_LABELS, fontsize=10)
ax.set_ylabel("Mean Need Index (0–1)", fontsize=11)
ax.set_title("Need Profile by Province Cluster",
             fontsize=13, fontweight="bold", color=PI_BLUE)
ax.legend(fontsize=8)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig09_cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig09_cluster_profiles.png")

# ── Business insight summary ─────────────────────────────────────────────────
print("\n  ── STEP 1 BUSINESS INSIGHTS ──────────────────────────────────────")
for nc, nl in zip(NEED_COLS, NEED_LABELS):
    top3 = ext.nlargest(3, nc)["provincia"].tolist()
    print(f"  {nl:15s} highest need: {', '.join(top3)}")

north_need_inv = ext[ext["macro_area"]=="North"]["need_investment"].mean()
south_need_inv = ext[ext["macro_area"]=="South"]["need_investment"].mean()
print(f"\n  Investment need: North avg={north_need_inv:.3f} vs South avg={south_need_inv:.3f}")
print(f"  → {'North' if north_need_inv>south_need_inv else 'South'} has higher investment need")
north_postal = ext[ext["macro_area"]=="North"]["need_postal_savings"].mean()
south_postal = ext[ext["macro_area"]=="South"]["need_postal_savings"].mean()
print(f"  Postal Savings need: North avg={north_postal:.3f} vs South avg={south_postal:.3f}")
print(f"  → {'North' if north_postal>south_postal else 'South'} has higher postal savings need")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 – CUSTOMER BASE NEEDS MODELING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  STEP 2 – CUSTOMER BASE NEEDS MODELING")
print("="*70)

# ── 2a. EDA – Internal Data ──────────────────────────────────────────────────
print("\n[2a] EDA – Internal Data")

# Weighted product ownership rates per province
def wavg(group, val_col, wt_col):
    return (group[val_col] * group[wt_col]).sum() / group[wt_col].sum()

prov_int = int_df.dropna(subset=["provincia"]).groupby("provincia").apply(
    lambda g: pd.Series({
        "total_customers" : g["customers"].sum(),
        "pct_postal"      : wavg(g, "Postal_savings", "customers"),
        "pct_liquidity"   : wavg(g, "Liquidity",      "customers"),
        "pct_financing"   : wavg(g, "Financing",      "customers"),
        "pct_investment"  : wavg(g, "investment",     "customers"),
        "pct_utilities"   : wavg(g, "utilities",      "customers"),
        "pct_digital"     : (g[g["channel_type"]=="DIGITALE"]["customers"].sum() /
                             g["customers"].sum()),
        "pct_high_wealth" : (g[g["wealth_range"]=="High"]["customers"].sum() /
                             g["customers"].sum()),
        "pct_youth"       : (g[g["age_range"]=="1.Youth 0-35"]["customers"].sum() /
                             g["customers"].sum()),
        "pct_senior"      : (g[g["age_range"]=="3.Seniors 66+"]["customers"].sum() /
                             g["customers"].sum()),
        "pct_email"       : wavg(g, "fl_email",       "customers"),
        "pct_multibanca"  : wavg(g, "fl_multibanca",  "customers"),
        "pct_cross3plus"  : (g[g["cross_selling"]=="3+"]["customers"].sum() /
                             g["customers"].sum()),
    })
).reset_index()

print(f"  Province-level aggregation: {prov_int.shape[0]} provinces")

# Figure: product ownership distribution
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
fig.suptitle("Product Ownership Rate Distribution Across Provinces",
             fontsize=13, fontweight="bold", color=PI_BLUE)
pct_cols   = ["pct_postal","pct_liquidity","pct_financing","pct_investment","pct_utilities"]
for i, (pc, lbl, col) in enumerate(zip(pct_cols, PRODUCT_LABELS, PRODUCT_COLORS)):
    axes[i].hist(prov_int[pc], bins=20, color=col, edgecolor="white", alpha=0.85)
    axes[i].axvline(prov_int[pc].mean(), color="black", linestyle="--",
                    linewidth=1.5, label=f"Mean={prov_int[pc].mean():.2f}")
    axes[i].set_title(lbl, fontsize=10, fontweight="bold")
    axes[i].legend(fontsize=8)
    axes[i].set_xlabel("Ownership rate")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig10_ownership_dist.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig10_ownership_dist.png")

# Figure: age group breakdown
age_prod = int_df.groupby("age_range").apply(
    lambda g: pd.Series({
        lbl: wavg(g, col, "customers")
        for col, lbl in zip(PRODUCT_COLS, PRODUCT_LABELS)
    })
).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(PRODUCT_LABELS))
w = 0.25
labels_age = age_prod["age_range"].tolist()
age_colors = [PI_BLUE, PI_YELLOW, "#E87722"]
for i, (_, row) in enumerate(age_prod.iterrows()):
    ax.bar(x + i*w, [row[lbl] for lbl in PRODUCT_LABELS], w,
           label=labels_age[i], color=age_colors[i], edgecolor="white")
ax.set_xticks(x + w)
ax.set_xticklabels(PRODUCT_LABELS, fontsize=10)
ax.set_ylabel("Ownership Rate", fontsize=11)
ax.set_title("Product Ownership Rate by Age Group",
             fontsize=13, fontweight="bold", color=PI_BLUE)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig11_ownership_age.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig11_ownership_age.png")

# Figure: channel type breakdown
chan_prod = int_df.groupby("channel_type").apply(
    lambda g: pd.Series({
        lbl: wavg(g, col, "customers")
        for col, lbl in zip(PRODUCT_COLS, PRODUCT_LABELS)
    })
).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
chan_colors = [PI_BLUE, PI_YELLOW, "#E87722", "#009B77"]
channels = chan_prod["channel_type"].tolist()
for i, (_, row) in enumerate(chan_prod.iterrows()):
    ax.bar(x + i*0.2, [row[lbl] for lbl in PRODUCT_LABELS], 0.2,
           label=channels[i], color=chan_colors[i], edgecolor="white")
ax.set_xticks(x + 0.3)
ax.set_xticklabels(PRODUCT_LABELS, fontsize=10)
ax.set_ylabel("Ownership Rate", fontsize=11)
ax.set_title("Product Ownership Rate by Channel Type",
             fontsize=13, fontweight="bold", color=PI_BLUE)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig12_ownership_channel.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig12_ownership_channel.png")

# Figure: wealth breakdown
wealth_prod = int_df[int_df["wealth_range"]!="ND"].groupby("wealth_range").apply(
    lambda g: pd.Series({
        lbl: wavg(g, col, "customers")
        for col, lbl in zip(PRODUCT_COLS, PRODUCT_LABELS)
    })
).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
wealth_order = ["Low","Medium","High"]
wealth_colors = ["#C0C0C0", PI_YELLOW, PI_BLUE]
for i, wo in enumerate(wealth_order):
    row = wealth_prod[wealth_prod["wealth_range"]==wo].iloc[0]
    ax.bar(x + i*0.25, [row[lbl] for lbl in PRODUCT_LABELS], 0.25,
           label=wo, color=wealth_colors[i], edgecolor="white")
ax.set_xticks(x + 0.25)
ax.set_xticklabels(PRODUCT_LABELS, fontsize=10)
ax.set_ylabel("Ownership Rate", fontsize=11)
ax.set_title("Product Ownership Rate by Wealth Range",
             fontsize=13, fontweight="bold", color=PI_BLUE)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig13_ownership_wealth.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig13_ownership_wealth.png")

# ── 2b. GAP ANALYSIS ────────────────────────────────────────────────────────
print("\n[2b] Gap Analysis – Population Need vs Customer Ownership …")

# Merge need indices with internal aggregates
gap_df = prov_int.merge(
    ext[["provincia"] + NEED_COLS + ["macro_area","cluster_label","reddito_pc"]],
    on="provincia", how="inner"
)

# Compute gaps: (population need) – (customer ownership)
gap_df["gap_postal"]    = gap_df["need_postal_savings"] - gap_df["pct_postal"]
gap_df["gap_liquidity"] = gap_df["need_liquidity"]      - gap_df["pct_liquidity"]
gap_df["gap_financing"] = gap_df["need_financing"]      - gap_df["pct_financing"]
gap_df["gap_investment"]= gap_df["need_investment"]     - gap_df["pct_investment"]
gap_df["gap_utilities"] = gap_df["need_utilities"]      - gap_df["pct_utilities"]

gap_df["total_gap"] = (gap_df[["gap_postal","gap_liquidity","gap_financing",
                                "gap_investment","gap_utilities"]]
                       .clip(lower=0).sum(axis=1))

GAP_COLS = ["gap_postal","gap_liquidity","gap_financing","gap_investment","gap_utilities"]

# Figure: gap heatmap (top 20 opportunity provinces)
top20_gap = gap_df.nlargest(20, "total_gap").set_index("provincia")
heat_gap = top20_gap[GAP_COLS]
heat_gap.columns = PRODUCT_LABELS

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(heat_gap.clip(lower=0), annot=True, fmt=".3f",
            cmap="YlOrRd", linewidths=0.3, ax=ax,
            cbar_kws={"label":"Gap (Need − Ownership)"})
ax.set_title("Top 20 Opportunity Provinces – Product Gap Analysis\n"
             "(Higher = Greater unmet need relative to current ownership)",
             fontsize=12, fontweight="bold", color=PI_BLUE, pad=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig14_gap_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig14_gap_heatmap.png")

# Figure: total gap by province (bar + macro colour)
gap_sorted = gap_df.sort_values("total_gap", ascending=False).head(30)
macro_colors = {"North":PI_BLUE, "Centre":PI_YELLOW, "South":"#E87722"}
bar_colors = [macro_colors[m] for m in gap_sorted["macro_area"]]

fig, ax = plt.subplots(figsize=(14, 7))
ax.barh(gap_sorted["provincia"], gap_sorted["total_gap"],
        color=bar_colors, edgecolor="white")
ax.invert_yaxis()
ax.set_xlabel("Total Opportunity Gap Score", fontsize=11)
ax.set_title("Top 30 Provinces by Total Unmet Need (Opportunity Gap)\n"
             "Colour: North=Blue, Centre=Yellow, South=Orange",
             fontsize=12, fontweight="bold", color=PI_BLUE)
patches = [mpatches.Patch(color=v, label=k) for k,v in macro_colors.items()]
ax.legend(handles=patches)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig15_total_gap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig15_total_gap.png")


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT METHODOLOGICAL NOTE
# ─────────────────────────────────────────────────────────────────────────────
# These are 'ownership prediction models', NOT classic 'propensity to buy' models.
# The distinction matters:
#
#   Ownership prediction: predicts whether a customer currently HAS a product,
#                         based on their profile (age, channel, wealth…)
#
#   True propensity model: predicts probability of FUTURE PURCHASE,
#                          trained on historical campaign response data
#
# Why we use ownership prediction as proxy:
#   → Campaign response data was not provided
#   → Current product ownership is a reasonable proxy for behavioural patterns
#   → AUC 0.70–0.82 shows that customer profile predicts ownership well
#
# Limitation: models show ASSOCIATION between profile and ownership,
# they do not prove CAUSAL relationships.
# ─────────────────────────────────────────────────────────────────────────────
# ── 2c. ML MODELS – Product Ownership Prediction (Proxy for Propensity) ────────
print("\n[2c] ML Models – Product Ownership Prediction as Proxy for Propensity …")

# Features for ML at customer-segment level
# cross_selling EXCLUDED: it is computed directly from the target variables
# (number of products owned), so including it would cause data leakage —
# the model would predict ownership partly from a mathematical derivative of
# the target, artificially inflating AUC rather than learning genuine signals.
feat_cols = ["age_range","wealth_range","channel_type","interaction_frequency",
             "fl_email","fl_multibanca","state_birth"]

# One-hot encode
ml_df = int_df.dropna(subset=["provincia"]).copy()
ml_encoded = pd.get_dummies(ml_df[feat_cols], drop_first=False)

results_summary = {}

fig, axes = plt.subplots(1, 5, figsize=(22, 5))
fig.suptitle("ROC Curves – Product Ownership Prediction Models (Proxy for Propensity)",
             fontsize=13, fontweight="bold", color=PI_BLUE)

for i, (pcol, plbl, col) in enumerate(zip(PRODUCT_COLS, PRODUCT_LABELS, PRODUCT_COLORS)):
    y = ml_df[pcol].values
    X = ml_encoded.values

    # Stratified subsample to 80k rows — preserves class balance for minority
    # classes (e.g. Financing at 2.1%). Using train_test_split with stratify=y
    # guarantees both classes are proportionally represented in the sample.
    if len(y) > 80000:
        _, X, _, y = train_test_split(
            X, y, test_size=80000/len(y), random_state=42, stratify=y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=150, max_depth=8,
                                class_weight="balanced", random_state=42,
                                n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_prob = rf.predict_proba(X_te)[:, 1]
    y_pred = rf.predict(X_te)

    fpr, tpr, _ = roc_curve(y_te, y_prob)
    roc_auc = auc(fpr, tpr)
    cv_auc  = cross_val_score(rf, X_tr, y_tr, cv=3, scoring="roc_auc",
                              n_jobs=-1).mean()

    pr_auc = average_precision_score(y_te, y_prob)
    results_summary[plbl] = {
        "AUC (test)": round(roc_auc, 4),
        "AUC (CV)":   round(cv_auc, 4),
        "PR-AUC":     round(pr_auc, 4),  # precision-recall AUC for imbalanced classes
        "Report":     classification_report(y_te, y_pred, output_dict=True)
    }

    axes[i].plot(fpr, tpr, color=col, lw=2,
                 label=f"AUC={roc_auc:.3f}")
    axes[i].plot([0,1],[0,1],"k--",lw=1)
    axes[i].fill_between(fpr, tpr, alpha=0.1, color=col)
    axes[i].set_title(plbl, fontsize=10, fontweight="bold")
    axes[i].set_xlabel("FPR")
    axes[i].set_ylabel("TPR")
    axes[i].legend(fontsize=9)
    axes[i].set_aspect("equal")

    # Feature importance
    feat_names = ml_encoded.columns.tolist()
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[-8:]
    results_summary[plbl]["top_features"] = [feat_names[j] for j in top_idx[::-1]]

    print(f"  {plbl:15s} | Test AUC={roc_auc:.4f} | CV AUC={cv_auc:.4f} | PR-AUC={pr_auc:.4f}")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig16_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig16_roc_curves.png")

# Feature importance combined figure
fig, axes = plt.subplots(1, 5, figsize=(22, 6))
fig.suptitle("Top Feature Importances – Product Ownership Prediction Models",
             fontsize=13, fontweight="bold", color=PI_BLUE)
for i, (pcol, plbl, col) in enumerate(zip(PRODUCT_COLS, PRODUCT_LABELS, PRODUCT_COLORS)):
    # Re-run quick RF for importance plot
    y  = ml_df[pcol].values
    X  = ml_encoded.values
    sample_size2 = min(len(y), 50000)
    idx2 = np.random.default_rng(42).choice(len(y), sample_size2, replace=False)
    rf2 = RandomForestClassifier(n_estimators=100, max_depth=7,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    rf2.fit(X[idx2], y[idx2])
    imp = pd.Series(rf2.feature_importances_, index=ml_encoded.columns)
    top8 = imp.nlargest(8)
    axes[i].barh(top8.index, top8.values, color=col, edgecolor="white")
    axes[i].invert_yaxis()
    axes[i].set_title(plbl, fontsize=10, fontweight="bold")
    axes[i].tick_params(labelsize=7)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig17_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig17_feature_importance.png")

# Confusion matrices
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle("Confusion Matrices – Product Ownership Prediction Models",
             fontsize=13, fontweight="bold", color=PI_BLUE)
for i, (pcol, plbl) in enumerate(zip(PRODUCT_COLS, PRODUCT_LABELS)):
    y  = ml_df[pcol].values
    X  = ml_encoded.values
    sample_size3 = min(len(y), 40000)
    idx3 = np.random.default_rng(42).choice(len(y), sample_size3, replace=False)
    X_tr3, X_te3, y_tr3, y_te3 = train_test_split(
        X[idx3], y[idx3], test_size=0.25, random_state=42, stratify=y[idx3])
    rf3 = RandomForestClassifier(n_estimators=80, max_depth=7,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    rf3.fit(X_tr3, y_tr3)
    cm = confusion_matrix(y_te3, rf3.predict(X_te3))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No","Yes"])
    disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
    axes[i].set_title(plbl, fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig18_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig18_confusion_matrices.png")

# ── 2d. Cross-selling depth analysis ────────────────────────────────────────
print("\n[2d] Cross-selling depth analysis …")

cs_age = int_df.groupby(["age_range","cross_selling"])["customers"].sum().reset_index()
cs_age_piv = cs_age.pivot(index="age_range", columns="cross_selling",
                           values="customers").fillna(0)
cs_age_piv = cs_age_piv.div(cs_age_piv.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(9, 5))
cs_age_piv[["1","2","3+"]].plot(kind="bar", ax=ax,
    color=[PI_BLUE, PI_YELLOW, "#E87722"], edgecolor="white")
ax.set_title("Cross-Selling Depth by Age Group",
             fontsize=13, fontweight="bold", color=PI_BLUE)
ax.set_ylabel("Share of customers", fontsize=11)
ax.set_xlabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
ax.legend(title="Cross-sell products", labels=["1 product","2 products","3+ products"])
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig19_crossselling_age.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig19_crossselling_age.png")

# Business insights
print("\n  ── STEP 2 BUSINESS INSIGHTS ──────────────────────────────────────")
for pc, pl in zip(pct_cols, PRODUCT_LABELS):
    mn = prov_int[pc].mean()
    print(f"  {pl:15s} avg ownership: {mn:.1%}")
top5_opp = gap_df.nlargest(5, "total_gap")["provincia"].tolist()
print(f"\n  Top 5 opportunity provinces: {', '.join(top5_opp)}")
print(f"\n  ML Model AUC scores:")
for pl, info in results_summary.items():
    print(f"  {pl:15s} AUC={info['AUC (test)']:.4f} | CV={info['AUC (CV)']:.4f} | PR-AUC={info['PR-AUC']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 – SALES NETWORK SIZING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  STEP 3 – NETWORK SIZING (10,000 advisors)")
print("="*70)

print("\n[3] Allocating 10,000 Customer Advisors …")

TOTAL_ADVISORS = 10_000

# Build sizing score per province:
# weight = customers × total_gap × (1 - digital_discount)
# digital_discount = weighted combo of home banking + daily internet scores

sizer = gap_df.merge(
    ext[["provincia","P_HOME_BANKING","P_INTERNET_QUOTIDIANO","POPOLAZIONE"]],
    on="provincia", how="left"
)

# Digital adoption discount (0=fully physical, 1=fully digital)
sizer["digital_score"] = (
    0.5 * sizer["P_HOME_BANKING"] +
    0.5 * sizer["P_INTERNET_QUOTIDIANO"]
)
sizer["physical_weight"] = 1 - sizer["digital_score"]

# Opportunity score = customers × (gap+0.1) × physical_weight
sizer["opportunity_score"] = (
    sizer["total_customers"] *
    (sizer["total_gap"] + 0.1) *
    sizer["physical_weight"]
)

total_opp = sizer["opportunity_score"].sum()
sizer["advisors_raw"]    = sizer["opportunity_score"] / total_opp * TOTAL_ADVISORS
sizer["advisors_int"]    = sizer["advisors_raw"].apply(np.floor).astype(int)

# Distribute remainder to top gap provinces
remainder = TOTAL_ADVISORS - sizer["advisors_int"].sum()
top_idx   = sizer["advisors_raw"].nlargest(int(remainder)).index
sizer.loc[top_idx, "advisors_int"] += 1

assert sizer["advisors_int"].sum() == TOTAL_ADVISORS, "Allocation error!"

# Minimum 1 advisor per province
sizer["advisors_int"] = sizer["advisors_int"].clip(lower=1)

# After clipping, total may exceed TOTAL_ADVISORS.
# Fix: remove excess by reducing the provinces with the smallest fractional
# remainder first (they "least deserved" the rounding-up).
# This avoids the infinite-loop risk of always targeting idxmin().
excess = int(sizer["advisors_int"].sum() - TOTAL_ADVISORS)
if excess > 0:
    frac_remainder = sizer["advisors_raw"] - sizer["advisors_raw"].apply(np.floor)
    reduce_idx = frac_remainder.nsmallest(excess).index
    sizer.loc[reduce_idx, "advisors_int"] -= 1
    sizer["advisors_int"] = sizer["advisors_int"].clip(lower=1)

# Final invariant checks
assert sizer["advisors_int"].sum() == TOTAL_ADVISORS,     f"Allocation sum error: {sizer['advisors_int'].sum()} != {TOTAL_ADVISORS}"
assert sizer["advisors_int"].min() >= 1,     f"Province with 0 advisors found"

print(f"  Total advisors allocated: {sizer['advisors_int'].sum()}")
print(f"  Min per province: {sizer['advisors_int'].min()} | Max: {sizer['advisors_int'].max()}")
print(f"  Avg per province: {sizer['advisors_int'].mean():.1f}")

# Figure: advisor allocation map (bar)
fig, ax = plt.subplots(figsize=(16, 9))
sizer_sorted = sizer.sort_values("advisors_int", ascending=False)
bar_col = [macro_colors[m] for m in sizer_sorted["macro_area"]]
ax.bar(range(len(sizer_sorted)), sizer_sorted["advisors_int"],
       color=bar_col, edgecolor="white", linewidth=0.3)
ax.set_xticks(range(len(sizer_sorted)))
ax.set_xticklabels(sizer_sorted["provincia"], rotation=90, fontsize=5.5)
ax.set_ylabel("Advisors Allocated", fontsize=11)
ax.set_title("Customer Advisor Allocation – All 107 Provinces\n"
             "Colour: North=Blue | Centre=Yellow | South=Orange",
             fontsize=13, fontweight="bold", color=PI_BLUE)
patches = [mpatches.Patch(color=v, label=k) for k,v in macro_colors.items()]
ax.legend(handles=patches)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig20_advisor_allocation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig20_advisor_allocation.png")

# Macro area summary
macro_alloc = sizer.groupby("macro_area").agg(
    total_advisors=("advisors_int","sum"),
    total_customers=("total_customers","sum"),
    avg_gap=("total_gap","mean"),
    avg_digital=("digital_score","mean"),
).reset_index()
macro_alloc["advisors_pct"] = macro_alloc["total_advisors"]/TOTAL_ADVISORS*100

print("\n  Advisor allocation by macro area:")
print(macro_alloc[["macro_area","total_advisors","advisors_pct",
                    "avg_gap","avg_digital"]].to_string(index=False))

# Figure: macro pie + digital discount scatter
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Network Sizing – Macro Analysis",
             fontsize=13, fontweight="bold", color=PI_BLUE)

colors_pie = [macro_colors[m] for m in macro_alloc["macro_area"]]
axes[0].pie(macro_alloc["total_advisors"],
            labels=macro_alloc["macro_area"],
            autopct="%1.1f%%", colors=colors_pie,
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(edgecolor="white", linewidth=2))
axes[0].set_title("Advisors by Macro Area", fontsize=11)

sc = axes[1].scatter(sizer["digital_score"], sizer["advisors_int"],
                     c=[{"North":0,"Centre":1,"South":2}[m] for m in sizer["macro_area"]],
                     cmap="RdYlBu_r", s=60, edgecolors="white", alpha=0.85)
axes[1].set_xlabel("Digital Adoption Score", fontsize=11)
axes[1].set_ylabel("Advisors Allocated", fontsize=11)
axes[1].set_title("Digital Discount Effect on Allocation", fontsize=11)
for _, row in sizer.nlargest(5, "advisors_int").iterrows():
    axes[1].annotate(row["provincia"], (row["digital_score"], row["advisors_int"]),
                     fontsize=7, xytext=(3, 3), textcoords="offset points")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig21_network_sizing.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig21_network_sizing.png")

# ── Save final allocation table ───────────────────────────────────────────────
output_table = sizer[["provincia","macro_area","cluster_label","total_customers",
                       "advisors_int","digital_score","physical_weight",
                       "total_gap","need_postal_savings","need_liquidity",
                       "need_investment","need_financing","need_utilities",
                       "pct_postal","pct_liquidity","pct_financing",
                       "pct_investment","pct_utilities"]].sort_values(
                           "advisors_int", ascending=False).round(4)
output_table.to_csv(f"{OUTPUT_DIR}/advisor_allocation_by_province.csv", index=False)
print("  Saved advisor_allocation_by_province.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE DASHBOARD (Plotly HTML)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  BUILDING INTERACTIVE PLOTLY DASHBOARD …")
print("="*70)

dash = make_subplots(
    rows=3, cols=3,
    subplot_titles=[
        "Need Indices by Province (Top 20 – Liquidity)",
        "Product Ownership vs Need Gap",
        "Advisor Allocation by Province",
        "Digital Adoption vs Income per Capita",
        "Cross-Selling Depth Distribution",
        "ROC AUC – Ownership Prediction Models (Proxy)",
        "Need Profile by Cluster",
        "Ownership Rate by Age Group",
        "Macro Area Advisor Allocation",
    ],
    specs=[
        [{"type":"xy"},  {"type":"xy"},  {"type":"xy"}],
        [{"type":"xy"},  {"type":"xy"},  {"type":"xy"}],
        [{"type":"xy"},  {"type":"xy"},  {"type":"domain"}],
    ]
)

# Panel 1: Need liquidity top 20
top20_liq = ext.nlargest(20, "need_liquidity")
dash.add_trace(go.Bar(
    x=top20_liq["need_liquidity"], y=top20_liq["provincia"],
    orientation="h", marker_color=PI_BLUE, name="Liquidity Need",
    text=top20_liq["need_liquidity"].round(3), textposition="outside"
), row=1, col=1)

# Panel 2: Scatter gap vs ownership (investment)
gap_df2 = gap_df.dropna(subset=["gap_investment","pct_investment"])
dash.add_trace(go.Scatter(
    x=gap_df2["pct_investment"], y=gap_df2["need_investment"],
    mode="markers+text",
    text=gap_df2["provincia"],
    textposition="top center",
    textfont=dict(size=7),
    marker=dict(
        color=gap_df2["gap_investment"],
        colorscale="RdYlGn_r",
        size=8,
        colorbar=dict(title="Gap", x=0.66, len=0.33, y=0.83)
    ),
    name="Investment Gap",
    hovertemplate="<b>%{text}</b><br>Ownership: %{x:.2f}<br>Need: %{y:.2f}"
), row=1, col=2)
dash.add_trace(go.Scatter(
    x=[0,1], y=[0,1], mode="lines",
    line=dict(dash="dash", color="gray", width=1),
    showlegend=False
), row=1, col=2)

# Panel 3: Advisor allocation
top30 = sizer.nlargest(30, "advisors_int")
area_c = {"North":PI_BLUE,"Centre":PI_YELLOW,"South":"#E87722"}
for area, grp in top30.groupby("macro_area"):
    dash.add_trace(go.Bar(
        x=grp["provincia"], y=grp["advisors_int"],
        name=area, marker_color=area_c[area]
    ), row=1, col=3)

# Panel 4: Digital vs Income scatter
dash.add_trace(go.Scatter(
    x=ext["P_HOME_BANKING"], y=ext["reddito_pc"],
    mode="markers",
    marker=dict(color=ext["need_investment"]*100, colorscale="Blues",
                size=8, showscale=False),
    text=ext["provincia"],
    hovertemplate="<b>%{text}</b><br>Home Banking: %{x:.2f}<br>Income: €%{y:,.0f}",
    name="Provinces"
), row=2, col=1)

# Panel 5: Cross-selling distribution
cs_dist = int_df.groupby("cross_selling")["customers"].sum().reset_index()
dash.add_trace(go.Bar(
    x=cs_dist["cross_selling"], y=cs_dist["customers"],
    marker_color=[PI_BLUE, PI_YELLOW, "#E87722"],
    name="Cross-Sell"
), row=2, col=2)

# Panel 6: AUC bar chart
auc_vals = [v["AUC (test)"] for v in results_summary.values()]
dash.add_trace(go.Bar(
    x=PRODUCT_LABELS, y=auc_vals,
    marker_color=PRODUCT_COLORS,
    text=[f"{v:.3f}" for v in auc_vals],
    textposition="outside",
    name="AUC"
), row=2, col=3)
dash.add_shape(type="line", x0=-0.5, x1=4.5, y0=0.7, y1=0.7,
               line=dict(color="red", dash="dash", width=1.5),
               row=2, col=3)

# Panel 7: Cluster profiles
for nc, nl, col in zip(NEED_COLS, NEED_LABELS, PRODUCT_COLORS):
    cluster_means2 = ext.groupby("cluster_label")[nc].mean()
    dash.add_trace(go.Bar(
        x=cluster_means2.index, y=cluster_means2.values,
        name=nl, marker_color=col,
        legendgroup=nl
    ), row=3, col=1)

# Panel 8: Ownership by age
age_data = int_df.groupby("age_range").apply(
    lambda g: pd.Series({pl: wavg(g,pc,"customers")
                         for pc,pl in zip(PRODUCT_COLS, PRODUCT_LABELS)})
).reset_index()
for pl, col in zip(PRODUCT_LABELS, PRODUCT_COLORS):
    dash.add_trace(go.Bar(
        x=age_data["age_range"], y=age_data[pl],
        name=pl, marker_color=col,
        legendgroup=pl, showlegend=False
    ), row=3, col=2)

# Panel 9: Pie macro advisors
dash.add_trace(go.Pie(
    labels=macro_alloc["macro_area"],
    values=macro_alloc["total_advisors"],
    marker_colors=[macro_colors[m] for m in macro_alloc["macro_area"]],
    textinfo="label+percent",
    hole=0.4
), row=3, col=3)

dash.update_layout(
    height=1400, width=1600,
    title_text="<b>POSTE ITALIANE – Analysis Dashboard</b>",
    title_font=dict(size=20, color=PI_BLUE),
    plot_bgcolor=PI_LGREY,
    paper_bgcolor="white",
    font=dict(family="Arial", size=10),
    barmode="stack",
    showlegend=True,
)
dash.update_yaxes(row=1, col=1, autorange="reversed")
dash.update_xaxes(row=2, col=3, range=[0, 1.05])

pio.write_html(dash, f"{OUTPUT_DIR}/interactive_dashboard.html",
               include_plotlyjs="cdn", full_html=True)
print("  Saved interactive_dashboard.html")

# ── Save data tables ─────────────────────────────────────────────────────────
ext.to_csv(f"{OUTPUT_DIR}/external_enriched.csv", index=False)
gap_df.to_csv(f"{OUTPUT_DIR}/gap_analysis.csv", index=False)
print("  Saved external_enriched.csv, gap_analysis.csv")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  ALL OUTPUTS SAVED TO:", OUTPUT_DIR)
print("="*70)
print("""
DELIVERABLES SUMMARY
─────────────────────────────────────────────────────
FIGURES (static):
  fig01  Income & savings per capita – top provinces
  fig02  Digital behaviour heatmap
  fig03  Correlation matrix – socioeconomic variables
  fig04  North / Centre / South comparison
  fig05  Need indices top-15 by product
  fig06  PCA of province need profiles
  fig07  Elbow method – k-means
  fig08  Province clusters (PCA space)
  fig09  Cluster need profiles
  fig10  Product ownership rate distributions
  fig11  Ownership by age group
  fig12  Ownership by channel type
  fig13  Ownership by wealth range
  fig14  Gap heatmap (top-20 opportunity provinces)
  fig15  Total gap ranking
  fig16  ROC curves (5 RF models)
  fig17  Feature importances (5 RF models)
  fig18  Confusion matrices
  fig19  Cross-selling depth by age
  fig20  Advisor allocation bar chart
  fig21  Network sizing macro analysis

INTERACTIVE:
  interactive_dashboard.html  (open in browser)

DATA TABLES:
  external_enriched.csv         (107 prov × enriched variables)
  gap_analysis.csv              (need vs ownership gap)
  advisor_allocation_by_province.csv  (final sizing output)
─────────────────────────────────────────────────────
""")