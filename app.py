
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Value Bets – Prototype", layout="wide")

st.title("🎯 Prototype Value Bets (Football)")
st.caption("Demo pédagogique : prédit la probabilité de victoire à domicile et détecte les value bets.")

@st.cache_data
def load_data():
    df = pd.read_csv("matches_sample.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

def dec_to_prob(odds):
    return 1.0 / odds

def remove_vig_row(oh, od, oa):
    p = np.array([1/oh, 1/od, 1/oa], dtype=float)
    return p / p.sum()

def ev_decimal(p_model, o_dec):
    q = 1 - p_model
    return p_model * (o_dec - 1) - q

def kelly_fraction(p_model, o_dec, k=0.5):
    b = o_dec - 1
    if b <= 0:
        return 0.0
    f_star = (p_model * (b + 1) - 1) / b
    return max(0.0, k * f_star)

df = load_data()

# Split train/upcoming
train = df[df["is_upcoming"] == 0].copy()
upcoming = df[df["is_upcoming"] == 1].copy()

# Features: differences + home advantage implicit
def make_features(data):
    data = data.copy()
    data["rating_diff"] = data["home_rating"] - data["away_rating"]
    data["form_diff"] = data["home_form"] - data["away_form"]
    return data[["rating_diff", "form_diff"]]

X = make_features(train)
y = (train["result"] == "H").astype(int)

# Model
if y.nunique() == 1:
    # fallback in the extreme case
    base_rate = y.mean() if len(y) > 0 else 0.45
    model = None
else:
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

# Predict for upcoming
U = make_features(upcoming)
if model is None:
    p_home = np.full(len(U), y.mean() if len(y) > 0 else 0.45)
else:
    p_home = model.predict_proba(U)[:,1]

upcoming = upcoming.copy()
upcoming["p_model_home"] = p_home

# Market implied probs (no vig)
p_market = upcoming.apply(lambda r: remove_vig_row(r["odds_home"], r["odds_draw"], r["odds_away"]), axis=1, result_type="expand")
upcoming[["p_mkt_home","p_mkt_draw","p_mkt_away"]] = p_market

# EV & Kelly for home selection
upcoming["ev_home"] = upcoming.apply(lambda r: ev_decimal(r["p_model_home"], r["odds_home"]), axis=1)
bankroll = st.sidebar.number_input("Bankroll (€)", min_value=10.0, value=100.0, step=10.0)
kelly_k = st.sidebar.slider("Kelly fraction (prudence)", 0.1, 1.0, 0.5, 0.1)
upcoming["kelly_home"] = upcoming.apply(lambda r: kelly_fraction(r["p_model_home"], r["odds_home"], k=kelly_k), axis=1)
cap_pct = st.sidebar.slider("Cap mise (% bankroll)", 0.1, 5.0, 1.0, 0.1)
upcoming["stake_home"] = (upcoming["kelly_home"] * bankroll).clip(0, cap_pct/100*bankroll)

min_ev = st.sidebar.slider("Seuil EV minimum (€ par 1€ misé)", -0.5, 0.5, 0.0, 0.01)
min_prob = st.sidebar.slider("Proba min (modèle) victoire domicile", 0.0, 1.0, 0.45, 0.01)

# Table of value bets (home only, for simplicity)
vb = upcoming[(upcoming["ev_home"] > min_ev) & (upcoming["p_model_home"] >= min_prob)].copy()
vb["value_%"] = (vb["ev_home"] * 100).round(2)
vb = vb.sort_values(["ev_home","p_model_home"], ascending=False)

st.subheader("🔎 Opportunités détectées (sélection: Victoire **domicile**)")
if vb.empty:
    st.info("Aucune value bet avec les seuils actuels. Ajustez les filtres dans la barre latérale.")
else:
    show_cols = ["date","league","home_team","away_team","odds_home","p_model_home","p_mkt_home","ev_home","stake_home"]
    nice = vb[show_cols].copy()
    nice = nice.rename(columns={
        "date":"Date","league":"Ligue","home_team":"Domicile","away_team":"Extérieur",
        "odds_home":"Cote (Home)","p_model_home":"Proba IA (Home)","p_mkt_home":"Proba Marché (Home)",
        "ev_home":"EV (€ par 1€)","stake_home":"Mise (€)"
    })
    nice["Proba IA (Home)"] = (nice["Proba IA (Home)"]*100).round(1).astype(str) + "%"
    nice["Proba Marché (Home)"] = (nice["Proba Marché (Home)"]*100).round(1).astype(str) + "%"
    nice["EV (€ par 1€)"] = nice["EV (€ par 1€)"].round(3)
    nice["Mise (€)"] = nice["Mise (€)"].round(2)
    st.dataframe(nice, use_container_width=True)

st.divider()
with st.expander("⚙️ Détails techniques (simple)"):
    st.markdown("""
    - Modèle : **Logistic Regression** sur des features très simples (différence de rating et de forme).
    - Marché : cotes converties en probabilités **sans marge** (retrait du vig).
    - Value bet : pari quand **EV > 0**, mise via **Kelly fractionné** + cap.
    - 🚨 Demo : données **synthétiques**. Remplace `matches_sample.csv` par vos vraies données.
    """)

# Export
csv_export = vb[["date","league","home_team","away_team","odds_home","p_model_home","ev_home","stake_home"]].copy()
csv_export.to_csv("value_bets_export.csv", index=False)
st.download_button("⬇️ Exporter les value bets (CSV)", data=csv_export.to_csv(index=False).encode("utf-8"),
                   file_name="value_bets_export.csv", mime="text/csv")
