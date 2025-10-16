import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Value Bets – AI Copilot (H/D/A)", layout="wide")
st.title("🤖 Value Bets – Copilote IA (1X2: Domicile / Nul / Extérieur)")
st.caption("Charge ton CSV (aucune donnée fictive). L'IA entraîne un modèle multinomial (H/D/A) et détecte des value bets sur les trois sélections.")

REQUIRED_COLS = [
    "date","league","home_team","away_team",
    "home_rating","away_rating","home_form","away_form",
    "odds_home","odds_draw","odds_away",
    "result","is_upcoming"
]

# ------- Helpers -------
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

def make_features(data):
    data = data.copy()
    data["rating_diff"] = data["home_rating"] - data["away_rating"]
    data["form_diff"] = data["home_form"] - data["away_form"]
    return data[["rating_diff","form_diff"]]

def train_multinomial(train_df):
    X = make_features(train_df)
    y = train_df["result"].astype(str)
    if y.nunique() < 2 or len(X)==0:
        return None, y.value_counts(normalize=True).to_dict(), None
    model = LogisticRegression(max_iter=4000, multi_class="multinomial")
    model.fit(X, y)
    class_order = list(model.classes_)
    return (model, {"priors": y.value_counts(normalize=True).to_dict(), "classes": class_order},
            dict(coef=model.coef_.tolist(), feat=["rating_diff","form_diff"], intercept=model.intercept_.tolist(), classes=class_order))

def score_upcoming_mkt(model, upcoming_df, bankroll, kelly_k, cap_pct):
    U = make_features(upcoming_df)
    if model is not None:
        probs = model.predict_proba(U)
        classes = model.classes_
    else:
        classes = np.array(["A","D","H"])
        probs = np.tile(np.array([1/3,1/3,1/3]), (len(U),1))

    df = upcoming_df.copy().reset_index(drop=True)
    wanted = ["H","D","A"]
    P = np.zeros((len(df), 3))
    for j, lab in enumerate(classes):
        if lab in wanted:
            P[:, wanted.index(lab)] = probs[:, j]
    df["p_model_home"] = P[:,0]
    df["p_model_draw"] = P[:,1]
    df["p_model_away"] = P[:,2]

    p_market = df.apply(lambda r: remove_vig_row(r["odds_home"], r["odds_draw"], r["odds_away"]), axis=1, result_type="expand")
    df[["p_mkt_home","p_mkt_draw","p_mkt_away"]] = p_market

    for side, colp, colo in [("home","p_model_home","odds_home"),
                             ("draw","p_model_draw","odds_draw"),
                             ("away","p_model_away","odds_away")]:
        df[f"ev_{side}"] = df.apply(lambda r: ev_decimal(r[colp], r[colo]), axis=1)
        df[f"kelly_{side}"] = df.apply(lambda r: kelly_fraction(r[colp], r[colo], k=kelly_k), axis=1)
        df[f"stake_{side}"] = (df[f"kelly_{side}"] * bankroll).clip(0, cap_pct/100*bankroll)

    return df

def explain_row_side(row, side):
    reasons = []
    if side == "home":
        if row.get("rating_diff",0) > 0: reasons.append(f"avantage rating domicile (+{int(row['rating_diff'])})")
        if row.get("form_diff",0) > 0: reasons.append(f"meilleure forme domicile (+{int(row['form_diff'])})")
        if row["p_model_home"] > row["p_mkt_home"]: reasons.append("proba IA > proba marché")
        if row["ev_home"] > 0: reasons.append(f"EV +{row['ev_home']:.3f}€/€")
    elif side == "draw":
        reasons.append("match équilibré (écarts rating/forme faibles)")
        if row["p_model_draw"] > row["p_mkt_draw"]: reasons.append("marché sous-estime le nul")
        if row["ev_draw"] > 0: reasons.append(f"EV +{row['ev_draw']:.3f}€/€")
    else:  # away
        if row.get("rating_diff",0) < 0: reasons.append(f"avantage rating extérieur ({int(-row['rating_diff'])})")
        if row.get("form_diff",0) < 0: reasons.append(f"meilleure forme extérieur ({int(-row['form_diff'])})")
        if row["p_model_away"] > row["p_mkt_away"]: reasons.append("proba IA > proba marché")
        if row["ev_away"] > 0: reasons.append(f"EV +{row['ev_away']:.3f}€/€")
    if not reasons:
        reasons.append("edge modeste mais positif selon le modèle")
    return " ; ".join(reasons)

def pick_best(vb_all):
    if vb_all.empty:
        return None
    return vb_all.sort_values("EV", ascending=False).iloc[0]

def nl_response(intent, state):
    text = intent.lower()
    vb_all = state.get("vb_all", pd.DataFrame())
    if any(k in text for k in ["pari", "aujourd'hui", "today", "ce soir", "suggestion"]):
        if vb_all.empty:
            return "Je ne détecte aucun value bet avec les seuils actuels et la sélection choisie."
        top = pick_best(vb_all)
        why = top["Pourquoi"]
        return (f"Je suggère **{top['Domicile']} vs {top['Extérieur']}** — **{top['Sélection']}**.\n"
                f"- Cote: {top['Cote']}\n"
                f"- Proba IA: {top['Proba IA']} ; Proba marché: {top['Proba Marché']}\n"
                f"- Mise conseillée: {top['Mise (€)']:.2f}€\n"
                f"**Pourquoi**: {why}")
    if any(k in text for k in ["pourquoi", "expliquer", "explain"]):
        if vb_all.empty:
            return "Aucune opportunité à expliquer pour l'instant."
        top = pick_best(vb_all)
        return f"Raison principale: {top['Pourquoi']}."
    if any(k in text for k in ["bankroll", "budget", "mise", "stake"]):
        return ("Ajuste la **Bankroll** et la **fraction de Kelly** dans la barre latérale. "
                "Plus la fraction est petite, plus tu réduis le risque.")
    if any(k in text for k in ["help", "aide"]):
        return "Demande-moi 'Quel pari aujourd'hui ?' ou 'Pourquoi ce pari ?' ou 'Comment régler la bankroll ?'"
    return "Je n'ai pas bien compris. Essaie : 'Quel pari aujourd'hui ?'."

# ------- Sidebar -------
st.sidebar.subheader("⚙️ Paramètres IA & Filtres")
bankroll = st.sidebar.number_input("Bankroll (€)", min_value=10.0, value=100.0, step=10.0)
kelly_k = st.sidebar.slider("Kelly fraction (prudence)", 0.1, 1.0, 0.5, 0.1)
cap_pct = st.sidebar.slider("Cap mise (% bankroll)", 0.1, 5.0, 1.0, 0.1)
min_ev = st.sidebar.slider("Seuil EV minimum (€ par 1€ misé)", -0.5, 0.5, 0.0, 0.01)
min_prob = st.sidebar.slider("Proba min (modèle)", 0.0, 1.0, 0.33, 0.01)
selection = st.sidebar.selectbox("Sélection", ["Domicile", "Nul", "Extérieur"])

# ------- Upload -------
st.subheader("📤 Charge ton CSV (aucune donnée fictive)")
uploaded = st.file_uploader("Choisis un fichier CSV", type=["csv"])
if not uploaded:
    st.info("En-têtes obligatoires : " + ", ".join(REQUIRED_COLS))
    st.stop()

# Read & validate
try:
    df = pd.read_csv(uploaded)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    for c in ["home_rating","away_rating","home_form","away_form","odds_home","odds_draw","odds_away","is_upcoming"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
except Exception as e:
    st.error(f"Erreur de lecture/format: {e}")
    st.stop()

missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Colonnes manquantes : {missing}")
    st.stop()

train = df[df["is_upcoming"] == 0].copy()
upcoming = df[df["is_upcoming"] == 1].copy()

if train.empty or upcoming.empty:
    st.warning("Il faut au moins 1 match historique et 1 match à venir.")
    st.stop()

# Train & score
model, priors_info, coefs = train_multinomial(train)
scored = score_upcoming_mkt(model, upcoming, bankroll, kelly_k, cap_pct)

# Build VB tables for each side
def format_vb(scored, side):
    if side == "Domicile":
        p_col, mkt_col, o_col, ev_col, stake_col = "p_model_home","p_mkt_home","odds_home","ev_home","stake_home"
    elif side == "Nul":
        p_col, mkt_col, o_col, ev_col, stake_col = "p_model_draw","p_mkt_draw","odds_draw","ev_draw","stake_draw"
    else:
        p_col, mkt_col, o_col, ev_col, stake_col = "p_model_away","p_mkt_away","odds_away","ev_away","stake_away"
    vb = scored[(scored[ev_col] > min_ev) & (scored[p_col] >= min_prob)].copy()
    vb = vb.sort_values([ev_col, p_col], ascending=False)
    if vb.empty:
        return vb, pd.DataFrame()
    nice = vb[[
        "date","league","home_team","away_team","rating_diff","form_diff", o_col, p_col, mkt_col, ev_col, stake_col
    ]].copy()
    nice.columns = ["Date","Ligue","Domicile","Extérieur","Δ Rating","Δ Forme","Cote","Proba IA","Proba Marché","EV","Mise (€)"]
    nice["Proba IA"] = (nice["Proba IA"]*100).round(1).astype(str) + "%"
    nice["Proba Marché"] = (nice["Proba Marché"]*100).round(1).astype(str) + "%"
    nice["EV"] = nice["EV"].round(3)
    nice["Mise (€)"] = nice["Mise (€)"].round(2)

    # unified df for copilot
    uni = nice.copy()
    uni["Sélection"] = side
    # reconstruct numeric proba for copilot message
    uni["_p_ia"] = vb[p_col].values
    uni["_p_mkt"] = vb[mkt_col].values
    uni["_cote"] = vb[o_col].values

    # reasons
    reasons = []
    for _, r in vb.iterrows():
        s = "home" if side=="Domicile" else ("draw" if side=="Nul" else "away")
        reasons.append(explain_row_side(r, s))
    uni["Pourquoi"] = reasons
    uni["Proba IA"] = (uni["_p_ia"]*100).round(1).astype(str) + "%"
    uni["Proba Marché"] = (uni["_p_mkt"]*100).round(1).astype(str) + "%"
    uni["Cote"] = uni["_cote"]
    return nice, uni[["Date","Ligue","Domicile","Extérieur","Sélection","Cote","Proba IA","Proba Marché","EV","Mise (€)","Pourquoi"]]

st.subheader(f"🔎 Opportunités — Sélection : **{selection}**")
nice_sel, uni_sel = format_vb(scored, selection)
if nice_sel.empty:
    st.info("Aucune value bet avec les seuils actuels.")
else:
    st.dataframe(nice_sel, use_container_width=True)

_, uni_home = format_vb(scored, "Domicile")
_, uni_draw = format_vb(scored, "Nul")
_, uni_away = format_vb(scored, "Extérieur")
vb_all = pd.concat([uni_home, uni_draw, uni_away], ignore_index=True) if not uni_home.empty or not uni_draw.empty or not uni_away.empty else pd.DataFrame()

csv_export = nice_sel.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Exporter (CSV)", data=csv_export, file_name="value_bets_selected.csv", mime="text/csv")

# ------- AI Copilot (chat) -------
st.divider()
st.subheader("💬 Copilote IA (chat)")
st.write("Exemples : *« Quel pari aujourd’hui ? », « Pourquoi ce pari ? », « Comment régler la bankroll ? »*")

if "history" not in st.session_state: st.session_state["history"] = []
if "vb_all" not in st.session_state: st.session_state["vb_all"] = vb_all

for role, msg in st.session_state["history"][:50]:
    st.chat_message(role).markdown(msg)

q = st.chat_input("Écris ici…")
if q:
    st.session_state["history"].append(("user", q))
    st.chat_message("user").markdown(q)
    reply = nl_response(q, {"vb_all": st.session_state["vb_all"]})
    st.session_state["history"].append(("assistant", reply))
    st.chat_message("assistant").markdown(reply)

with st.expander("ℹ️ Détails techniques"):
    st.markdown("""
- Modèle : **régression logistique multinomiale** (1X2) sur deux features (Δ rating, Δ forme).
- Marché : cotes converties en proba **sans marge** (retrait du vig).
- Détection sur **trois sélections** : Domicile, Nul, Extérieur.
- Mise : **Kelly fractionné** + plafonnement par pourcentage.
- Prochaines étapes : calibration, Over/Under, intégration API de cotes, historique & ROI.
    """)
