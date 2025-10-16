
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import requests
import os, datetime as dt
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="El Chipo Pronos – IA 1X2 (Auto + API, 6h)", layout="wide")
st.title("🔥 El Chipo Pronos – Copilote IA (1X2) • Auto-MAJ 6 h + API gratuite (UE + Champions League)")
st.caption("Sources: Upload CSV, URL CSV distante, ou **API gratuite** (football-data.org + The Odds API). Cache 6 h + bouton Rafraîchir.")

REQUIRED_COLS = [
    "date","league","home_team","away_team",
    "home_rating","away_rating","home_form","away_form",
    "odds_home","odds_draw","odds_away",
    "result","is_upcoming"
]

# ---------- Helpers ----------
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

# ---------- Data sources ----------
st.sidebar.header("📡 Source de données")
src_mode = st.sidebar.radio("Choisis la source :", ["Upload CSV", "URL CSV distante", "API (gratuit)"], index=0)

# API config
with st.sidebar.expander("🔑 Clés API gratuites"):
    FD_TOKEN = st.text_input("football-data.org (X-Auth-Token)", type="password", help="Crée un compte gratuit sur football-data.org")
    ODDS_API_KEY = st.text_input("The Odds API (apiKey)", type="password", help="Gratuit avec quota mensuel")
    region = st.selectbox("Région bookmakers (Odds API)", ["eu","uk","us","au"], index=0)
    # Par défaut: Ligue 1 (2015), Premier League (2021), La Liga (2014), Serie A (2019), Bundesliga (2002), Champions League (2001)
    comps = st.text_input("Competitions football-data (IDs)", "2015,2021,2014,2019,2002,2001")

@st.cache_data(ttl=60*60*6, show_spinner=False)  # 6h
def fd_fixtures_results(fd_token: str, comps_csv: str) -> pd.DataFrame:
    base = "https://api.football-data.org/v4"
    headers = {"X-Auth-Token": fd_token}
    today = dt.date.today()
    start = (today - dt.timedelta(days=30)).isoformat()
    end = (today + dt.timedelta(days=7)).isoformat()
    frames = []
    for comp in [c.strip() for c in comps_csv.split(",") if c.strip()]:
        url = f"{base}/competitions/{comp}/matches?dateFrom={start}&dateTo={end}"
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        js = r.json().get("matches", [])
        for m in js:
            frames.append({
                "date": m["utcDate"][:10],
                "league": comp,
                "home_team": m["homeTeam"]["name"],
                "away_team": m["awayTeam"]["name"],
                "home_rating": 75, "away_rating": 75,
                "home_form": 3, "away_form": 3,
                "odds_home": None, "odds_draw": None, "odds_away": None,
                "result": ("H" if (m["score"]["winner"]=="HOME_TEAM") else ("A" if m["score"]["winner"]=="AWAY_TEAM" else ("D" if m["status"]=="FINISHED" else ""))),
                "is_upcoming": 0 if m["status"]=="FINISHED" else 1
            })
    df = pd.DataFrame(frames)
    if not df.empty:
        df = df.drop_duplicates(subset=["date","league","home_team","away_team"])
    return df

@st.cache_data(ttl=60*60*6, show_spinner=False)  # 6h
def oddsapi_1x2(odds_api_key: str, region: str = "eu") -> pd.DataFrame:
    sport_keys = [
        "soccer_france_ligue_one",
        "soccer_epl",
        "soccer_spain_la_liga",
        "soccer_italy_serie_a",
        "soccer_germany_bundesliga",
        "soccer_uefa_champs_league"
    ]
    rows = []
    for sk in sport_keys:
        url = f"https://api.the-odds-api.com/v4/sports/{sk}/odds"
        params = {"apiKey": odds_api_key, "regions": region, "markets": "h2h", "oddsFormat": "decimal"}
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 402:
            # quota épuisé → on revient avec ce qu'on a
            continue
        r.raise_for_status()
        for ev in r.json():
            home, away = ev["home_team"], ev["away_team"]
            best_home = best_draw = best_away = None
            for bk in ev.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    if mk.get("key") != "h2h": continue
                    outcomes = mk.get("outcomes", [])
                    for o in outcomes:
                        if o["name"] == home:
                            best_home = max(best_home or 0, float(o["price"]))
                        elif o["name"] == away:
                            best_away = max(best_away or 0, float(o["price"]))
                        elif o["name"].lower() in ("draw","tie"):
                            best_draw = max(best_draw or 0, float(o["price"]))
            rows.append({"home_team": home, "away_team": away,
                         "odds_home": best_home, "odds_draw": best_draw, "odds_away": best_away})
    return pd.DataFrame(rows)

def join_fixtures_odds(df_fix: pd.DataFrame, df_odds: pd.DataFrame) -> pd.DataFrame:
    if df_fix.empty: return df_fix
    if df_odds is None or df_odds.empty: return df_fix
    A = df_fix.assign(_h=df_fix["home_team"].str.lower().str.strip(),
                      _a=df_fix["away_team"].str.lower().str.strip())
    B = df_odds.assign(_h=df_odds["home_team"].str.lower().str.strip(),
                       _a=df_odds["away_team"].str.lower().str.strip())
    M = A.merge(B[["_h","_a","odds_home","odds_draw","odds_away"]], on=["_h","_a"], how="left")
    return M.drop(columns=["_h","_a"])

@st.cache_data(ttl=60*60*6)  # 6h
def fetch_remote_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.content.decode("utf-8")))

refresh = st.sidebar.button("🔄 Rafraîchir maintenant (vider le cache)")
if refresh:
    st.cache_data.clear()
    st.success("Cache vidé. Recharge des données…")

df = None

if src_mode == "Upload CSV":
    st.subheader("📤 Charge ton CSV (aucune donnée fictive)")
    uploaded = st.file_uploader("Choisis un fichier CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.caption("✅ Fichier chargé depuis l'upload.")
elif src_mode == "URL CSV distante":
    st.subheader("🌍 URL CSV distante (Google Sheets / GitHub raw)")
    default_url = st.text_input("URL CSV", value="", placeholder="https://.../export?format=csv")
    if default_url:
        try:
            df = fetch_remote_csv(default_url)
            st.caption("✅ Fichier chargé depuis l'URL distante (cache 6 h).")
        except Exception as e:
            st.error(f"Impossible de charger l'URL : {e}")
else:  # API (gratuit)
    st.subheader("🛰️ Source API gratuite (fixtures football-data + cotes The Odds API)")
    if not FD_TOKEN:
        st.warning("Entre ton **X-Auth-Token football-data.org** pour charger les matchs.")
        st.stop()
    try:
        fixtures = fd_fixtures_results(FD_TOKEN, comps)
    except Exception as e:
        st.error(f"Erreur football-data.org : {e}")
        st.stop()

    odds_df = None
    if ODDS_API_KEY:
        try:
            odds_df = oddsapi_1x2(ODDS_API_KEY, region=region)
        except Exception as e:
            st.warning(f"Cotes non chargées (The Odds API) : {e}")

    df = join_fixtures_odds(fixtures, odds_df)
    if df.empty:
        st.info("Aucun match trouvé pour les compétitions sélectionnées.")
        st.stop()
    st.success("✅ Données chargées via API (cache 6 h).")
    st.dataframe(df.head(20), use_container_width=True)

if df is None:
    st.info("Aucune donnée chargée pour l'instant. Utilise l'une des trois sources ci-dessus.")
    st.stop()

# ---------- Validation ----------
try:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Colonnes manquantes : {missing}")
        st.stop()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    for c in ["home_rating","away_rating","home_form","away_form","odds_home","odds_draw","odds_away","is_upcoming"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
except Exception as e:
    st.error(f"Erreur de lecture/format: {e}")
    st.stop()

# ---------- Sidebar (IA params) ----------
st.sidebar.subheader("⚙️ Paramètres IA & Filtres")
bankroll = st.sidebar.number_input("Bankroll (€)", min_value=10.0, value=100.0, step=10.0)
kelly_k = st.sidebar.slider("Kelly fraction (prudence)", 0.1, 1.0, 0.5, 0.1)
cap_pct = st.sidebar.slider("Cap mise (% bankroll)", 0.1, 5.0, 1.0, 0.1)
min_ev = st.sidebar.slider("Seuil EV minimum (€ par 1€ misé)", -0.5, 0.5, 0.0, 0.01)
min_prob = st.sidebar.slider("Proba min (modèle)", 0.0, 1.0, 0.33, 0.01)
selection = st.sidebar.selectbox("Sélection", ["Domicile", "Nul", "Extérieur"])

# ---------- Split ----------
train = df[df["is_upcoming"] == 0].copy()
upcoming = df[df["is_upcoming"] == 1].copy()

if train.empty or upcoming.empty:
    st.warning("Il faut au moins 1 match **historique** (is_upcoming=0) et 1 match **à venir** (is_upcoming=1).")
    st.stop()

# ---------- Train & Score ----------
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

    # market probs (no vig) only if odds present
    def safe_row(r):
        try:
            if pd.notna(r["odds_home"]) and pd.notna(r["odds_draw"]) and pd.notna(r["odds_away"]):
                return remove_vig_row(r["odds_home"], r["odds_draw"], r["odds_away"])
        except Exception:
            pass
        return [np.nan, np.nan, np.nan]

    p_market = df.apply(lambda r: pd.Series(safe_row(r)), axis=1)
    df[["p_mkt_home","p_mkt_draw","p_mkt_away"]] = p_market.values

    for side, colp, colo in [("home","p_model_home","odds_home"),
                             ("draw","p_model_draw","odds_draw"),
                             ("away","p_model_away","odds_away")]:
        df[f"ev_{side}"] = df.apply(lambda r: ev_decimal(r[colp], r[colo]) if pd.notna(r[colo]) else np.nan, axis=1)
        df[f"kelly_{side}"] = df.apply(lambda r: kelly_fraction(r[colp], r[colo], k=kelly_k) if pd.notna(r[colo]) else 0.0, axis=1)
        df[f"stake_{side}"] = (df[f"kelly_{side}"] * bankroll).clip(0, cap_pct/100*bankroll)
    return df

model, priors_info, coefs = train_multinomial(train)
scored = score_upcoming_mkt(model, upcoming, bankroll, kelly_k, cap_pct)

# ---------- View builders ----------
def explain_row_for_side(r, side_name):
    s = "home" if side_name=="Domicile" else ("draw" if side_name=="Nul" else "away")
    return explain_row_side(r, s)

def format_vb(scored, side):
    if side == "Domicile":
        p_col, mkt_col, o_col, ev_col, stake_col = "p_model_home","p_mkt_home","odds_home","ev_home","stake_home"
    elif side == "Nul":
        p_col, mkt_col, o_col, ev_col, stake_col = "p_model_draw","p_mkt_draw","odds_draw","ev_draw","stake_draw"
    else:
        p_col, mkt_col, o_col, ev_col, stake_col = "p_model_away","p_mkt_away","odds_away","ev_away","stake_away"
    vb = scored[(scored[ev_col].fillna(-1) > min_ev) & (scored[p_col] >= min_prob)].copy()
    vb = vb.sort_values([ev_col, p_col], ascending=False)
    if vb.empty:
        return vb, pd.DataFrame()
    nice = vb[[
        "date","league","home_team","away_team","rating_diff","form_diff", o_col, p_col, mkt_col, ev_col, stake_col
    ]].copy()
    nice.columns = ["Date","Ligue","Domicile","Extérieur","Δ Rating","Δ Forme","Cote","Proba IA","Proba Marché","EV","Mise (€)"]
    nice["Proba IA"] = (nice["Proba IA"]*100).round(1).astype(str) + "%"
    nice["Proba Marché"] = nice["Proba Marché"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
    nice["EV"] = nice["EV"].round(3)
    nice["Mise (€)"] = nice["Mise (€)"].round(2)

    # unified df for copilot
    uni = nice.copy()
    uni["Sélection"] = side
    uni["_p_ia"] = vb[p_col].values
    uni["_p_mkt"] = vb[mkt_col].values
    uni["_cote"] = vb[o_col].values
    reasons = []
    for _, r in vb.iterrows():
        reasons.append(explain_row_for_side(r, side))
    uni["Pourquoi"] = reasons
    uni["Proba IA"] = (uni["_p_ia"]*100).round(1).astype(str) + "%"
    uni["Proba Marché"] = [f"{x*100:.1f}%" if pd.notna(x) else "—" for x in vb[mkt_col].values]
    uni["Cote"] = vb[o_col].values
    return nice, uni[["Date","Ligue","Domicile","Extérieur","Sélection","Cote","Proba IA","Proba Marché","EV","Mise (€)","Pourquoi"]]

# ---------- Show tables ----------
st.subheader(f"🔎 Opportunités — Sélection : **{selection}**")
nice_sel, uni_sel = format_vb(scored, selection)
if nice_sel.empty:
    st.info("Aucune value bet avec les seuils actuels.")
else:
    st.dataframe(nice_sel, use_container_width=True)

# Combined for copilot
_, uni_home = format_vb(scored, "Domicile")
_, uni_draw = format_vb(scored, "Nul")
_, uni_away = format_vb(scored, "Extérieur")
vb_all = pd.concat([uni_home, uni_draw, uni_away], ignore_index=True) if not uni_home.empty or not uni_draw.empty or not uni_away.empty else pd.DataFrame()

# Export
csv_export = nice_sel.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Exporter (CSV)", data=csv_export, file_name="el_chipo_pronos_value_bets.csv", mime="text/csv")

# ---------- Copilot ----------
st.divider()
st.subheader("💬 Copilote IA (El Chipo)")
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

with st.expander("ℹ️ Détails tech & quotas (6 h)"):
    st.markdown("""
- **football-data.org (gratuit)**: calendriers/résultats. IDs par défaut: Ligue 1 (2015), Premier League (2021), La Liga (2014), Serie A (2019), Bundesliga (2002), **Champions League (2001)**.
- **The Odds API (gratuit)**: clés pour UE + **soccer_uefa_champs_league** ajoutée aux sports.
- **Cache 6 h**: les appels API/URL sont mis en cache pendant 6 heures. Bouton **Rafraîchir** pour forcer une mise à jour.
- Si les cotes ne sont pas renvoyées (quota épuisé), l'app calcule quand même les **probas IA** (EV = `NaN`).
    """)
