
# 🔥 El Chipo Pronos — IA Value Bets (1X2)

Appli Streamlit **sans données fictives** : charge ton CSV, l’IA entraîne un modèle **multinomial (Domicile / Nul / Extérieur)**, détecte les **value bets** et propose la **mise (Kelly)**. Mini **copilote IA** intégré.

## 🚀 Lancer en local
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## 🧾 Colonnes attendues (CSV)
`date, league, home_team, away_team, home_rating, away_rating, home_form, away_form, odds_home, odds_draw, odds_away, result, is_upcoming`

- `date` : YYYY-MM-DD
- `result` : H / D / A (laisser vide pour les matchs à venir)
- `is_upcoming` : 1 (à venir) ou 0 (historique)

## 🌐 Déploiement gratuit (Streamlit Cloud)
1. Crée un repo GitHub public (ex. `el-chipo-pronos`).
2. Ajoute ces fichiers : `app.py`, `requirements.txt`, `value_bets_template.csv`, `README.md`.
3. Va sur https://share.streamlit.io → **New app** → choisis ton repo → fichier principal : `app.py` → **Deploy**.

---
👑 **El Chipo Pronos** — Analyse IA, Value Bets & Gestion de Bankroll  
© 2025 El Chipo Labs
