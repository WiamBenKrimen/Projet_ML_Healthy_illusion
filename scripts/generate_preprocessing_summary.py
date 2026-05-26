from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'data' / 'dataset.csv'
MD = ROOT / 'preprocessing_decisions.md'

if not DATASET.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET}")

print(f"Reading dataset: {DATASET}")
df = pd.read_csv(DATASET)

n = len(df)

# Missing
miss = df.isna().sum()
miss_pct = (miss / n * 100).round(2)

rows = []
for c in df.columns:
    pct = miss_pct[c]
    if pct == 0:
        action = 'Aucune action'
    elif pct < 5:
        action = 'Supprimer ligne (cas isolé) ou imputer (médiane/mode)'
    elif pct > 50:
        action = 'Supprimer colonne'
    else:
        action = 'Imputer (médiane ou mode)'
    rows.append((c, int(miss[c]), float(pct), action))

# Outliers via IQR for numeric cols
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
out_rows = []
for c in num_cols:
    col = df[c].dropna()
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    out = df[(df[c] < lower) | (df[c] > upper)][c]
    out_count = int(out.shape[0])
    out_pct = round(out_count / n * 100, 2)
    out_rows.append((c, float(q1), float(q3), float(iqr), float(lower), float(upper), out_count, out_pct))

now = datetime.now().isoformat(timespec='seconds')

md_lines = []
md_lines.append(f"\n---\n\n**Section générée automatiquement le {now}**\n")
md_lines.append('\n## Tableau automatique des valeurs manquantes (généré)\n')
md_lines.append('| Variable | Valeurs manquantes | Pourcentage (%) | Décision proposée |')
md_lines.append('|:--|--:|--:|:--:|')
for r in rows:
    md_lines.append(f'| {r[0]} | {r[1]} | {r[2]} | {r[3]} |')

md_lines.append('\n## Résumé automatique des outliers (méthode IQR)\n')
md_lines.append('| Variable | Q1 | Q3 | IQR | Lower bound | Upper bound | Outliers count | Outliers % |')
md_lines.append('|:--|--:|--:|--:|--:|--:|--:|--:|')
for r in out_rows:
    md_lines.append(f'| {r[0]} | {r[1]:.6g} | {r[2]:.6g} | {r[3]:.6g} | {r[4]:.6g} | {r[5]:.6g} | {r[6]} | {r[7]} |')

md_lines.append('\n\n')

print(f"Appending results to {MD}")
with open(MD, 'a', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))

print('Done.')
