#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script pour vérifier l'équilibre du dataset"""

import pandas as pd
from pathlib import Path

DATASET_PATH = Path("data/processed/dataset.csv")

def check_imbalance(df, target_column, verbose=True):
    """Analyse complète du déséquilibre"""
    class_counts = df[target_column].value_counts()
    class_percentages = df[target_column].value_counts(normalize=True) * 100
    
    majority_class_size = class_counts.max()
    minority_class_size = class_counts.min()
    imbalance_ratio = majority_class_size / minority_class_size
    minority_percentage = class_percentages.min()
    
    if minority_percentage >= 40:
        imbalance_level = "Équilibré"
    elif minority_percentage >= 20:
        imbalance_level = "Légèrement déséquilibré"
    elif minority_percentage >= 10:
        imbalance_level = "Déséquilibré"
    elif minority_percentage >= 5:
        imbalance_level = "Très déséquilibré"
    else:
        imbalance_level = "Extrêmement déséquilibré"
    
    if verbose:
        print("=" * 60)
        print("ANALYSE DU DÉSÉQUILIBRE")
        print("=" * 60)
        print(f"\nDistribution des classes:")
        for class_label in class_counts.index:
            print(f"  Classe {class_label}: {class_counts[class_label]:,} ({class_percentages[class_label]:.2f}%)")
        
        print(f"\n📊 Métriques:")
        print(f"  Ratio (majoritaire/minoritaire): {imbalance_ratio:.2f}")
        print(f"  Classe minoritaire: {minority_percentage:.2f}%")
        print(f"  Niveau: {imbalance_level}")
    
    return {
        'minority_percentage': minority_percentage,
        'imbalance_level': imbalance_level,
        'imbalance_ratio': imbalance_ratio
    }

if __name__ == "__main__":
    if not DATASET_PATH.exists():
        print(f"❌ Dataset non trouvé: {DATASET_PATH}")
        print("\nExécutez d'abord: py data_collection.py")
        exit(1)
    
    print("📂 Chargement du dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"✅ {len(df)} lignes chargées\n")
    
    results = check_imbalance(df, 'bad_nutrition')