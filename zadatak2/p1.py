import pandas as pd
import numpy as np

# Učitavanje podataka iz CSV datoteke
print("Učitavanje podataka...")
df = pd.read_csv('data.csv')

# Prepoznavanje nedostajućih vrijednosti
print("\nBroj nedostajućih vrijednosti po stupcima:")
missing_values = df.isnull().sum()

# Ispis samo stupaca koji imaju nedostajuće vrijednosti
missing_columns = missing_values[missing_values > 0]
if len(missing_columns) > 0:
    print(missing_columns)
else:
    print("Nema nedostajućih vrijednosti u podacima.")

# Ispis ukupnog broja nedostajućih vrijednosti
total_missing = df.isnull().sum().sum()
print(f"\nUkupan broj nedostajućih vrijednosti: {total_missing}")

# Ispis postotka nedostajućih vrijednosti
percent_missing = (total_missing / (df.shape[0] * df.shape[1])) * 100
print(f"Postotak nedostajućih vrijednosti: {percent_missing:.2f}%")

# Jednostavna funkcija za prepoznavanje uniformnih ili nelogičnih distribucija
def prepoznaj_nelogicne_distribucije(df):
    print("\n--- PREPOZNAVANJE UNIFORMNIH ILI NELOGIČNIH DISTRIBUCIJA ---")
    
    # Provjera uniformnih stupaca (samo jedna vrijednost)
    uniform_columns = []
    for col in df.columns:
        if df[col].nunique() == 1:
            uniform_columns.append(col)
    
    if uniform_columns:
        print(f"\nStupci s uniformnom distribucijom (samo jedna vrijednost):")
        for col in uniform_columns:
            print(f"  - {col}: {df[col].iloc[0]}")
    
    # Provjera neravnomjernih distribucija u numeričkim stupcima
    numeric_columns = df.select_dtypes(include=['number']).columns
    imbalanced_numeric = {}
    
    for col in numeric_columns:
        zeros_pct = (df[col] == 0).mean() * 100
        if zeros_pct > 90:
            imbalanced_numeric[col] = f"{zeros_pct:.1f}% nula"
    
    if imbalanced_numeric:
        print(f"\nNumerički stupci s neravnomjernom distribucijom:")
        for col, desc in imbalanced_numeric.items():
            print(f"  - {col}: {desc}")
    
    # Provjera neravnomjernih distribucija u kategoričkim stupcima
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    imbalanced_cat = {}
    
    for col in categorical_columns:
        if df[col].nunique() < 10:  # Samo za stupce s razumnim brojem kategorija
            top_val_pct = df[col].value_counts(normalize=True).iloc[0] * 100
            if top_val_pct > 90:
                top_val = df[col].value_counts().index[0]
                imbalanced_cat[col] = f"{top_val_pct:.1f}% '{top_val}'"
    
    if imbalanced_cat:
        print(f"\nKategorički stupci s neravnomjernom distribucijom:")
        for col, desc in imbalanced_cat.items():
            print(f"  - {col}: {desc}")
    
    if not uniform_columns and not imbalanced_numeric and not imbalanced_cat:
        print("Nisu pronađene nelogične distribucije u podacima.")

# Poziv funkcije za prepoznavanje nelogičnih distribucija
prepoznaj_nelogicne_distribucije(df)
