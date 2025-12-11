"""
Script pour analyser la structure du dataset Cresci-2017
"""
import os
import csv
from pathlib import Path

def count_users_in_dataset(dataset_path):
    """Compte le nombre d'utilisateurs dans un dataset"""
    users_file = dataset_path / "users.csv"
    if users_file.exists():
        with open(users_file, 'r', encoding='utf-8') as f:
            return sum(1 for line in f) - 1  # -1 pour enlever l'en-tête
    return 0

def analyze_cresci_dataset():
    """Analyse tous les datasets Cresci-2017"""
    base_path = Path("datasets_full.csv")

    datasets = {
        "Genuine Accounts (Humains)": "genuine_accounts.csv",
        "Fake Followers": "fake_followers.csv",
        "Social Spambots 1": "social_spambots_1.csv",
        "Social Spambots 2": "social_spambots_2.csv",
        "Social Spambots 3": "social_spambots_3.csv",
        "Traditional Spambots 1": "traditional_spambots_1.csv",
        "Traditional Spambots 2": "traditional_spambots_2.csv",
        "Traditional Spambots 3": "traditional_spambots_3.csv",
        "Traditional Spambots 4": "traditional_spambots_4.csv",
    }

    print("=" * 80)
    print("📊 ANALYSE DU DATASET CRESCI-2017")
    print("=" * 80)
    print()

    total_bots = 0
    total_humans = 0

    for name, folder in datasets.items():
        # Essayer d'abord la structure simple (folder/)
        dataset_path = base_path / folder
        if not (dataset_path / "users.csv").exists():
            # Sinon essayer la structure double (folder/folder/)
            dataset_path = base_path / folder / folder

        if dataset_path.exists() and (dataset_path / "users.csv").exists():
            count = count_users_in_dataset(dataset_path)
            print(f"✓ {name:30} : {count:6} comptes")

            if "Genuine" in name:
                total_humans += count
            else:
                total_bots += count
        else:
            print(f"✗ {name:30} : Non extrait")

    print()
    print("=" * 80)
    print(f"TOTAL HUMAINS : {total_humans:6} comptes")
    print(f"TOTAL BOTS    : {total_bots:6} comptes")
    print(f"TOTAL GÉNÉRAL : {total_humans + total_bots:6} comptes")
    print("=" * 80)
    print()

    # Afficher les features disponibles
    print("🔧 FEATURES DISPONIBLES DANS users.csv")
    print("=" * 80)

    # Chercher le premier fichier users.csv disponible
    sample_path = base_path / "social_spambots_1.csv" / "users.csv"
    if not sample_path.exists():
        sample_path = base_path / "social_spambots_1.csv" / "social_spambots_1.csv" / "users.csv"

    if sample_path.exists():
        with open(sample_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            print("Colonnes disponibles :")
            for i, col in enumerate(headers, 1):
                print(f"  {i:2}. {col}")

    print()

if __name__ == "__main__":
    analyze_cresci_dataset()
