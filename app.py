import os
import pandas as pd
import streamlit as st

# Configuration de la page (Mode large)
st.set_page_config(
    page_title="Cour de Cassation - Analyse des Cabinets", layout="wide"
)


# --- 1. CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    if os.path.exists("data.parquet"):
        return pd.read_parquet("data.parquet")
    elif os.path.exists("data.csv"):
        return pd.read_csv("data.csv")
    else:
        return None


df = load_data()

# Titre principal
st.title("⚖️ Cour de Cassation — Observatoire des Cabinets & Avocats")
st.markdown(
    "Analyse statistique des pourvois et taux de cassation (Période 2021–2025)."
)

if df is None:
    st.warning(
        "⚠️ Aucun fichier de données (`data.parquet` ou `data.csv`) n'a été trouvé dans le dossier du projet. Veuillez y placer votre base nettoyée."
    )
else:
    # --- 2. BARRE LATÉRALE DE FILTRES ---
    st.sidebar.header("🎛️ Filtres d'analyse")

    # Mode d'affichage principal
    mode_affichage = st.sidebar.radio(
        "Mode d'affichage",
        ["Classement Général / Par Chambre", "🔍 Fiche Avocat / Cabinet ciblé"],
    )

    st.sidebar.markdown("---")

    # Filtre Chambre
    chambres_dispos = {
        "Toutes les chambres (Cumul global)": "all",
        "Chambre Criminelle (CR)": "cr",
        "Chambre Sociale (SOC)": "soc",
        "Première Chambre Civile (CIV1)": "civ1",
        "Deuxième Chambre Civile (CIV2)": "civ2",
        "Troisième Chambre Civile (CIV3)": "civ3",
        "Chambre Commerciale (COMM)": "comm",
    }
    choix_chambre_label = st.sidebar.selectbox(
        "Sélectionner la Chambre", list(chambres_dispos.keys())
    )
    choix_chambre_code = chambres_dispos[choix_chambre_label]

    # --- 3. LOGIQUE SELON LE MODE CHOISI ---

    if mode_affichage == "Classement Général / Par Chambre":
        st.sidebar.markdown("---")
        seuil_min = st.sidebar.slider(
            "Seuil minimal de pourvois", min_value=1, max_value=200, value=50
        )
        top_n = st.sidebar.slider("Afficher le Top", 5, 50, 20)

        # Filtrage par chambre
        df_travail = df.copy()
        if choix_chambre_code != "all":
            df_travail = df_travail[
                df_travail["chamber_clean"] == choix_chambre_code
            ]

        # Stats globales du périmètre sélectionné
        tot_p = len(df_travail)
        tot_c = (
            df_travail["is_cassation"].sum() if "is_cassation" in df_travail else 0
        )
        taux_g = (tot_c / tot_p * 100) if tot_p > 0 else 0

        # Affichage KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Pourvois", f"{tot_p:,}".replace(",", " "))
        col2.metric("Total Cassations", f"{tot_c:,}".replace(",", " "))
        col3.metric("Taux Moyen Global", f"{taux_g:.2f}%")

        st.markdown("---")
        st.subheader(
            f"📋 Classement — {choix_chambre_label} (Min. {seuil_min} dossiers)"
        )

        # Agrégation par cabinet
        col_cabinet = "cabinet_unifie" if "cabinet_unifie" in df_travail.columns else "cabinet_clean"
        
        if col_cabinet in df_travail.columns and "is_cassation" in df_travail.columns:
            stats = (
                df_travail.groupby(col_cabinet)
                .agg(
                    Pourvois=("is_cassation", "count"),
                    Cassations=("is_cassation", "sum"),
                )
                .reset_index()
            )
            stats["Taux de Cassation"] = (
                stats["Cassations"] / stats["Pourvois"]
            ) * 100

            # Application du seuil
            stats_filtered = stats[stats["Pourvois"] >= seuil_min].sort_values(
                by=["Taux de Cassation", "Pourvois"], ascending=[False, False]
            )

            # Ajout du rang
            stats_filtered.insert(0, "Rang", range(1, len(stats_filtered) + 1))
            stats_filtered = stats_filtered.head(top_n)

            # Formatage du taux pour l'affichage
            stats_filtered["Taux de Cassation"] = stats_filtered[
                "Taux de Cassation"
            ].apply(lambda x: f"{x:.2f}%")

            st.dataframe(
                stats_filtered.set_index("Rang"), use_container_width=True
            )
        else:
            st.error(
                "Les colonnes requises sont introuvables dans votre DataFrame."
            )

    else:
        # MODE FICHE AVOCAT CIBLÉ (Recherche multi-résultats)
        st.subheader("🔍 Analyse détaillée par Avocat / Cabinet")
        
        recherche = st.sidebar.text_input(
            "Rechercher un mot-clé (ex: BOUTHORS, BORE, GATINEAU...)", value="BOUTHORS"
        )

        col_cabinet_recherche = "cabinet_clean" if "cabinet_clean" in df.columns else "cabinet_unifie"

        if recherche and col_cabinet_recherche in df.columns:
            # 1. Filtrer les lignes contenant le mot-clé
            df_recherche = df[
                df[col_cabinet_recherche].str.contains(recherche, case=False, na=False)
            ].copy()

            if not df_recherche.empty:
                # 2. Récupérer la liste de TOUS les noms complets distincts trouvés
                noms_complets_trouves = sorted(df_recherche[col_cabinet_recherche].dropna().unique())

                st.info(f"🔎 **{len(noms_complets_trouves)}** nom(s) complet(s) trouvé(s) pour le mot-clé `{recherche}`.")

                # 3. Sélecteur pour choisir le nom exact parmi ceux trouvés
                cabinet_selectionne = st.selectbox(
                    "Sélectionner le nom exact à analyser :",
                    noms_complets_trouves
                )

                # 4. Filtrer les données pour le nom exact sélectionné
                df_avocat = df_recherche[df_recherche[col_cabinet_recherche] == cabinet_selectionne]

                tot_p_av = len(df_avocat)
                tot_c_av = df_avocat["is_cassation"].sum() if "is_cassation" in df_avocat else 0
                taux_av = (tot_c_av / tot_p_av * 100) if tot_p_av > 0 else 0

                st.markdown(f"### 📋 Bilan cumulé pour : `{cabinet_selectionne}`")

                # Affichage des KPIs
                col1, col2, col3 = st.columns(3)
                col1.metric("Pourvois Totaux", tot_p_av)
                col2.metric("Cassations Obtenues", tot_c_av)
                col3.metric("Taux Global", f"{taux_av:.2f}%")

                st.markdown("---")
                st.markdown("#### 📊 Ventilation par Chambre")

                if tot_p_av > 0 and "chamber_clean" in df_avocat.columns:
                    stats_ch = (
                        df_avocat.groupby("chamber_clean")
                        .agg(
                            Pourvois=("is_cassation", "count"),
                            Cassations=("is_cassation", "sum"),
                        )
                        .reset_index()
                    )
                    stats_ch["Taux de Cassation"] = (
                        stats_ch["Cassations"] / stats_ch["Pourvois"]
                    ) * 100
                    stats_ch = stats_ch.sort_values(
                        by=["Pourvois", "Taux de Cassation"], ascending=[False, False]
                    )
                    stats_ch["Taux de Cassation"] = stats_ch["Taux de Cassation"].apply(
                        lambda x: f"{x:.2f}%"
                    )

                    st.dataframe(stats_ch, use_container_width=True)
            else:
                st.warning("Aucun résultat trouvé pour cette recherche.")