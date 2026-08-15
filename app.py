import os
import re
import io
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Portail IPC - Observatoire de la Cour de Cassation",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MAPPING_CHAMBRES = {
    "cr": "Chambre Criminelle (CR)", 
    "soc": "Chambre Sociale (SOC)",
    "civ1": "1ère Chambre Civile (CIV1)", 
    "civ2": "2ème Chambre Civile (CIV2)",
    "civ3": "3ème Chambre Civile (CIV3)", 
    "comm": "Chambre Commerciale (COMM)",
    "ordo": "Ordonnances (ORDO)",
    "pl": "Assemblée Plénière (PL)",
    "mi": "Chambre Mixte (MI)"
}

@st.cache_data
def load_data():
    DATA_URL = "https://github.com/laurentsbert-sketch/observatoire-cassation/releases/download/v1.0.0/data_ipc.parquet"
    
    try:
        req = urllib.request.Request(DATA_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            parquet_bytes = io.BytesIO(response.read())
            
        df = pd.read_parquet(parquet_bytes)
        
        if "decision_date" in df.columns:
            df["decision_date"] = pd.to_datetime(df["decision_date"], errors='coerce')
            df["annee"] = df["decision_date"].dt.year
            
        if "office_officiel_2026" not in df.columns:
            df["office_officiel_2026"] = df.get("demandeur_avocat", "Non renseigné")
        if "demandeur_avocat_raw" not in df.columns:
            df["demandeur_avocat_raw"] = df.get("demandeur_avocat", "Non renseigné")
            
        return df
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données depuis la Release GitHub : {e}")
        st.info("💡 Vérifiez que le tag 'v1.0.0' et le fichier 'data_ipc.parquet' existent bien dans vos Releases GitHub.")
        return None

def compute_ipc_scores(df_subset, col_entity, w_act=0.20, w_perf=0.45, w_reg=0.15, w_spec=0.20):
    df_clean = df_subset[~df_subset[col_entity].isin(["Non renseigné", "Office Non Identifié / Hors Ordre"])].copy()
    if df_clean.empty:
        return pd.DataFrame()

    df_fond_global = df_subset[df_subset["chamber"] != "ordo"]
    
    ch_tranche = df_fond_global.groupby("chamber").agg(
        C_tot=("is_cassation", "sum"),
        R_tot=("is_rejet", "sum")
    )
    ch_tranche["T_ch_avg"] = ch_tranche["C_tot"] / (ch_tranche["C_tot"] + ch_tranche["R_tot"])
    avg_ch_map = ch_tranche["T_ch_avg"].to_dict()

    records = []
    max_N = df_clean.groupby(col_entity).size().max() or 1

    for entity, group in df_clean.groupby(col_entity):
        N_c = len(group)
        group_fond = group[group["chamber"] != "ordo"]
        N_fond = len(group_fond)
        
        cass = group["is_cassation"].sum()
        rejet = group["is_rejet"].sum()
        tranches = cass + rejet
        
        I_act = round(min(100.0, 100.0 * (np.log(1 + N_c) / np.log(1 + max_N))), 1)

        if tranches > 0:
            delta_sum = 0.0
            for ch, ch_group in group_fond.groupby("chamber"):
                c_ch = ch_group["is_cassation"].sum()
                r_ch = ch_group["is_rejet"].sum()
                t_ch = c_ch + r_ch
                if t_ch > 0:
                    rate_c_ch = c_ch / t_ch
                    avg_ref = avg_ch_map.get(ch, 0.25)
                    delta_sum += (len(ch_group) / max(1, N_fond)) * (rate_c_ch - avg_ref)
            I_perf = round(float(np.clip(50.0 + 200.0 * delta_sum, 0.0, 100.0)), 1)
        else:
            I_perf = 0.0

        yearly_rates = []
        for yr, yr_group in group_fond.groupby("annee"):
            c_y = yr_group["is_cassation"].sum()
            r_y = yr_group["is_rejet"].sum()
            if (c_y + r_y) > 0:
                yearly_rates.append(c_y / (c_y + r_y))
                
        if len(yearly_rates) > 1:
            std_y = np.std(yearly_rates)
            mean_y = np.mean(yearly_rates)
            I_reg = round(float(np.clip(100.0 * (1.0 - (std_y / (mean_y + 0.1))), 0.0, 100.0)), 1)
        else:
            I_reg = 0.0 if tranches == 0 else 50.0

        ch_counts = group_fond["chamber"].value_counts(normalize=True)
        if not ch_counts.empty:
            hhi = np.sum(ch_counts**2)
            K = 6.0
            I_spec = round(float(np.clip(100.0 * (hhi - (1.0 / K)) / (1.0 - (1.0 / K)), 0.0, 100.0)), 1)
            top_ch = ch_counts.index[0]
            top_ch_name = MAPPING_CHAMBRES.get(top_ch, top_ch)
        else:
            I_spec = 0.0
            top_ch_name = "Ordonnances Exclusives"

        score_ipc = round(w_act * I_act + w_perf * I_perf + w_reg * I_reg + w_spec * I_spec, 1)
        
        tx_fond = round((cass / tranches) * 100, 1) if tranches > 0 else 0.0
        marge_err = round(1.96 * np.sqrt((tx_fond/100 * (1 - tx_fond/100)) / max(1, tranches)) * 100, 1) if tranches > 0 else 0.0

        records.append({
            col_entity: entity,
            "Score IPC Global": score_ipc,
            "Total Pourvois": N_c,
            "Tranchés au Fond": tranches,
            "Cassations": cass,
            "Rejets": rejet,
            "Taux au Fond (%)": tx_fond,
            "Marge Erreur (%)": marge_err,
            "Indice Activité": I_act,
            "Indice Performance": I_perf,
            "Indice Régularité": I_reg,
            "Indice Spécialisation": I_spec,
            "Chambre Dominante": top_ch_name
        })

    return pd.DataFrame(records)

if 'page' not in st.session_state: st.session_state.page = "global"
if 'selected_cabinet' not in st.session_state: st.session_state.selected_cabinet = None
if 'selected_row' not in st.session_state: st.session_state.selected_row = None

df = load_data()

st.title("⚖️ Indice de Performance de Cassation (IPC)")
st.caption("Analyse décisionnelle et transparente des avocats aux Conseils devant la Cour de cassation (2021 - 2026)")

if df is not None:
    col_date = "decision_date"

    st.sidebar.header("🎛️ Filtres Temporels & Périmètre")
    
    exclure_ordo = st.sidebar.checkbox("🚫 Exclure la chambre Ordonnances (`ordo`)", value=False)
    if exclure_ordo:
        df = df[df["chamber"] != "ordo"]

    if col_date in df.columns and not df[col_date].isna().all():
        type_filtre_date = st.sidebar.radio("Filtrage temporel", ["Plage de dates (Précis)", "Par Années"])
        if type_filtre_date == "Plage de dates (Précis)":
            min_d, max_d = df[col_date].min().date(), df[col_date].max().date()
            dates = st.sidebar.date_input("Période exacte", value=(min_d, max_d), min_value=min_d, max_value=max_d)
            if isinstance(dates, tuple) and len(dates) == 2:
                df = df[(df[col_date].dt.date >= dates[0]) & (df[col_date].dt.date <= dates[1])]
        else:
            annees_dispos = sorted([int(y) for y in df["annee"].dropna().unique()])
            annees_sel = st.sidebar.multiselect("Années", annees_dispos, default=annees_dispos)
            if annees_sel:
                df = df[df["annee"].isin(annees_sel)]

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Pondérations de l'Indice IPC")
    w_perf = st.sidebar.slider("Performance Contextualisée", 0.0, 1.0, 0.45, 0.05)
    w_act = st.sidebar.slider("Activité / Expérience", 0.0, 1.0, 0.20, 0.05)
    w_spec = st.sidebar.slider("Spécialisation (HHI)", 0.0, 1.0, 0.20, 0.05)
    w_reg = st.sidebar.slider("Régularité", 0.0, 1.0, 0.15, 0.05)
    
    w_sum = w_perf + w_act + w_spec + w_reg or 1.0
    w_perf, w_act, w_spec, w_reg = w_perf/w_sum, w_act/w_sum, w_spec/w_sum, w_reg/w_sum

    if st.session_state.page == "global":
        
        tab_norm, tab_raw, tab_dem, tab_ch = st.tabs([
            "🏆 Offices Ordinaux (Normés IPC 2026)", 
            "🔍 Données Brutes Extraintes", 
            "🏛️ Top Demandeurs Institutionnels", 
            "📐 Benchmarks Chambres"
        ])

        with tab_norm:
            st.subheader("🏆 Classement des Offices d'Avocats aux Conseils (Annuaire Ordinal 2026)")
            seuil_min = st.slider("Seuil minimum de pourvois", 1, 200, 15, key="slider_norm")
            
            df_ipc = compute_ipc_scores(df, "office_officiel_2026", w_act, w_perf, w_reg, w_spec)
            
            if not df_ipc.empty:
                df_ipc_filt = df_ipc[df_ipc["Total Pourvois"] >= seuil_min].sort_values("Score IPC Global", ascending=False)
                
                st.write("💡 *Cliquez sur une ligne pour ouvrir la fiche synthétique du cabinet :*")
                event = st.dataframe(df_ipc_filt, width="stretch", selection_mode="single-row", on_select="rerun", height=450)
                
                if event.selection["rows"]:
                    idx = event.selection["rows"][0]
                    st.session_state.selected_cabinet = df_ipc_filt.iloc[idx]["office_officiel_2026"]
                    st.session_state.page = "cabinet"
                    st.rerun()

        with tab_raw:
            st.subheader("🔍 Audit & Visualisation des Données Brutes Extraintes")
            seuil_raw = st.slider("Seuil minimum de pourvois", 1, 200, 15, key="slider_raw")
            
            df_raw = compute_ipc_scores(df, "demandeur_avocat_raw", w_act, w_perf, w_reg, w_spec)
            if not df_raw.empty:
                df_raw_filt = df_raw[df_raw["Total Pourvois"] >= seuil_raw].sort_values("Total Pourvois", ascending=False)
                st.dataframe(df_raw_filt, width="stretch", height=450)

        with tab_dem:
            st.subheader("🏛️ Demandeurs Récurrents Non Anonymisés")
            mask_real_dem = (
                (df["demandeur"] != "Non renseigné") &
                (~df["demandeur"].astype(str).str.contains(r'\[.*?\]', regex=True, na=False)) &
                (df["demandeur"].astype(str).str.strip() != "")
            )
            df_dem = df[mask_real_dem]
            seuil_dem = st.slider("Seuil minimum de pourvois", 1, 100, 10, key="slider_dem")
            
            stats_dem = df_dem.groupby("demandeur").agg(
                Total_Pourvois=("is_cassation", "count"),
                Cassations=("is_cassation", "sum"),
                Rejets=("is_rejet", "sum")
            )
            stats_dem["Taux au Fond (%)"] = round((stats_dem["Cassations"] / (stats_dem["Cassations"] + stats_dem["Rejets"])) * 100, 1)
            stats_dem = stats_dem[stats_dem["Total_Pourvois"] >= seuil_dem].sort_values("Total_Pourvois", ascending=False)
            st.dataframe(stats_dem, width="stretch", height=450)

        with tab_ch:
            st.subheader("📐 Taux Moyen de Cassation par Chambre")
            if "chamber" in df.columns:
                ch_summary = df.groupby("chamber").agg(
                    Total=("id", "count"),
                    Cassations=("is_cassation", "sum"),
                    Rejets=("is_rejet", "sum")
                )
                ch_summary["Taux au Fond (%)"] = round((ch_summary["Cassations"] / (ch_summary["Cassations"] + ch_summary["Rejets"])) * 100, 1)
                ch_summary.index = ch_summary.index.map(lambda x: MAPPING_CHAMBRES.get(x, x))
                st.dataframe(ch_summary.sort_values("Total", ascending=False), width="stretch", height=450)

    elif st.session_state.page == "cabinet":
        cab = st.session_state.selected_cabinet
        
        col_back, col_title = st.columns([1, 5])
        with col_back:
            if st.button("⬅️ Retour"):
                st.session_state.page = "global"
                st.rerun()
        with col_title:
            st.subheader(f"📋 Fiche Synthétique IPC : {cab}")
        
        df_cab = df[df["office_officiel_2026"] == cab].copy()
        df_cab_score = compute_ipc_scores(df_cab, "office_officiel_2026", w_act, w_perf, w_reg, w_spec)
        
        if not df_cab_score.empty:
            row_s = df_cab_score.iloc[0]
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Indice IPC Global", f"{row_s['Score IPC Global']} / 100")
            c2.metric("Indice de Performance", f"{row_s['Indice Performance']} / 100")
            c3.metric("Indice Spécialisation (HHI)", f"{row_s['Indice Spécialisation']} / 100")
            c4.metric("Chambre Dominante", row_s['Chambre Dominante'])

            st.markdown("---")
            
            st.markdown("### 📊 Sous-Indices de Performance Contextualisée")
            col_i1, col_i2, col_i3, col_i4 = st.columns(4)
            col_i1.progress(row_s['Indice Performance'] / 100.0, text=f"Performance: {row_s['Indice Performance']}")
            col_i2.progress(row_s['Indice Activité'] / 100.0, text=f"Activité: {row_s['Indice Activité']}")
            col_i3.progress(row_s['Indice Spécialisation'] / 100.0, text=f"Spécialisation: {row_s['Indice Spécialisation']}")
            col_i4.progress(row_s['Indice Régularité'] / 100.0, text=f"Régularité: {row_s['Indice Régularité']}")

            st.markdown("---")
            st.markdown("### 📜 Liste des Décisions de l'Office")
            
            cols_display = ["decision_date", "numero_affaire", "demandeur", "chamber", "solution_raw"]
            cols_present = [c for c in cols_display if c in df_cab.columns]
            
            event_dec = st.dataframe(
                df_cab[cols_present].sort_values("decision_date", ascending=False), 
                width="stretch", 
                selection_mode="single-row", 
                on_select="rerun",
                height=400
            )
            
            if event_dec.selection["rows"]:
                idx = event_dec.selection["rows"][0]
                st.session_state.selected_row = df_cab.iloc[idx]
                st.session_state.page = "decision"
                st.rerun()

    elif st.session_state.page == "decision":
        col_back, col_title = st.columns([1, 5])
        with col_back:
            if st.button("⬅️ Retour"):
                st.session_state.page = "cabinet"
                st.rerun()
        
        row = st.session_state.selected_row
        
        with col_title:
            st.subheader(f"📄 Pourvoi N° : {row.get('numero_affaire', 'Inconnu')}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Date", str(row.get("decision_date", ""))[:10])
        c2.metric("Chambre", MAPPING_CHAMBRES.get(row.get("chamber"), row.get("chamber")))
        c3.metric("Demandeur", str(row.get("demandeur", "Non renseigné")))
        c4.metric("Issue", str(row.get("solution_raw", "")))

        st.markdown("---")
        
        if row.get("summary") and str(row.get("summary")) not in ["None", ""]:
            st.markdown("### 📌 Résumé / Moyens")
            st.info(row["summary"])

        st.markdown("### 📜 Texte intégral de l'Arrêt")
        st.text_area(
            "Texte intégral",
            value=str(row.get("texte_integral", "Texte non disponible")),
            height=500,
            disabled=True,
            label_visibility="collapsed"
        )
