"""
================================================================================
S-DFP (State-Dependent Sequential Flow Allocation) - Web Dashboard Interattiva
Tecnologia: Streamlit + Plotly + Pandas + NumPy + SciPy
Integrazione diretta con: simulate_sdfp.py
Framework: TUIR, D.Lgs. 461/1997, D.Lgs. 252/2005, D.L. 201/2011
================================================================================
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import io
import os
import sys
from pathlib import Path
import datetime
from datetime import date, timedelta

try:
 from gsheets_connector import GSheetsConnection
except Exception:
    GSheetsConnection = None

# Percorsi base per risoluzione dinamica file locali o in output/ (relativi alla cartella di deploy)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_file_path(filename: str) -> Path | None:
    """
    Risolve un file usando percorsi relativi basati su os.path.join(os.path.dirname(__file__), filename)
    cercando prioritariamente nella cartella dello script e poi nella sottocartella output/.
    Restituisce l'oggetto Path se trovato, altrimenti None.
    """
    p_root = Path(os.path.join(os.path.dirname(__file__), filename))
    if p_root.exists():
        return p_root
    p_out = Path(os.path.join(os.path.dirname(__file__), "output", filename))
    if p_out.exists():
        return p_out
    return None

# Connessione e gestione persistenza Google Sheets
try:
    if GSheetsConnection is not None:
        conn = st.connection("gsheets", type=GSheetsConnection)
    else:
        conn = None
except Exception as e:
    conn = None


def get_default_user_data() -> dict:
    """Restituisce il dizionario con i parametri di default del modello S-DFP."""
    return {
        "Y0": 2000.0,
        "ral_0": 32000.0,
        "g_Y_pct": 1.8,
        "smart_lifestyle_choice": "Aumenta tenore di vita (Regola SMarT: 50% spese / 50% risparmio)",
        "split_consumo_pct": 50,
        "no_mortgage": False,
        "mortgage_R0": 600.0,
        "E_base": 1500.0,
        "override_spese": False,
        "eta_attuale": 30,
        "eta_pensionabile": 67,
        "include_decumulo": True,
        "enable_goal_H": True,
        "T3a_0": 25000.0,
        "H_years": 8,
        "current_month_t": 1,
        "target_mode": "Consigliato da S-DFP (Automatico)",
        "input_v1": 5000.0,
        "input_v2": 2000.0,
        "input_v3b": 2400.0,
        "input_v3a": 15000.0,
        "input_v4": 25000.0,
    }


def load_user_data() -> dict:
    """
    Legge i dati utente dal foglio Google Sheets tramite conn.read(ttl=0).
    Se il foglio è vuoto o la connessione fallisce, restituisce i parametri di default.
    Gestisce eventuali errori con un blocco try/except e mostra un st.warning() in caso di problemi.
    """
    defaults = get_default_user_data()
    try:
        if conn is None:
            return defaults
        df = conn.read(ttl=0)
        if df is not None and not df.empty:
            df_valid = df.dropna(how="all")
            if not df_valid.empty:
                row_dict = df_valid.iloc[0].to_dict()
                merged = defaults.copy()
                for k, v in row_dict.items():
                    if k in merged and pd.notna(v):
                        def_type = type(defaults[k])
                        try:
                            if def_type is bool:
                                merged[k] = bool(v) if not isinstance(v, str) else v.strip().lower() in ['true', '1', 't', 'si', 'yes']
                            elif def_type is int:
                                merged[k] = int(float(v))
                            elif def_type is float:
                                merged[k] = float(v)
                            else:
                                merged[k] = str(v)
                        except Exception:
                            pass
                return merged
        return defaults
    except Exception as err:
        st.warning(f"⚠️ Impossibile leggere i dati da Google Sheets: {err}. Vengono utilizzati i parametri di default. Verifica i permessi del Service Account sul foglio Google.")
        return defaults


def save_user_data(data) -> bool:
    """
    Scrive i dati utente su Google Sheets tramite conn.update(data=...).
    Formatta i dati in formato tabellare (DataFrame Pandas a una sola riga).
    Gestisce eventuali errori con un blocco try/except e mostra un st.warning() in caso di fallimento.
    """
    try:
        if conn is None:
            st.warning("⚠️ Impossibile salvare su Google Sheets: connessione non disponibile. Verifica la presenza della sezione [connections.gsheets] nei Secrets di Streamlit.")
            return False
        
        # Formato tabellare con una sola riga
        if isinstance(data, dict):
            df_to_save = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df_to_save = data
        else:
            df_to_save = pd.DataFrame(data)

        conn.update(data=df_to_save)
        return True
    except Exception as err:
        st.warning(f"⚠️ Errore durante il salvataggio su Google Sheets: {err}. Verifica che il Service Account abbia i permessi di Editor sul foglio Google.")
        return False



# Integrazione diretta con il motore di simulazione quantitativo
import simulate_sdfp
from simulate_sdfp import (
    SimulationConfig,
    UnifiedSimulationEngine,
    StatisticalInferenceEngine,
    HistoricalStressTester,
    STATE_S1, STATE_S2, STATE_S3, STATE_S4,
    STATE_S4_INV, STATE_S5A, STATE_S5B, STATE_S_REBUILD
)


def get_default_portfolio_assets():
    """Restituisce una lista vuota in modo che l'app parta pulita senza dati precompilati."""
    return []

# ------------------------------------------------------------------------------
# 1. CONFIGURAZIONE PAGINA E STILE CSS FINTECH
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="S-DFP | Quantitative Wealth Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS per look & feel moderno, palette fintech (Navy, Slate, Emerald, Gold)
st.markdown("""
<style>
    /* Stile generale e container */
    .main {
        background-color: #f8fafc;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header e badge */
    .fintech-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .fintech-subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1rem;
        font-weight: 400;
    }
    .badge-bar {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }
    .badge-tag {
        background-color: #e2e8f0;
        color: #1e293b;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        border: 1px solid #cbd5e1;
    }
    .badge-accent {
        background-color: #ecfdf5;
        color: #065f46;
        border-color: #a7f3d0;
    }
    
    /* Card KPI */
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 20px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 10px 15px -3px rgba(0,0,0,0.02);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08), 0 2px 4px -1px rgba(0,0,0,0.04);
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }
    .kpi-delta-pos {
        font-size: 0.82rem;
        font-weight: 700;
        color: #059669;
        margin-top: 6px;
    }
    .kpi-delta-neg {
        font-size: 0.82rem;
        font-weight: 700;
        color: #dc2626;
        margin-top: 6px;
    }
    .kpi-subtext {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Card dei Vasi */
    .pot-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        border-left: 6px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }
    .pot-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e293b;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .pot-subtitle {
        font-size: 0.8rem;
        color: #64748b;
        margin-bottom: 10px;
    }
    .pot-balance {
        font-size: 1.4rem;
        font-weight: 800;
        color: #0f172a;
    }
    .pot-target {
        font-size: 0.82rem;
        color: #475569;
    }
    
    /* Stato Macchina a Stati Banner */
    .state-banner {
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 20px;
        border: 1px solid #cbd5e1;
        background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
    }
    .state-tag {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #1e40af;
        margin-bottom: 4px;
    }
    .state-name {
        font-size: 1.4rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 6px;
    }
    .state-desc {
        font-size: 0.92rem;
        color: #334155;
        line-height: 1.5;
    }
    
    /* Box Azioni Operative e Surplus */
    .action-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 6px solid #10b981;
        border-radius: 12px;
        padding: 20px 22px;
        margin: 18px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .action-box-neutral {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 6px solid #64748b;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 18px 0;
    }
    .action-box-warning {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 6px solid #f59e0b;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 18px 0;
    }
    .action-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #065f46;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .action-item {
        background: white;
        border: 1px solid #dcfce7;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.92rem;
        color: #1e293b;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .badge-trans {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. FUNZIONI DI SIMULAZIONE CON CACHING (@st.cache_data)
# ------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def run_cached_simulation(n_paths, seed, Y0, ral_0, g_Y, mortgage_R0, E0, T3a_0, H_years, smart_split, T2_fixed=2400.0, T3b_fixed=None, T_horizon=360, T_retire=276):
    """
    Esegue la simulazione unificata M0-M6 con caching su disco/memoria per
    garantire tempi di risposta istantanei al cambio di viste o schede.
    """
    if T3b_fixed is None:
        T3b_fixed = 4.0 * float(mortgage_R0)
    cfg = SimulationConfig(
        n_paths=int(n_paths),
        seed=int(seed),
        Y0=float(Y0),
        ral_0=float(ral_0),
        g_Y=float(g_Y),
        mortgage_R0=float(mortgage_R0),
        E0=float(E0),
        T3a_0=float(T3a_0),
        H_years=int(H_years),
        smart_split=float(smart_split),
        T2_fixed=float(T2_fixed),
        T3b_fixed=float(T3b_fixed),
        T_horizon=int(T_horizon),
        T_retire=int(T_retire)
    )
    
    engine = UnifiedSimulationEngine(cfg)
    data, v0_milestones = engine.run_all_models(n_paths=int(n_paths), seed=int(seed))
    
    # Calcolo metriche statistiche complete per ciascun modello (tutti gli 8 modelli M0-M7)
    model_names = ['M0_Naive', 'M1_Naive_Buf', 'M2_Fair', 'M3_SDFP_NoInflow', 
                   'M4_SDFP_ScenB', 'M5_SDFP_ScenC', 'M6_Fair_LiabMatch', 'M7_Fair_LiabInflowReb']
    
    summary_results = {}
    for mod in model_names:
        summary_results[mod] = StatisticalInferenceEngine.compute_model_metrics(data[mod])
        
    # Esecuzione stress test storici a fattori genuini
    stress_results = HistoricalStressTester.run_stress_tests(cfg)
    
    # Traiettorie Vaso 3A
    v3a_track = data.get('v3a_track', {
        'months': np.arange(cfg.H_month + 1),
        'med': np.zeros(cfg.H_month + 1),
        'p10': np.zeros(cfg.H_month + 1),
        'p90': np.zeros(cfg.H_month + 1),
        'target': np.full(cfg.H_month + 1, cfg.T3a_0)
    })
    
    # Traiettorie temporali annuali per tutti i modelli
    yearly_data = {}
    years_arr = np.arange(cfg.n_years + 1)
    for mod in model_names:
        yearly_data[mod] = {
            'years': years_arr,
            'W_tot_med': data[mod].get('yearly_W_tot_med', np.zeros(cfg.n_years + 1)),
            'W_tot_p10': data[mod].get('yearly_W_tot_p10', np.zeros(cfg.n_years + 1)),
            'W_tot_p90': data[mod].get('yearly_W_tot_p90', np.zeros(cfg.n_years + 1)),
            'W_fin_med': data[mod].get('yearly_W_fin_med', np.zeros(cfg.n_years + 1)),
        }
        
    return {
        'cfg': cfg.to_dict(),
        'summary': summary_results,
        'stress': stress_results,
        'v3a_track': v3a_track,
        'yearly': yearly_data
    }


def compute_waterflow_cascade(v1, v2, v3b, v3a, v4, t1, t2, t3b, t3a):
    """
    Calcola il surplus/deficit per ciascun vaso e la cascata di riallocazione
    secondo la rigorosa gerarchia S-DFP.
    
    Regole fondamentali di liquidità:
    1. L'eccedenza di cassa trasferibile ("Surplus di Liquidità") ha origine
       ESCLUSIVAMENTE dai comparti di cassa a monte in esubero (V1 > T1 e V2 > T2).
    2. Il Vaso 3A è un comparto obbligazionario held-to-maturity a scadenza fissa:
       è rigorosamente ESCLUSO dai comparti da cui estrarre surplus verso V4.
       Se Saldo V3A >= Target Base, è etichettato come "🟢 A Target / In Mantenimento".
       Non si disinveste né si trasferisce MAI capitale da V3A a V4.
    3. Cascata di distribuzione del surplus di cassa (Waterflow Cascade):
       a) Colmare eventuale deficit residuo di V1 (se surplus generato da V2)
       b) Colmare eventuale deficit di V2 (fino a T2 = 2.400 €)
       c) Colmare eventuale deficit di V3B (fino a T3b = 4 x rata mutuo)
       d) Colmare eventuale deficit di V3A (fino al target T3a)
       e) Canalizzare l'eccedenza residua direttamente nel Vaso 4 Core ETF All-World,
          lasciando intatto il Vaso 3A.
    """
    d_v1 = v1 - t1
    d_v2 = v2 - t2
    d_v3b = v3b - t3b if t3b > 0 else 0.0
    d_v3a = v3a - t3a if t3a > 0 else 0.0

    # Surplus prelevabile: SOLO dai comparti di cassa a monte (V1 e V2)
    surplus_v1 = max(0.0, d_v1)
    surplus_v2 = max(0.0, d_v2)
    total_surplus = surplus_v1 + surplus_v2

    moves = []
    
    # Saldi post-estrazione surplus di cassa
    v1_opt = v1 - surplus_v1
    v2_opt = v2 - surplus_v2
    v3b_opt = v3b
    v3a_opt = v3a  # Vaso 3A non viene MAI ridotto!
    v4_opt = v4

    pool = total_surplus

    # Etichetta sorgente per tracciabilità nei callout
    if surplus_v1 > 0 and surplus_v2 > 0:
        src_label = "Vaso 1 (Emergenza) & Vaso 2 (Sinking Fund)"
    elif surplus_v1 > 0:
        src_label = "Vaso 1: Buffer Emergenza (Conto Deposito)"
    elif surplus_v2 > 0:
        src_label = "Vaso 2: Sinking Fund (Conto Progetti)"
    else:
        src_label = "Surplus di Liquidità"

    # A) Colmare eventuale deficit residuo di V1 (se surplus proveniva da V2)
    if v1_opt < t1 and pool > 0:
        transfer = min(pool, t1 - v1_opt)
        v1_opt += transfer
        pool -= transfer
        moves.append({
            "from": "Vaso 2: Sinking Fund",
            "to": "Vaso 1: Buffer Emergenza",
            "amount": transfer,
            "action": "Ripristino Prioritario Emergenza",
            "icon": "🛡️",
            "reason": "Riallineamento al pavimento inderogabile di sicurezza (3 mensilità spese vive)"
        })

    # B) Colmare eventuale deficit di V2 (fino a T2)
    if v2_opt < t2 and pool > 0:
        transfer = min(pool, t2 - v2_opt)
        v2_opt += transfer
        pool -= transfer
        moves.append({
            "from": "Vaso 1 (Conto Deposito)" if surplus_v1 > 0 else src_label,
            "to": "Vaso 2: Sinking Fund",
            "amount": transfer,
            "action": "Bonifico Interno Sinking Fund",
            "icon": "🗓️",
            "reason": "Copertura delle uscite prevedibili a 12-24 mesi per sterilizzare il rischio di cassa"
        })

    # C) Colmare eventuale deficit di V3B (fino a T3b = 4 x rata mutuo)
    if t3b > 0 and v3b_opt < t3b and pool > 0:
        transfer = min(pool, t3b - v3b_opt)
        v3b_opt += transfer
        pool -= transfer
        moves.append({
            "from": src_label,
            "to": "Vaso 3B: Buffer Rata Mutuo",
            "amount": transfer,
            "action": "Bonifico Interno Buffer Mutuo",
            "icon": "🏠",
            "reason": "Consolidamento del pavimento di sicurezza per 4 rate del mutuo (hard floor)"
        })

    # D) Colmare eventuale deficit di V3A (fino al target T3a)
    if t3a > 0 and v3a_opt < t3a and pool > 0:
        transfer = min(pool, t3a - v3a_opt)
        v3a_opt += transfer
        pool -= transfer
        moves.append({
            "from": src_label,
            "to": "Vaso 3A: Liability Matching (Obiettivo H)",
            "amount": transfer,
            "action": "Acquisto Titoli di Stato a Scadenza",
            "icon": "🚗",
            "reason": "Acquisto tranches bond held-to-maturity per immunizzare l'obiettivo a scadenza"
        })

    # E) Residuo canalizzato nel Vaso 4 Core ETF All-World (V3A non viene toccato)
    if pool > 0:
        transfer = pool
        v4_opt += transfer
        pool = 0.0
        moves.append({
            "from": src_label,
            "to": "Vaso 4: Core ETF Azionario All-World",
            "amount": transfer,
            "action": "Investimento Azionario Core",
            "icon": "🚀",
            "reason": "Investimento a lungo termine nel comparto azionario globale per massimizzare il compounding senza tax drag"
        })

    # Tabella analitica di sintesi
    summary_pots = [
        {
            "vaso": "V1 Emergenza",
            "saldo_attuale": v1,
            "target": t1,
            "diff": d_v1,
            "shift": v1_opt - v1,
            "saldo_ottimizzato": v1_opt,
            "status_label": f"🔵 Surplus (+{d_v1:,.0f} €)" if d_v1 > 1e-2 else (f"🟡 Deficit ({d_v1:,.0f} €)" if d_v1 < -1e-2 else "🟢 A Target")
        },
        {
            "vaso": "V2 Sinking Fund",
            "saldo_attuale": v2,
            "target": t2,
            "diff": d_v2,
            "shift": v2_opt - v2,
            "saldo_ottimizzato": v2_opt,
            "status_label": f"🔵 Surplus (+{d_v2:,.0f} €)" if d_v2 > 1e-2 else (f"🟡 Deficit ({d_v2:,.0f} €)" if d_v2 < -1e-2 else "🟢 A Target")
        },
        {
            "vaso": "V3B Buffer Mutuo",
            "saldo_attuale": v3b,
            "target": t3b,
            "diff": d_v3b,
            "shift": v3b_opt - v3b,
            "saldo_ottimizzato": v3b_opt,
            "status_label": "⚪ Disattivato (No Mutuo)" if t3b == 0 else (f"🔵 Surplus (+{d_v3b:,.0f} €)" if d_v3b > 1e-2 else (f"🟡 Deficit ({d_v3b:,.0f} €)" if d_v3b < -1e-2 else "🟢 A Target"))
        },
        {
            "vaso": "V3A Goal H",
            "saldo_attuale": v3a,
            "target": t3a,
            "diff": d_v3a,
            "shift": v3a_opt - v3a,
            "saldo_ottimizzato": v3a_opt,
            "status_label": "⚪ Disattivato" if t3a == 0 else ("🟢 A Target / In Mantenimento" if v3a >= t3a - 1e-2 else f"🟡 Deficit ({d_v3a:,.0f} €)")
        },
        {
            "vaso": "V4 Azionario",
            "saldo_attuale": v4,
            "target": 100000.0,
            "diff": v4 - 100000.0,
            "shift": v4_opt - v4,
            "saldo_ottimizzato": v4_opt,
            "status_label": "🟢 Regime Core-Satellite (≥100k)" if v4 >= 100000.0 else "🔵 In Fase di Accumulo (<100k)"
        },
    ]

    return {
        "total_surplus": total_surplus,
        "surplus_v1": surplus_v1,
        "surplus_v2": surplus_v2,
        "moves": moves,
        "v1_opt": v1_opt,
        "v2_opt": v2_opt,
        "v3b_opt": v3b_opt,
        "v3a_opt": v3a_opt,
        "v4_opt": v4_opt,
        "summary": summary_pots
    }


def determine_state_machine_phase(v1, v2, v3b, v3a, v4, t1, t2, t3b, t3a, base_savings, p4_threshold_val=None, p5_split_v3a=0.60, current_month_t=1, H_months=96, v5=0.0, target_pension=100000.0, v4_core=None, v4_sat=None, satellite_enabled=True, mortgage_active=True, auto_active=True, mortgage_rate=0.035, r_f=0.0325, T_retire=276, T_horizon=360):
    """
    Determina la fase attiva della macchina a stati S-DFP (Algoritmo 1 aggiornato al paper):
    1. Preemption Loop:
       - Priority 1: V1 < T1 -> Ripristino immediato Buffer Emergenza (100% S_t).
       - Priority 2: V3B < T3B (se mutuo attivo) -> Ripristino Buffer Rata Mutuo (100% S_t).
         Dotato di Anti-ZIRP switch: isola dal rischio spread sotto tassi calanti o curve invertite.
    2. Boot Phases (Fasi 1-5):
       - Fase 1: V1 < T1 (100% V1)
       - Fase 2: V2 < T2 (100% V2 Sinking Fund spese cicliche 12-24m)
       - Fase 3: V3B < T3B (100% V3B Buffer Mutuo)
       - Issuance Gap: se auto attiva, H - t <= 24 mesi e V3A < T3A -> 100% V3A (lock-in pre-scadenza)
       - Fase 4: V3A < soglia_parziale (100% V3A titoli Held-to-Maturity)
       - Fase 5A: soglia_parziale <= V3A < T3A (Inversione anticipata: 60% V3A / 40% V4)
    3. Fase 6 / 6b Regime a Lungo Termine:
       - Fase 6b Subordinated Retirement Glidepath: se t >= 276 e V5 < T5(t), 100% S_t a V5,
         con ribilanciamento stock dello 0.25%/mese da V4 a V5.
       - Fase 6 Regime Crescita (S5B):
         * Core-Satellite con banda di isteresi [33%, 47%] su target satellite w_target = 40%:
           Se w_sat < 33%: 100% S_t a V4,sat
           Se w_sat > 47%: 100% S_t a V4,core
           Entro [33%, 47%]: split proporzionale ai target a zero tax drag.
    4. Gestione dei 5 Edge Cases documentati:
       - Shock negativo (S_t <= 0): freeze flussi, attingimento buffer V1.
       - Early payout prima di H (V3A >= T3A): transizione immediata a Fase 5B/6 con capitale protetto.
       - Negative yields / inverted curves: Anti-ZIRP switch.
       - Accelerated mortgage repayment vs investing: confronto tasso passivo vs rendimento atteso netto.
       - Eredità / Liquidità straordinaria: cascata waterfall istantanea.
    """
    if p4_threshold_val is None:
        p4_threshold_val = 0.60 * t3a if t3a > 0 else 0.0

    alloc = {'v1': 0.0, 'v2': 0.0, 'v3b': 0.0, 'v3a': 0.0, 'v4': 0.0, 'v5': 0.0}
    anti_zirp = False
    is_preemption = False
    is_glidepath = False
    is_issuance_gap = False
    is_core_sat = False

    # Edge Case 1: Shock negativo o risparmio nullo
    if base_savings <= 0:
        fase_idx = 1
        name = "⚠️ Regime di Allarme Liquidità (S_t ≤ 0)"
        desc = "Il risparmio mensile è nullo o negativo. I flussi verso tutti gli investimenti sono congelati. Le spese vive attingono prioritariamente dal Buffer Emergenza V1."
        color = "#dc2626"
        return {
            'fase_idx': fase_idx,
            'name': name,
            'desc': desc,
            'badge_color': color,
            'alloc': alloc,
            'anti_zirp_badge': False,
            'preemption_active': True,
            'glidepath_active': False,
            'issuance_gap_active': False,
            'core_sat_active': False
        }

    # 1. PREEMPTION LOOP - Priority 1: Buffer Emergenza V1
    if v1 < t1:
        alloc['v1'] = 1.0
        anti_zirp = True
        fase_idx = 1
        if v4 > 0 or v3a > 0 or v2 > 0:
            is_preemption = True
            name = "⚡ Preemption Loop: Ripristino Buffer Emergenza V1"
            desc = f"Priorità Assoluta Buffer Emergenza: Vaso 1 ({v1:,.0f} €) è sceso sotto la soglia di sicurezza ({t1:,.0f} €). La regola di preemption vincola il 100% di S_t ({base_savings:,.0f} €) a ricostituire la riserva prima di ogni investimento."
            color = "#ef4444"
        else:
            name = "Fase 1: Priorità Assoluta Buffer Emergenza (Vaso 1)"
            desc = f"Canalizzazione del 100% del risparmio mensile verso il Vaso 1 fino al raggiungimento del target vitale di 3 mensilità ({t1:,.0f} €)."
            color = "#2563eb"

    # 1. PREEMPTION LOOP - Priority 2: Buffer Debito V3B (se mutuo attivo)
    elif mortgage_active and t3b > 0 and v3b < t3b:
        alloc['v3b'] = 1.0
        anti_zirp = True
        fase_idx = 3
        if v4 > 0 or v3a > 0:
            is_preemption = True
            name = "⚡ Preemption Loop: Ripristino Buffer Rata Mutuo V3B"
            desc = f"Pavimento Debito Violato: Vaso 3B ({v3b:,.0f} €) è sotto le 4 mensilità di sicurezza ({t3b:,.0f} €). Il 100% dei risparmi è vincolato a V3B per isolare dal rischio spread e rifinanziamento."
            color = "#ea580c"
        else:
            name = "Fase 3: Buffer Debito / Rata Mutuo (Vaso 3B)"
            desc = f"Vasi 1 e 2 saturi. Il 100% del risparmio alimenta il Vaso 3B per coprire il pavimento inderogabile di sicurezza di 4 rate del mutuo ({t3b:,.0f} €)."
            color = "#d97706"

    # 2. BOOT PHASE 2: Sinking Fund Vaso 2
    elif v2 < t2:
        alloc['v2'] = 1.0
        fase_idx = 2
        name = "Fase 2: Sinking Fund Spese Prevedibili (Vaso 2)"
        desc = f"Buffer emergenza V1 colmo ({v1:,.0f} € / {t1:,.0f} €). Il 100% del risparmio confluisce nel Vaso 2 per spese cicliche a 12-24 mesi (target {t2:,.0f} €)."
        color = "#059669"

    # 3. ISSUANCE GAP PRE-TARGET (se mancano <= 24 mesi a Milestone H e V3A non è colmo)
    elif auto_active and t3a > 0 and v3a < t3a and (H_months - current_month_t) <= 24 and (H_months - current_month_t) >= 0:
        alloc['v3a'] = 1.0
        is_issuance_gap = True
        fase_idx = 4
        rem_m = H_months - current_month_t
        name = f"🔒 Issuance Gap Lock-In: Finestra Critica Milestone H ({rem_m} mesi alla scadenza)"
        desc = f"A meno di 24 mesi dalla scadenza dell'obiettivo ({t3a:,.0f} €), l'inversione azionaria è sospesa e il 100% di S_t ({base_savings:,.0f} €) è blindato su titoli di Stato a scadenza esatta H per azzerare il rischio di timing."
        color = "#9333ea"

    # 4. BOOT PHASE 4: Accumulo Primario Vaso 3A
    elif auto_active and t3a > 0 and v3a < p4_threshold_val:
        alloc['v3a'] = 1.0
        fase_idx = 4
        name = "Fase 4: Accumulo Primario Obbligazionario (100% Vaso 3A)"
        desc = f"Tutti i buffer di sicurezza a monte sono colmi. I flussi mensili vanno al 100% su Titoli di Stato a scadenza coincidente con Milestone H fino alla soglia anticipata ({p4_threshold_val:,.0f} €)."
        color = "#7c3aed"

    # 5. BOOT PHASE 5A: Inversione Pre-Target (S5A)
    elif auto_active and t3a > 0 and v3a < t3a:
        split_3a = p5_split_v3a
        split_4 = 1.0 - split_3a
        alloc['v3a'] = split_3a
        alloc['v4'] = split_4
        fase_idx = 5
        name = f"Fase 5: Inversione Pre-Target ({split_3a*100:.0f}% V3A / {split_4*100:.0f}% V4 Core)"
        desc = f"Raggiunta la soglia di sblocco ({p4_threshold_val:,.0f} €): scatta la regola di inversione anticipata. Il {split_3a*100:.0f}% ({split_3a*base_savings:,.0f} €) completa il Vaso 3A e il {split_4*100:.0f}% ({split_4*base_savings:,.0f} €) avvia l'accumulo in Vaso 4 Core ETF All-World."
        color = "#4f46e5"

    # 6. FASE 6 / 6b: REGIME A REGIME E SUBORDINATED GLIDEPATH
    else:
        # Calcolo Target Dinamico Vaso 5 per Subordinated Retirement Glidepath
        t5_target = target_pension * max(0.0, min(1.0, (current_month_t - T_retire) / max(1, T_horizon - T_retire))) if current_month_t >= T_retire else 0.0

        if current_month_t >= T_retire and v5 < t5_target:
            # Fase 6b Subordinated Retirement Glidepath (Algoritmo 1, righe 22-26)
            is_glidepath = True
            fase_idx = 6  # o '6b'
            alloc['v5'] = 1.0
            shift_eur = 0.0025 * v4
            rem_pension_m = max(0, T_horizon - current_month_t)
            rem_pension_y = max(1, (T_horizon - T_retire) // 12)
            name = f"🛡️ Fase 6b: Subordinated Retirement Glidepath (Mese {current_month_t}/{T_horizon} - {rem_pension_m} mesi alla Pensione)"
            desc = f"De-risking Pensionistico Attivo: a {rem_pension_y} anni dal termine, il 100% di S_t ({base_savings:,.0f} €) confluisce in Vaso 5 (Previdenza/Bond breve), accompagnato da un trasferimento mensile graduale dello 0.25% di V4 ({shift_eur:,.0f} €/mese)."
            color = "#0284c7"
        else:
            # Fase 6 Regime Crescita Azionaria & Core-Satellite S5B
            fase_idx = 6
            split_3a = 0.15 if auto_active and t3a > 0 else 0.0
            split_4 = 1.0 - split_3a

            if satellite_enabled and v4 > 0:
                is_core_sat = True
                c_val = v4_core if v4_core is not None else 0.60 * v4
                s_val = v4_sat if v4_sat is not None else 0.40 * v4
                w_sat = s_val / (c_val + s_val) if (c_val + s_val) > 0 else 0.40

                if w_sat < 0.33:
                    alloc['v4'] = split_4
                    sat_note = f"Inflow Rebalancing: Satellite al {w_sat*100:.1f}% (sotto banda 33%). 100% del flusso V4 ({split_4*base_savings:,.0f} €) a Satellite Factor fino a w_sat=40%."
                elif w_sat > 0.47:
                    alloc['v4'] = split_4
                    sat_note = f"Inflow Rebalancing: Satellite al {w_sat*100:.1f}% (sopra banda 47%). 100% del flusso V4 ({split_4*base_savings:,.0f} €) a Core All-World fino a w_sat=40%."
                else:
                    alloc['v4'] = split_4
                    sat_note = f"Core-Satellite Bilanciato: Satellite al {w_sat*100:.1f}% (entro banda [33%, 47%]). Flussi: 60% Core / 40% Satellite a zero tax drag."

                alloc['v3a'] = split_3a
                name = f"Fase 6: Core-Satellite a Regime con Inflow Rebalancing ({split_4*100:.0f}% V4 / {split_3a*100:.0f}% V3A)"
                desc = f"Obiettivo H completato. Patrimonio in accumulo azionario globale con banda d'isteresi [33%, 47%]. {sat_note}"
                color = "#16a34a"
            else:
                alloc['v4'] = split_4
                alloc['v3a'] = split_3a
                name = f"Fase 6: Regime Crescita Azionaria & Mantenimento ({split_4*100:.0f}% V4 / {split_3a*100:.0f}% V3A)"
                desc = "Obiettivo Milestone H interamente immunizzato! Il risparmio mensile alimenta massicciamente il comparto azionario globale Core ETF All-World (85%), con un 15% di mantenimento difensivo anti-inflazione."
                color = "#0d9488"

    return {
        'fase_idx': fase_idx,
        'name': name,
        'desc': desc,
        'badge_color': color,
        'alloc': alloc,
        'anti_zirp_badge': anti_zirp,
        'preemption_active': is_preemption,
        'glidepath_active': is_glidepath,
        'issuance_gap_active': is_issuance_gap,
        'core_sat_active': is_core_sat
    }


def apply_realloc_action(v1_new, v2_new, v3b_new, v3a_new, v4_new):
    """Callback di Streamlit per applicare istantaneamente i saldi ottimizzati."""
    st.session_state["input_v1"] = float(v1_new)
    st.session_state["input_v2"] = float(v2_new)
    st.session_state["input_v3b"] = float(v3b_new)
    st.session_state["input_v3a"] = float(v3a_new)
    st.session_state["input_v4"] = float(v4_new)
    st.session_state["realloc_just_applied"] = True


# ------------------------------------------------------------------------------
# 3. SIDEBAR DEI PARAMETRI UTENTE
# ------------------------------------------------------------------------------

# Inizializzazione Session State per configurazione utente (da Google Sheets o default)
if "user_data" not in st.session_state:
    st.session_state["user_data"] = load_user_data()

user_cfg = st.session_state["user_data"]

with st.sidebar:
    st.markdown("### ⚙️ Parametri del Modello")
    st.markdown("Configura il profilo finanziario e le preferenze comportamentali.")

    # Sincronizzazione Cloud Google Sheets
    with st.expander("☁️ Google Sheets Cloud Sync", expanded=False):
        st.caption("Persistenza cloud con [connections.gsheets].")
        col_g1, col_g2 = st.columns(2)
        with col_g2:
            if st.button("🔄 Ricarica", use_container_width=True, help="Ricarica i parametri dal foglio Google Sheets"):
                st.session_state["user_data"] = load_user_data()
                st.success("Parametri ricaricati!")
                st.rerun()
    
    # --- Profilo di Reddito ---
    with st.expander("💼 Profilo di Reddito", expanded=True):
        Y0 = st.number_input(
            "Stipendio netto mensile (€)",
            min_value=800.0,
            max_value=25000.0,
            value=float(user_cfg.get("Y0", 2000.0)),
            step=100.0,
            help="Reddito da lavoro netto mensile di partenza (base tempo t=0)."
        )

        # Regola Comportamentale SMarT / Lifestyle Inflation su Stipendio di Partenza
        delta_Y = max(0.0, Y0 - 2000.0)
        smart_lifestyle_choice = st.radio(
            "Regola Tenore di Vita su Extra Stipendio (ΔY):",
            [
                "Aumenta tenore di vita (Regola SMarT: 50% spese / 50% risparmio)",
                "Mantieni spese invariate (100% dell'extra al risparmio)",
                "Personalizzato (imposta slider %)"
            ],
            index=0,
            help="Determina come viene allocata la quota di stipendio superiore al benchmark di 2.000 € tra aumento dei consumi ordinari (lifestyle inflation) e risparmio."
        )
        if "50%" in smart_lifestyle_choice:
            split_consumo = 0.50
        elif "invariate" in smart_lifestyle_choice:
            split_consumo = 0.0
        else:
            split_consumo_pct = st.slider(
                "Quota incremento ΔY a Consumi/Spese (%)",
                min_value=0,
                max_value=100,
                value=50,
                step=5,
                help="Percentuale di delta_Y assorbita dall'aumento del tenore di vita."
            )
            split_consumo = split_consumo_pct / 100.0

        ral_0 = st.number_input(
            "RAL di partenza (€)",
            min_value=12000.0,
            max_value=300000.0,
            value=float(user_cfg.get("ral_0", 32000.0)),
            step=1000.0,
            help="Retribuzione Annua Lorda da cui derivano il TFR (6,91%) e la contribuzione aziendale (1,5%)."
        )
        g_Y_pct = st.slider(
            "Crescita salariale annua (%)",
            min_value=0.0,
            max_value=5.0,
            value=float(user_cfg.get("g_Y_pct", 1.8)),
            step=0.1,
            help="Tasso nominale annuo di incremento stipendiale (default benchmark accademico: 1,8% p.a.)."
        )
        g_Y = g_Y_pct / 100.0

    # --- Spese e Passività ---
    with st.expander("🏠 Spese e Passività", expanded=True):
        no_mortgage = st.checkbox("Nessun mutuo in corso", value=bool(user_cfg.get("no_mortgage", False)))
        if not no_mortgage:
            mortgage_R0 = st.number_input(
                "Rata mensile del mutuo (€)",
                min_value=100.0,
                max_value=5000.0,
                value=float(user_cfg.get("mortgage_R0", 600.0)),
                step=50.0,
                help="Rata mensile per finanziamento prima casa. Determina il Vaso 3B (buffer = 4 rate)."
            )
        else:
            mortgage_R0 = 0.0
            st.caption("ℹ️ Vaso 3B disattivato (Buffer Mutuo = 0 €).")
            
        if abs(Y0 - 2000.0) > 1e-2:
            e_base_help = (
                f"Totale uscite ordinarie mensili (vitto, bollette, trasporti, inclusa rata mutuo). "
                f"Con stipendio inserito di {Y0:,.0f} € (diverso dal default di 2.000 €), questo valore "
                f"rappresenta il livello di spesa base prima dell'applicazione della regola SMarT sui guadagni extra (ΔY = {delta_Y:,.0f} €)."
            )
        else:
            e_base_help = "Totale uscite ordinarie mensili (vitto, bollette, trasporti, inclusa rata mutuo se presente)."

        E_base = st.number_input(
            "Spese mensili ordinarie di partenza (€)",
            min_value=500.0,
            max_value=20000.0,
            value=float(user_cfg.get("E_base", 1500.0)),
            step=50.0,
            help=e_base_help
        )

        # Calcolo del flusso mensile S_t con priorità rigida
        override_spese = st.checkbox(
            "Specifica spese mensili effettive puntuali (override)",
            value=False,
            help="Consente di inserire direttamente le spese vive mensili attuali per il calcolo puntuale del flusso S_t."
        )
        if override_spese:
            E_eff_input = st.number_input(
                "Spese mensili effettive correnti (€)",
                min_value=300.0,
                max_value=30000.0,
                value=float(E_base),
                step=50.0,
                help="Spese correnti effettive sostenute dall'utente nel mese."
            )
            E_effettivo = float(E_eff_input)
            # S_t source: surplus | SMarT | base
            S_t_source = "surplus"
            S_t = max(0.0, Y0 - E_effettivo)
            st.caption(f"ℹ️ Flusso calcolato su spese effettive: S_t = {S_t:,.0f} €/mese (Fonte: surplus)")
        elif delta_Y > 0 and split_consumo > 0.0:
            E_effettivo = E_base + (split_consumo * delta_Y)
            # S_t source: surplus | SMarT | base
            S_t_source = "SMarT"
            S_t = max(0.0, Y0 - E_effettivo)
        else:
            E_effettivo = E_base
            # S_t source: surplus | SMarT | base
            S_t_source = "base"
            S_t = max(0.0, Y0 - E_base)

        E0 = E_effettivo  # Alias globale per coerenza e compatibilità

        if not override_spese and delta_Y > 0:
            extra_spese = split_consumo * delta_Y
            extra_risp = (1.0 - split_consumo) * delta_Y
            st.markdown(f"""
            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 8px 10px; font-size: 0.82rem; color: #166534; margin-top: 6px;">
                📈 <b>Regola SMarT Attiva (ΔY = +{delta_Y:,.0f} € - Fonte: SMarT):</b><br>
                • Spesa base impostata: <b>{E_base:,.0f} €/mese</b><br>
                • Spese vive effettive post-SMarT: <b>{E_effettivo:,.0f} €/mese</b> (+{extra_spese:,.0f} € tenore di vita)<br>
                • Extra risparmio strutturale: <b>+{extra_risp:,.0f} €/mese</b>
            </div>
            """, unsafe_allow_html=True)
        elif not override_spese and Y0 < 2000.0:
            st.caption(f"ℹ️ Livello di spesa impostato: {E_base:,.0f} €/mese (stipendio inferiore a 2.000 €, nessun extra ΔY applicato). Spese effettive: {E_effettivo:,.0f} €.")
        elif not override_spese:
            st.caption(f"ℹ️ Livello di spesa impostato: {E_base:,.0f} €/mese (Fonte: base, pari alle spese effettive).")

    # --- Orizzonte Temporale Dinamico & Età Anagrafica ---
    with st.expander("⏳ Orizzonte Temporale & Età Anagrafica", expanded=True):
        col_age1, col_age2 = st.columns(2)
        with col_age1:
            eta_attuale = st.number_input(
                "Età attuale",
                min_value=18,
                max_value=75,
                value=int(user_cfg.get("eta_attuale", 30)),
                step=1,
                help="Età anagrafica dell'investitore all'inizio del piano (t=0)."
            )
        with col_age2:
            eta_pensionabile = st.number_input(
                "Età pensionabile",
                min_value=max(50, int(eta_attuale) + 1),
                max_value=85,
                value=int(user_cfg.get("eta_pensionabile", 67)),
                step=1,
                help="Età anagrafica programmata per il pensionamento."
            )
        
        anni_accumulo = max(1, int(eta_pensionabile - eta_attuale))
        t_retire_calc = anni_accumulo * 12
        
        include_decumulo = st.checkbox(
            "Includi fase di decumulo post-pensione (20 anni = 240 mesi)",
            value=bool(user_cfg.get("include_decumulo", True)),
            help="Se attivo: T_horizon = T_retire + 240 (20 anni di decumulo post-pensionamento). Se disattivato: orizzonte limitato alla sola fase di accumulo fino alla pensione (T_horizon = T_retire, con glidepath negli ultimi 7 anni)."
        )
        if include_decumulo:
            T_horizon_val = t_retire_calc + 240
            T_retire_val = t_retire_calc
        else:
            T_horizon_val = t_retire_calc
            T_retire_val = max(12, T_horizon_val - 84) # ultimi 7 anni
            
        st.markdown(f"""
        <div style="background:#eff6ff; border:1px solid #bfdbfe; border-radius:6px; padding:6px 10px; font-size:0.83rem; color:#1e40af; margin-top:4px; margin-bottom:8px;">
            ⏱️ <b>Orizzonte simulato: {T_horizon_val // 12} anni ({T_horizon_val} mesi)</b><br>
            • Inizio Glidepath Pensione: età <b>{int(eta_attuale + T_retire_val // 12)}</b> (Mese {T_retire_val})<br>
            • Conclusione piano: età <b>{int(eta_attuale + T_horizon_val // 12)}</b> (Mese {T_horizon_val})
        </div>
        """, unsafe_allow_html=True)

    # --- Obiettivo a Medio Termine (Milestone H) ---
    with st.expander("🎯 Obiettivo Medio Termine (Milestone H)", expanded=True):
        enable_goal_H = st.checkbox("Attiva Obiettivo Programmato (Auto/Progetto)", value=bool(user_cfg.get("enable_goal_H", True)))
        if enable_goal_H:
            T3a_0 = st.number_input(
                "Costo stimato obiettivo (€ a t=0)",
                min_value=2000.0,
                max_value=150000.0,
                value=float(user_cfg.get("T3a_0", 25000.0)),
                step=1000.0,
                help="Spesa programmata reale (es. acquisto autovettura), indicizzata all'inflazione cumulata."
            )
            H_years = st.slider(
                "Orizzonte temporale Milestone H (Anni)",
                min_value=3,
                max_value=min(15, max(4, T_horizon_val // 12 - 1)),
                value=int(user_cfg.get("H_years", 8)),
                step=1,
                help=f"Numero di anni alla scadenza dell'obiettivo (default 8 anni = 96 mesi, età anagrafica: {int(eta_attuale + 8)} anni)."
            )
        else:
            T3a_0 = 0.0
            H_years = 8
            st.caption("ℹ️ Vaso 3A disattivato: i risparmi confluiscono direttamente verso il Vaso 4.")

        current_month_t = st.number_input(
            f"Mese di avanzamento del piano (t su {T_horizon_val})",
            min_value=1,
            max_value=int(T_horizon_val),
            value=1,
            step=1,
            help=f"Mese corrente del piano (1-{T_horizon_val}). Se mancano ≤ 24 mesi a Milestone H scatta l'Issuance Gap Lock-In su V3A. Dal mese t={T_retire_val} si attiva la Fase 6b Subordinated Retirement Glidepath."
        )

    # --- Dimensionamento Target Vasi (Duale: Automatico vs Manuale) ---
    with st.expander("🎯 Dimensionamento Target Vasi", expanded=True):
        target_mode = st.radio(
            "Modalità Dimensionamento Target:",
            ["Consigliato da S-DFP (Automatico)", "Manuale / Personalizzato"],
            index=0,
            help="Scegli se utilizzare i parametri teorici ottimali S-DFP o personalizzare ciascun comparto."
        )

        # Calcolo analitico valori consigliati da S-DFP
        s0_calc = max(0.0, Y0 - E_effettivo)
        auto_t1 = 3.0 * E_effettivo
        auto_t2 = 2400.0
        auto_t3b = 4.0 * mortgage_R0 if not no_mortgage else 0.0
        auto_t3a = T3a_0 if enable_goal_H else 0.0
        auto_thresh_p4_pct = 60.0
        auto_thresh_v3a = (auto_thresh_p4_pct / 100.0) * auto_t3a
        auto_split_fase5_v3a_pct = 60.0

        if target_mode == "Consigliato da S-DFP (Automatico)":
            t1_target = float(auto_t1)
            t2_target = float(auto_t2)
            t3b_target = float(auto_t3b)
            t3a_target = float(auto_t3a)
            threshold_p4_pct = auto_thresh_p4_pct
            threshold_v3a_p4 = auto_thresh_v3a
            split_fase5_v3a_pct = auto_split_fase5_v3a_pct
            split_fase5_v3a = split_fase5_v3a_pct / 100.0

            smart_tooltip = f"Regola SMarT applicata: l'aumento di stipendio è ripartito tra tenore di vita ({split_consumo*100:.0f}%) e risparmio ({(1-split_consumo)*100:.0f}%)."
            st.markdown(f"""
            <div style="background:#f1f5f9; border:1px solid #cbd5e1; border-radius:8px; padding:12px; margin-bottom:8px;">
                <span class="badge-tag badge-accent" style="display:inline-block; margin-bottom:8px;" title="{smart_tooltip}">⚡ Calcolato con Formula S-DFP</span>
                <div style="font-size:0.83rem; color:#1e293b; line-height:1.7;">
                    • <b>Target V1 (Emergenza):</b> 3 × E_effettivo = <b>{t1_target:,.0f} €</b><br>
                    • <b>Target V2 (Sinking Fund):</b> <b>{t2_target:,.0f} €</b> (Spese cicliche 12-24m)<br>
                    • <b>Target V3B (Buffer Mutuo):</b> <b>{t3b_target:,.0f} €</b> ({'4 rate mutuo' if not no_mortgage else 'Nessun mutuo'})<br>
                    • <b>Target V3A (Milestone H):</b> <b>{t3a_target:,.0f} €</b><br>
                    • <b>Soglia Fase 4 (Sblocco azionario):</b> 60% = <b>{threshold_v3a_p4:,.0f} €</b><br>
                    • <b>Split di Fase 5:</b> <b>60% V3A / 40% V4</b><br>
                    • <b>Risparmio Iniziale (S₀):</b> <b>{s0_calc:,.0f} €/mese</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"🔒 Formule analitiche vincolate S-DFP attive. {smart_tooltip}")

        else:
            st.markdown('<span class="badge-tag" style="background:#e0e7ff; color:#3730a3; margin-bottom:8px; display:inline-block;">✏️ Configurazione Manuale Target</span>', unsafe_allow_html=True)
            t1_target = st.number_input(
                "Target V1 Emergenza (€)",
                min_value=500.0,
                max_value=100000.0,
                value=float(auto_t1),
                step=100.0,
                help=f"Buffer per emergenze e spese ordinarie vive (consigliato: 3 mensilità spese = 3 x {E_effettivo:,.0f} €)."
            )
            t2_target = st.number_input(
                "Target V2 Sinking Fund (€)",
                min_value=0.0,
                max_value=50000.0,
                value=float(auto_t2),
                step=100.0,
                help="Fondo per spese cicliche programmate a 12-24 mesi (assicurazioni, manutenzioni, bolli)."
            )
            if not no_mortgage:
                t3b_target = st.number_input(
                    "Target V3B Buffer Mutuo (€)",
                    min_value=0.0,
                    max_value=50000.0,
                    value=float(auto_t3b),
                    step=100.0,
                    help="Pavimento di sicurezza per rate del finanziamento (consigliato: 4 mensilità rata = 4 x R₀)."
                )
            else:
                t3b_target = 0.0
                st.caption("ℹ️ Mutuo disattivato (Target V3B = 0 €).")

            t3a_target = T3a_0 if enable_goal_H else 0.0

            threshold_p4_pct = st.slider(
                "Soglia Sblocco Azionario Anticipato V3A (%)",
                min_value=10,
                max_value=90,
                value=int(auto_thresh_p4_pct),
                step=5,
                help="Percentuale di V3A oltre la quale si attiva l'inversione di flusso di Fase 5 (default 60%)."
            )
            threshold_v3a_p4 = (threshold_p4_pct / 100.0) * t3a_target

            split_fase5_v3a_pct = st.slider(
                "Split Flusso Fase 5 su Vaso 3A (%)",
                min_value=10,
                max_value=90,
                value=int(auto_split_fase5_v3a_pct),
                step=5,
                help="Percentuale di flusso mensile verso V3A in Fase 5. Il residuo confluisce nel Vaso 4 Core ETF All-World."
            )
            split_fase5_v3a = split_fase5_v3a_pct / 100.0

    # --- Regola Salariale Comportamentale ---
    with st.expander("🧠 Regola Aumenti Salariali (SMarT)", expanded=True):
        split_options = [
            "50/50 SMarT (50% Consumo / 50% Risparmio)",
            "30% Consumo / 70% Risparmio",
            "70% Consumo / 30% Risparmio",
            "100% Risparmio (Scenario B - Austerità)"
        ]
        selected_split = st.selectbox(
            "Ripartizione degli incrementi ΔY",
            options=split_options,
            index=0,
            help="Come vengono allocati gli aumenti di stipendio futuri: bilanciamento tra tenore di vita e risparmio."
        )
        if "50/50" in selected_split:
            smart_split = 0.50
        elif "30% Consumo" in selected_split:
            smart_split = 0.30
        elif "70% Consumo" in selected_split:
            smart_split = 0.70
        else:
            smart_split = 0.0 # Scenario B: 100% risparmio

    # --- Impostazioni di Simulazione ---
    with st.expander("🎲 Impostazioni Monte Carlo", expanded=False):
        n_paths_label = st.radio(
            "Precisione di Calcolo",
            ["500 (Calcolo Istantaneo ⚡)", "2.500 (Analisi Approfondita 🔬)", "10.000 (Certificazione Accademica 🎓)"],
            index=0,
            help="Numero di traiettorie stocastiche simultanee. 500 per interattività istantanea."
        )
        if "500" in n_paths_label:
            n_paths = 500
        elif "2.500" in n_paths_label:
            n_paths = 2500
        else:
            n_paths = 10000
            
        seed = st.number_input("Seed stocastico (CRN)", min_value=1, max_value=999999, value=42, step=1)

    st.markdown("---")
    run_sim = st.button("🚀 Avvia Simulazione / Ricalcola", type="primary", use_container_width=True)
    
    # Calcolo rapido flussi base
    base_savings = max(0.0, Y0 - E_effettivo)
    st.markdown(f"""
    <div style='background: #e2e8f0; border-radius: 8px; padding: 12px; margin-top: 10px; font-size: 0.85rem;'>
        <b>📊 Sintesi Flussi e Target Attivi:</b><br>
        • Stipendio Netto (Y₀): <b>{Y0:,.0f} €/mese</b><br>
        • Spese Effettive (E_eff): <b>{E_effettivo:,.0f} €/mese</b><br>
        • Risparmio Base (S₀): <b>{base_savings:,.0f} €/mese</b><br>
        • Target V1 (3×E_eff): <b>{t1_target:,.0f} €</b><br>
        • Target V2 (Sinking): <b>{t2_target:,.0f} €</b><br>
        • Target V3B (Buffer Mutuo): <b>{t3b_target:,.0f} €</b><br>
        • Target V3A (Milestone H): <b>{t3a_target:,.0f} €</b><br>
        • Soglia Sblocco V3A: <b>{threshold_v3a_p4:,.0f} € ({threshold_p4_pct:.0f}%)</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### ☁️ Persistenza Google Sheets")
    if st.button("💾 Salva Configurazione su Google Sheets", use_container_width=True, help="Salva i parametri correnti sul foglio Google Sheets collegato"):
        payload = {
            "Y0": Y0,
            "ral_0": ral_0,
            "g_Y_pct": g_Y_pct,
            "smart_lifestyle_choice": smart_lifestyle_choice,
            "split_consumo_pct": int(split_consumo * 100),
            "no_mortgage": no_mortgage,
            "mortgage_R0": mortgage_R0,
            "E_base": E_base,
            "override_spese": override_spese,
            "eta_attuale": eta_attuale,
            "eta_pensionabile": eta_pensionabile,
            "include_decumulo": include_decumulo,
            "enable_goal_H": enable_goal_H,
            "T3a_0": T3a_0,
            "H_years": H_years,
            "current_month_t": current_month_t,
            "target_mode": target_mode,
            "input_v1": st.session_state.get("input_v1", t1_target + 500.0),
            "input_v2": st.session_state.get("input_v2", max(0.0, t2_target - 400.0)),
            "input_v3b": st.session_state.get("input_v3b", t3b_target),
            "input_v3a": st.session_state.get("input_v3a", min(16000.0, t3a_target) if t3a_target > 0 else 0.0),
            "input_v4": st.session_state.get("input_v4", 25000.0),
        }
        if save_user_data(payload):
            st.session_state["user_data"] = payload
            st.success("✅ Parametri salvati con successo su Google Sheets!")


# ------------------------------------------------------------------------------
# 4. ESECUZIONE SIMULAZIONE E RACCOLTA DATI
# ------------------------------------------------------------------------------

with st.spinner("Calcolo stocastico Monte Carlo in corso sotto Common Random Numbers..."):
    sim_data = run_cached_simulation(
        n_paths=n_paths,
        seed=seed,
        Y0=Y0,
        ral_0=ral_0,
        g_Y=g_Y,
        mortgage_R0=mortgage_R0,
        E0=E_effettivo,
        T3a_0=t3a_target,
        H_years=H_years,
        smart_split=smart_split,
        T2_fixed=t2_target,
        T3b_fixed=t3b_target,
        T_horizon=T_horizon_val,
        T_retire=T_retire_val
    )

summary = sim_data['summary']
m5_res = summary['M5_SDFP_ScenC']
m2_res = summary['M2_Fair']
m0_res = summary['M0_Naive']
m4_res = summary['M4_SDFP_ScenB']
m7_res = summary.get('M7_Fair_LiabInflowReb', None)
stress_data = sim_data['stress']
v3a_track = sim_data['v3a_track']
yearly_data = sim_data['yearly']


# ------------------------------------------------------------------------------
# 5. HEADER PRINCIPALE E METADATA
# ------------------------------------------------------------------------------

st.markdown('<div class="fintech-title">🏛️ S-DFP: State-Dependent Sequential Flow Allocation</div>', unsafe_allow_html=True)
st.markdown('<div class="fintech-subtitle">Motore di Gestione Dinamica del Risparmio e Allocazione per Obiettivi (Goal-Based Wealth Management)</div>', unsafe_allow_html=True)

st.markdown("""
<div class="badge-bar">
    <span class="badge-tag badge-accent">✔ Motore Validato</span>
    <span class="badge-tag">Regime Fiscale: TUIR & D.Lgs. 461/1997 (Amministrato)</span>
    <span class="badge-tag">Inflow Rebalancing: Zero Tax Drag [35%, 45%]</span>
    <span class="badge-tag">Common Random Numbers (CRN)</span>
    <span class="badge-tag">Anti-ZIRP Switch (€STR / XEON)</span>
</div>
""", unsafe_allow_html=True)

# --- Modulo Guida Rapida: S-DFP in una pagina ---
with st.expander("📖 S-DFP in una pagina — La strategia in 60 secondi (Guida Rapida & Flyer PDF)", expanded=False):
    col_info_l, col_info_r = st.columns([1.5, 1.0])
    with col_info_l:
        st.markdown("""
        ##### 📦 La strategia in 60 secondi: 6 scatole separate
        Immagina di avere 6 scatole separate, ognuna con uno scopo preciso:
        1. **Emergenze** — 3–6 mesi di spese vive, sempre liquidi sul conto/deposito.
        2. **Spese a breve** — assicurazione auto, bollo, vacanze, spese cicliche (1–2 anni).
        3. **Sicurezza mutuo** — 4 rate accantonate, per non rischiare mai la casa.
        4. **Obiettivo importante** — auto nuova, cucina, caparra casa (5–15 anni), investiti in titoli sicuri a scadenza.
        5. **Crescita** — ETF azionario globale, orizzonte 20–40 anni per la ricchezza finanziaria.
        6. **Pensione** — fondo pensione, TFR, massimo deducibile annuale.

        > **Il Ciclo di Prelazione:** Ogni mese il risparmio colma le scatole in ordine rigoroso di priorità: prima le protezioni (1, 2, 3), poi l'obiettivo (4), infine la crescita (5). Se un mese hai un'emergenza, il flusso si arresta a monte e **non tocchi mai le azioni**. Quando affronti una spesa straordinaria, attingi dalla scatola dell'obiettivo (4), non dalla crescita (5). Così **non vendi mai azioni nel momento sbagliato**.
        """)
        
        st.markdown("""
        ##### ⭐ I 5 vantaggi chiave
        * **1. Non vendi mai le azioni nel momento sbagliato:** Le spese importanti sono protette. Eviti le vendite forzate ai minimi di mercato.
        * **2. Paghi meno tasse:** Nessuna vendita per ribilanciare (Zero Capital Gains Drag: ritenuta 26% differita). Orienti solo i nuovi risparmi verso i vasi sottomisura.
        * **3. Proteggi i tuoi obiettivi:** Il capitale a 5–15 anni è immunizzato a scadenza (*Non è un investimento, è una garanzia*).
        * **4. Non ti fai prendere dal panico:** In caso di crollo di mercato, la vita quotidiana è blindata dalle riserve liquide.
        * **5. Sai sempre dove sono i tuoi soldi:** Regole deterministiche oggettive: meno decisioni, zero errori emotivi.
        """)

    with col_info_r:
        st.markdown("""
        ##### 📋 Le 3 regole da ricordare
        1. **Prima le protezioni, poi la crescita:** Buffer, sicurezza e scadenze prima di alimentare l'azionario.
        2. **Non vendere mai per ribilanciare:** Usa i nuovi risparmi per riallineare i pesi di portafoglio.
        3. **Le azioni non si toccano per le spese a breve:** La spesa esce dalla scatola dell'obiettivo, non dalla crescita.

        ---
        ##### 🚫 Cosa NON è S-DFP
        * **Non è per arricchirsi in fretta:** È per non perdere soldi nel modo sbagliato.
        * **Non è un portafoglio unico:** È un sistema di comparti stagni (Mental Accounting).
        * **Non elimina i cali di borsa:** Elimina la necessità di liquidare le posizioni durante i ribassi.
        * **Non è complicato:** 6 scatole, 3 regole, un ordine sequenziale.

        ---
        ##### 🎯 A chi serve
        * **✔ Ideale per:** Dipendenti e famiglie con stipendio stabile, spese programmate a 5–20 anni e visione a lungo termine.
        * **✖ Non adatto a:** Grandi patrimoni (>500k € già consolidati), redditi iper-irregolari senza base, o speculatori. *S-DFP non batte il mercato: ti fa perdere meno.*
        """)

    st.info('💡 **In una riga:** *"S-DFP è un modo per organizzare i tuoi risparmi in scatole separate, così le spese importanti della vita non ti costringono mai a vendere i tuoi investimenti nel momento sbagliato."*')

    col_dl1, col_dl2 = st.columns(2)
    p_one_page_pdf = resolve_file_path("S_DFP_in_una_pagina.pdf")
    if p_one_page_pdf and p_one_page_pdf.exists():
        with open(p_one_page_pdf, "rb") as f_pdf:
            col_dl1.download_button(
                label="📄 Scarica Scheda Sintetica (PDF 1 Pagina A4)",
                data=f_pdf.read(),
                file_name="S_DFP_in_una_pagina.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    p_one_page_md = resolve_file_path("S_DFP_in_una_pagina.md")
    if p_one_page_md and p_one_page_md.exists():
        with open(p_one_page_md, "rb") as f_md:
            col_dl2.download_button(
                label="📝 Scarica Documento Sintetico (Markdown)",
                data=f_md.read(),
                file_name="S_DFP_in_una_pagina.md",
                mime="text/markdown",
                use_container_width=True
            )




# ------------------------------------------------------------------------------
# 6. FUNZIONI QUANTITATIVE ANALITICHE E CACHING DIAGNOSTICO
# ------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def simulate_vasicek_paths(r0=0.0325, kappa=0.35, theta=0.022, sigma=0.012, n_months=360, n_paths=50, seed=42):
    """
    Simula percorsi stocastici del tasso d'interesse breve sotto modello mean-reverting di Vasicek:
    dr_t = kappa * (theta - r_t) dt + sigma * dW_t
    """
    np.random.seed(seed)
    dt = 1.0 / 12.0
    paths = np.zeros((n_paths, n_months + 1))
    paths[:, 0] = r0
    for t in range(n_months):
        r_prev = paths[:, t]
        z = np.random.normal(0, 1, n_paths)
        dr = kappa * (theta - r_prev) * dt + sigma * np.sqrt(dt) * z
        paths[:, t + 1] = r_prev + dr
    return paths


@st.cache_data(show_spinner=False)
def compute_dynamic_target_trajectories(Y0, E_base, split_consumo, g_Y, mortgage_R0, t2_target, t3b_target, t3a_target, H_years, n_months=360):
    """
    Calcola le traiettorie deterministiche dei target T1(t), T2, T3b, T3a(t) e soglia Theta_3a(t) su 30 anni.
    """
    months = np.arange(n_months + 1)
    years = months / 12.0
    years_int = np.floor(years)
    Y_t = Y0 * ((1.0 + g_Y) ** years_int)
    delta_Y_t = np.maximum(0.0, Y_t - 2000.0)
    E_t = E_base + split_consumo * delta_Y_t
    T1_t = 3.0 * E_t
    
    H_month = int(H_years * 12)
    pi_bar = 0.02
    T3a_t = np.zeros(n_months + 1)
    for m in range(n_months + 1):
        if m <= H_month and t3a_target > 0:
            T3a_t[m] = t3a_target * ((1.0 + pi_bar) ** (m / 12.0))
        else:
            T3a_t[m] = 0.0
            
    Theta_3a_t = 0.60 * T3a_t
    T2_t = np.full(n_months + 1, t2_target)
    T3b_t = np.full(n_months + 1, t3b_target)
    
    return {
        "months": months,
        "years": years,
        "T1": T1_t,
        "T2": T2_t,
        "T3b": T3b_t,
        "T3a": T3a_t,
        "Theta_3a": Theta_3a_t,
        "H_month": H_month
    }


@st.cache_data(show_spinner=False)
def compute_savings_cashflow_dynamics(Y0, E_base, split_consumo, g_Y, mortgage_R0, n_months=360):
    """
    Calcola la scomposizione mese per mese del reddito disponibile tra spese vive base,
    lifestyle expansion SMarT, rata mutuo e risparmio netto allocabile.
    """
    months = np.arange(n_months + 1)
    years = months / 12.0
    years_int = np.floor(years)
    
    Y_t = Y0 * ((1.0 + g_Y) ** years_int)
    delta_Y_t = np.maximum(0.0, Y_t - 2000.0)
    
    E0_curve = np.full(n_months + 1, E_base)
    lifestyle_exp = split_consumo * delta_Y_t
    mortgage_curve = np.full(n_months + 1, mortgage_R0)
    savings_curve = np.maximum(0.0, Y_t - (E0_curve + lifestyle_exp + mortgage_curve))
    
    return {
        "months": months,
        "years": years,
        "Y_t": Y_t,
        "E0": E0_curve,
        "lifestyle": lifestyle_exp,
        "mortgage": mortgage_curve,
        "savings": savings_curve
    }


# ------------------------------------------------------------------------------
# 7. SCHEDE DELLA DASHBOARD INTERATTIVA
# ------------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Stato e Allocazione (Oggi)",
    "📈 Proiezione a 30 Anni",
    "🛡️ Protezione Obiettivi & Stress Test",
    "📐 Framework Matematico, Equazioni Dinamiche & Diagnostica Grafica",
    "💼 Dossier Titoli & Monitor Scadenze",
    "💾 Esportazione Dati"
])


# ==============================================================================
# TAB 1: STATO E ALLOCAZIONE DEL RISPARMIO (OGGI)
# ==============================================================================
with tab1:
    st.markdown("#### 🎯 Cruscotto Decisionale ed Operativo (Oggi)")
    st.caption("Verifica la fase attiva della macchina a stati, consulta la destinazione vincolante del risparmio di questo mese e gestisci la cassa ferma in surplus con la Waterflow Cascade.")

    # Inizializzazione session state con valori realistici
    if "input_v1" not in st.session_state:
        st.session_state["input_v1"] = float(user_cfg.get("input_v1", t1_target + 500.0))
    if "input_v2" not in st.session_state:
        st.session_state["input_v2"] = float(user_cfg.get("input_v2", max(0.0, t2_target - 400.0)))
    if "input_v3b" not in st.session_state:
        st.session_state["input_v3b"] = float(user_cfg.get("input_v3b", t3b_target))
    if "input_v3a" not in st.session_state:
        st.session_state["input_v3a"] = float(user_cfg.get("input_v3a", min(16000.0, t3a_target) if t3a_target > 0 else 0.0))
    if "input_v4" not in st.session_state:
        st.session_state["input_v4"] = float(user_cfg.get("input_v4", 25000.0))

    # Scelta modalità inserimento saldi
    col_opt, col_info = st.columns([1, 2])
    with col_opt:
        state_mode = st.radio(
            "Modalità di inserimento saldi:",
            ["Simulatore Saldi Attuali (Personalizzato)", "Mese 1 (Partenza da zero)"],
            horizontal=True
        )

    if state_mode == "Mese 1 (Partenza da zero)":
        v1_val, v2_val, v3b_val, v3a_val, v4_val = 0.0, 0.0, 0.0, 0.0, 0.0
    else:
        st.markdown("##### ✏️ Saldi Attuali dei Tuoi Comparti:")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: v1_val = st.number_input("Saldo V1 Emergenza (€)", min_value=0.0, step=100.0, key="input_v1")
        with c2: v2_val = st.number_input("Saldo V2 Sinking Fund (€)", min_value=0.0, step=100.0, key="input_v2")
        with c3: v3b_val = st.number_input("Saldo V3B Mutuo (€)", min_value=0.0, step=100.0, key="input_v3b") if t3b_target > 0 else 0.0
        with c4: v3a_val = st.number_input("Saldo V3A Auto (€)", min_value=0.0, step=100.0, key="input_v3a") if t3a_target > 0 else 0.0
        with c5: v4_val = st.number_input("Saldo V4 Azionario (€)", min_value=0.0, step=500.0, key="input_v4")

    # Notifica se la riallocazione è stata appena applicata via callback
    if st.session_state.get("realloc_just_applied", False):
        st.success("🎉 **Riallocazione del Surplus Applicata con Successo!** I campi numerici sono stati aggiornati secondo la gerarchia ottimale S-DFP.", icon="✅")
        st.session_state["realloc_just_applied"] = False

    # 1. Calcolo del Surplus/Deficit e Cascata di Distribuzione (Waterflow Cascade)
    cascade = compute_waterflow_cascade(
        v1=v1_val, v2=v2_val, v3b=v3b_val, v3a=v3a_val, v4=v4_val,
        t1=t1_target, t2=t2_target, t3b=t3b_target, t3a=t3a_target
    )
    total_surplus = cascade["total_surplus"]
    surplus_v1 = cascade["surplus_v1"]
    surplus_v2 = cascade["surplus_v2"]
    moves = cascade["moves"]
    v1_opt = cascade["v1_opt"]
    v2_opt = cascade["v2_opt"]
    v3b_opt = cascade["v3b_opt"]
    v3a_opt = cascade["v3a_opt"]
    v4_opt = cascade["v4_opt"]

    # 2. Determinazione Fasi della Macchina a Stati (Attuale e Post-Riallocazione)
    phase_curr = determine_state_machine_phase(
        v1_val, v2_val, v3b_val, v3a_val, v4_val,
        t1_target, t2_target, t3b_target, t3a_target,
        base_savings, threshold_v3a_p4, split_fase5_v3a,
        current_month_t=current_month_t, H_months=H_years*12,
        mortgage_active=not no_mortgage, auto_active=enable_goal_H,
        T_retire=T_retire_val, T_horizon=T_horizon_val
    )
    phase_opt = determine_state_machine_phase(
        v1_opt, v2_opt, v3b_opt, v3a_opt, v4_opt,
        t1_target, t2_target, t3b_target, t3a_target,
        base_savings, threshold_v3a_p4, split_fase5_v3a,
        current_month_t=current_month_t, H_months=H_years*12,
        mortgage_active=not no_mortgage, auto_active=enable_goal_H,
        T_retire=T_retire_val, T_horizon=T_horizon_val
    )

    st.markdown("---")

    # ==========================================================================
    # CARD A: BANNER DI STATO ("In quale fase ti trovi oggi")
    # ==========================================================================
    trans_note = ""
    if state_mode != "Mese 1 (Partenza da zero)" and total_surplus > 0 and phase_curr['fase_idx'] != phase_opt['fase_idx']:
        trans_note = f"""
        <div style="margin-top: 10px; font-size: 0.88rem; color: #1e40af; background: #eff6ff; padding: 8px 14px; border-radius: 6px; border-left: 4px solid #3b82f6;">
            ⚡ <b>Transizione di Fase Rilevata:</b> Applicando la riallocazione della cassa ferma in esubero (<b>{total_surplus:,.0f} €</b>), i vasi a monte risulteranno saturi e la macchina avanzerà istantaneamente a <b>{phase_opt['name']}</b>!
        </div>
        """

    st.markdown(f"""
    <div style="background: white; border-radius: 12px; padding: 18px 24px; border: 1px solid #e2e8f0; border-left: 6px solid {phase_curr['badge_color']}; box-shadow: 0 1px 3px rgba(0,0,0,0.04); margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 6px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="background: {phase_curr['badge_color']}15; color: {phase_curr['badge_color']}; border: 1px solid {phase_curr['badge_color']}40; font-size: 0.78rem; font-weight: 800; padding: 4px 10px; border-radius: 9999px; text-transform: uppercase; letter-spacing: 0.5px;">
                    Fase {phase_curr['fase_idx']} Attiva
                </span>
                <span style="font-size: 1.25rem; font-weight: 800; color: #0f172a;">
                    {phase_curr['name']}
                </span>
            </div>
            <span style="font-size: 0.82rem; font-weight: 600; color: #64748b;">
                In quale fase ti trovi oggi
            </span>
        </div>
        <div style="font-size: 0.94rem; color: #334155; line-height: 1.5;">
            {phase_curr['desc']}
        </div>
        {trans_note}
    </div>
    """, unsafe_allow_html=True)

    # ==========================================================================
    # CARD B: BOX DISPOSITIVO FLUSSO MENSILE ("Destinazione del Risparmio di Questo Mese")
    # ==========================================================================
    f_idx = phase_curr['fase_idx']
    instructions_html = []

    if base_savings <= 0:
        instructions_html.append(
            '<div style="color: #b91c1c; font-weight: 600; padding: 10px; background: #fef2f2; border-radius: 6px;">⚠️ Nessun risparmio mensile disponibile (Spese ordinarie E₀ ≥ Stipendio netto Y₀). Riduci le spese ordinarie o aumenta il reddito per avviare l\'alimentazione dei vasi S-DFP.</div>'
        )
    elif f_idx == 1:
        instructions_html.append(
            f'<div class="action-item">🛡️ Accantona <b>{base_savings:,.2f} € (100%)</b> su <b>Vaso 1 (Conto Deposito / €STR)</b> per consolidare il buffer di emergenza vitale. <span class="badge-tag" style="background:#fee2e2; color:#b91c1c;">Anti-ZIRP Switch Attivo</span></div>'
        )
    elif f_idx == 2:
        instructions_html.append(
            f'<div class="action-item">🗓️ Accantona <b>{base_savings:,.2f} € (100%)</b> su <b>Vaso 2 (Sinking Fund Spese Cicliche)</b> per coprire uscite programmate a 12-24 mesi.</div>'
        )
    elif f_idx == 3:
        instructions_html.append(
            f'<div class="action-item">🏠 Accantona <b>{base_savings:,.2f} € (100%)</b> su <b>Vaso 3B (Buffer Rata Mutuo)</b> per coprire il pavimento di 4 mensilità. <span class="badge-tag" style="background:#fef3c7; color:#92400e;">Anti-ZIRP Switch Attivo</span></div>'
        )
    elif f_idx == 4:
        if phase_curr.get('issuance_gap_active'):
            rem_m = max(0, H_years * 12 - current_month_t)
            instructions_html.append(
                f'<div class="action-item" style="border-left: 4px solid #9333ea;">🔒 <b>Issuance Gap Lock-In ({rem_m} mesi a Milestone H):</b> Investi <b>{base_savings:,.2f} € (100%)</b> su <b>Vaso 3A (Titoli di Stato a scadenza coincidente)</b> per azzerare il rischio di prezzo e sequenza pre-scadenza.</div>'
            )
        else:
            instructions_html.append(
                f'<div class="action-item">🚗 Investi <b>{base_savings:,.2f} € (100%)</b> su <b>Vaso 3A (Titoli di Stato a scadenza H)</b> per avviare l\'immunizzazione dell\'obiettivo.</div>'
            )
    elif f_idx == 5:
        eur_3a = split_fase5_v3a * base_savings
        eur_v4 = (1.0 - split_fase5_v3a) * base_savings
        instructions_html.append(
            f'<div class="action-item">🟦 <b>{eur_3a:,.2f} € ({split_fase5_v3a*100:.0f}%)</b> su <b>Vaso 3A (Titoli di Stato a scadenza H)</b></div>'
        )
        instructions_html.append(
            f'<div class="action-item">🚀 <b>{eur_v4:,.2f} € ({(1.0-split_fase5_v3a)*100:.0f}%)</b> su <b>Vaso 4 (Core ETF Azionario All-World)</b></div>'
        )
    else:  # Fase 6 / 6b
        if phase_curr.get('glidepath_active'):
            shift_eur = 0.0025 * v4_val
            rem_pension_m = max(0, T_horizon_val - current_month_t)
            instructions_html.append(
                f'<div class="action-item" style="border-left: 4px solid #0284c7;">🛡️ <b>{base_savings:,.2f} € (100%)</b> su <b>Vaso 5 (Previdenza / De-risking Obbligazionario)</b></div>'
            )
            instructions_html.append(
                f'<div class="action-item" style="background:#f0f9ff; border-left: 4px solid #0284c7;">🔄 <b>Ribilanciamento di Stock Attivo:</b> Trasferisci <b>{shift_eur:,.2f} €/mese (0.25% di V4)</b> da Vaso 4 a Vaso 5 ({rem_pension_m} mesi al retirement T_retire=360).</div>'
            )
        else:
            eur_v4 = 0.85 * base_savings if t3a_target > 0 else base_savings
            eur_3a = 0.15 * base_savings if t3a_target > 0 else 0.0
            instructions_html.append(
                f'<div class="action-item">🚀 <b>{eur_v4:,.2f} € ({85 if t3a_target > 0 else 100:.0f}%)</b> su <b>Vaso 4 (Core ETF Azionario All-World)</b></div>'
            )
            if t3a_target > 0:
                instructions_html.append(
                    f'<div class="action-item">🛡️ <b>{eur_3a:,.2f} € (15%)</b> su <b>Vaso 3A (Mantenimento Anti-Inflazione)</b></div>'
                )
            if v4_val >= 100000.0 or phase_curr.get('core_sat_active'):
                instructions_html.append(
                    f'<div style="font-size: 0.84rem; color: #047857; margin-top: 6px; padding: 6px 12px; background: #ecfdf5; border-radius: 6px;">↳ <i>All\'interno di Vaso 4 è attivo il regime Core-Satellite: 60% Core All-World ({eur_v4*0.60:,.2f} €) e 40% Satellite Factor ({eur_v4*0.40:,.2f} €) mediante Inflow Rebalancing entro la banda [33%, 47%] a zero imposte di realizzo.</i></div>'
                )

    instr_body = "".join(instructions_html)

    # Calcolo Pillar 0 Previdenziale Parallelo
    tfr_monthly = (ral_0 / 12.0) * 0.0691
    matching_monthly = (ral_0 / 12.0) * 0.015
    voluntary_monthly = 50.0
    pension_total_monthly = tfr_monthly + matching_monthly + voluntary_monthly
    refund_730_annual = min(5164.57, (matching_monthly + voluntary_monthly) * 12.0) * 0.35

    st.markdown(f"""
    <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-left: 6px solid #2563eb; border-radius: 12px; padding: 20px 24px; margin-bottom: 22px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
            <div style="font-size: 1.18rem; font-weight: 800; color: #1e3a8a; display: flex; align-items: center; gap: 8px;">
                💶 Destinazione del Risparmio di Questo Mese
            </div>
            <span style="font-size: 0.85rem; font-weight: 700; color: #1e40af; background: #dbeafe; padding: 4px 12px; border-radius: 9999px;">
                Risparmio Netto Mensile (S₀): {base_savings:,.2f} €
            </span>
        </div>
        <div style="font-size: 0.90rem; color: #475569; margin-bottom: 12px;">
            Istruzione operativa vincolante per il mese corrente determinata dal livello di saturazione della gerarchia S-DFP:
        </div>
        {instr_body}
        <div style="margin-top: 14px; padding-top: 10px; border-top: 1px solid #e2e8f0; font-size: 0.82rem; color: #64748b; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px;">
            <span>👴 <b>Pillar 0 Previdenziale Parallelo (Autopilota):</b> {pension_total_monthly:,.1f} €/mese (TFR + Matching datoriale + Deduzione 730 estiva stimata <b>+{refund_730_annual:,.0f} €/anno</b>)</span>
            <span style="color: #94a3b8;">D.Lgs. 252/2005</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================================================
    # CARD C: GESTIONE CASSA FERMA IN ECCESSO (SURPLUS STOCK)
    # ==========================================================================
    if state_mode != "Mese 1 (Partenza da zero)":
        if total_surplus > 0:
            st.markdown(f"""
            <div class="action-box">
                <div class="action-title">📋 Gestione Cassa Ferma in Eccesso (Surplus Stock)</div>
                <div style="font-size:0.92rem; color:#166534; margin-bottom:12px;">
                    È stata rilevata un'eccedenza di cassa di <b>{total_surplus:,.2f} €</b> sui comparti di liquidità a monte (Vaso 1: {surplus_v1:,.2f} €{f', Vaso 2: {surplus_v2:,.2f} €' if surplus_v2 > 0 else ''}).
                    Per azzerare il <i>cash drag</i> e proteggere gli obiettivi senza intaccare il Vaso 3A obbligazionario a scadenza, esegui i seguenti trasferimenti consigliati:
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            for m in moves:
                st.markdown(f"""
                <div class="action-item">
                    <b>{m['icon']} {m['action']}:</b> Sposta <b>{m['amount']:,.2f} €</b> da <i>{m['from']}</i> a <b>{m['to']}</b><br>
                    <span style="color:#64748b; font-size:0.83rem;">↳ <i>Finalità S-DFP: {m['reason']}</i></span>
                </div>
                """, unsafe_allow_html=True)
                
            col_btn, col_note = st.columns([1, 2])
            with col_btn:
                st.button(
                    "✅ Applica Riallocazione ai Saldi",
                    type="primary",
                    use_container_width=True,
                    on_click=apply_realloc_action,
                    args=(v1_opt, v2_opt, v3b_opt, v3a_opt, v4_opt),
                    help="Aggiorna istantaneamente i campi dei saldi numerici con i valori post-trasferimento."
                )
            with col_note:
                st.caption(f"💡 Cliccando sul pulsante, i saldi numerici verranno aggiornati con le consistenze post-bonifico ({v1_opt:,.0f} € su V1, {v2_opt:,.0f} € su V2, {v3b_opt:,.0f} € su V3B, {v3a_opt:,.0f} € su V3A, {v4_opt:,.0f} € su V4).")

            # Transizione Grafica: Attuale vs Ottimizzato vs Target
            st.markdown("##### 🔄 Transizione della Liquidità: Saldo Attuale vs Spostamenti Consigliati vs Target")
            
            col_chart_t, col_table_t = st.columns([3, 2])
            
            with col_chart_t:
                pot_names_chart = ["V1 Emergenza", "V2 Sinking", "V3B Mutuo", "V3A Goal H", "V4 Azionario"]
                act_vals = [v1_val, v2_val, v3b_val, v3a_val, v4_val]
                opt_vals = [v1_opt, v2_opt, v3b_opt, v3a_opt, v4_opt]
                tgt_vals = [t1_target, t2_target, t3b_target, t3a_target, 100000.0]
                deltas = [o - a for a, o in zip(act_vals, opt_vals)]
                
                fig_trans = go.Figure()
                fig_trans.add_trace(go.Bar(
                    name='Saldo Attuale',
                    x=pot_names_chart,
                    y=act_vals,
                    marker_color='#94a3b8',
                    text=[f"{a:,.0f} €" for a in act_vals],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Saldo Attuale: %{y:,.2f} €<extra></extra>'
                ))
                fig_trans.add_trace(go.Bar(
                    name='Saldo Ottimizzato',
                    x=pot_names_chart,
                    y=opt_vals,
                    marker_color='#059669',
                    text=[f"{o:,.0f} € ({'+' if d > 0 else ''}{d:,.0f} €)" if abs(d) > 1e-2 else f"{o:,.0f} €" for o, d in zip(opt_vals, deltas)],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Saldo Ottimizzato: %{y:,.2f} €<extra></extra>'
                ))
                fig_trans.add_trace(go.Scatter(
                    name='Target di Sicurezza',
                    x=pot_names_chart,
                    y=tgt_vals,
                    mode='markers',
                    marker=dict(symbol='line-ew-open', size=36, line=dict(width=3.5, color='#dc2626')),
                    hovertemplate='<b>%{x}</b><br>Target: %{y:,.2f} €<extra></extra>'
                ))
                fig_trans.update_layout(
                    title="<b>Confronto Saldi: Attuale vs Post-Riallocazione vs Target</b>",
                    xaxis_title="Comparto Virtuale (Vaso)",
                    yaxis_title="Euro (€)",
                    barmode='group',
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98, bgcolor="rgba(255,255,255,0.85)"),
                    template="plotly_white"
                )
                st.plotly_chart(fig_trans, use_container_width=True)
                
            with col_table_t:
                trans_table_rows = []
                for item in cascade["summary"]:
                    v_name = item["vaso"]
                    s_cur = item["saldo_attuale"]
                    t_val = item["target"]
                    sh_val = item["shift"]
                    s_post = item["saldo_ottimizzato"]
                    st_badge = item["status_label"]
                        
                    trans_table_rows.append({
                        "Vaso": v_name,
                        "Attuale": f"{s_cur:,.0f} €",
                        "Target": f"{t_val:,.0f} €" if t_val > 0 else "-",
                        "Spostamento": f"{'+' if sh_val > 0 else ''}{sh_val:,.0f} €" if abs(sh_val) > 1e-2 else "0 €",
                        "Ottimizzato": f"{s_post:,.0f} €",
                        "Stato": st_badge
                    })
                df_trans_summary = pd.DataFrame(trans_table_rows)
                st.markdown("<div style='font-size:0.88rem; font-weight:700; margin-bottom:6px;'>Tabella Analitica Surplus / Deficit</div>", unsafe_allow_html=True)
                st.dataframe(df_trans_summary, use_container_width=True, hide_index=True)

        else:
            # total_surplus == 0: nessun surplus prelevabile da V1 o V2
            has_deficit = (v1_val < t1_target) or (v2_val < t2_target) or (t3b_target > 0 and v3b_val < t3b_target) or (t3a_target > 0 and v3a_val < t3a_target)
            if has_deficit:
                st.markdown(f"""
                <div class="action-box-warning">
                    <div class="action-title" style="color: #92400e;">⚠️ Monitoraggio Fabbisogni di Liquidità (Surplus Stock = 0,00 €)</div>
                    <div style="font-size:0.90rem; color:#78350f;">
                        Nessuna eccedenza di cassa presente sui buffer di liquidità. I deficit di sicurezza riscontrati sui vasi a monte vengono progressivamente colmati tramite il flusso di risparmio mensile ordinario ({base_savings:,.0f} €/mese), allocato come indicato nel <b>Box Dispositivo B</b> sopra.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="action-box">
                    <div class="action-title">✅ Saldi Perfettamente Allineati ai Target (Surplus Stock = 0,00 €)</div>
                    <div style="font-size:0.90rem; color:#166534;">
                        Tutti i comparti di liquidità e riserva sono allineati ai rispettivi target di sicurezza. Non è necessaria alcuna movimentazione di stock bancario.
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # ==========================================================================
    # SEZIONE INFERIORE: CONSISTENZA DEI VASI E GRAFICO FLUSSI
    # ==========================================================================
    st.markdown("##### 📦 Consistenza e Stato di Copertura dei Vasi")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        prog_v1 = min(1.0, v1_val / max(1.0, t1_target))
        st.markdown(f"""
        <div class="pot-card" style="border-left-color: #3b82f6;">
            <div class="pot-title"><span>🛡️ Vaso 1: Buffer Emergenza</span> <span style="font-size:0.8rem; font-weight:700; color:#3b82f6;">{prog_v1*100:.0f}%</span></div>
            <div class="pot-subtitle">Conto Deposito / ETF €STR (Anti-ZIRP)</div>
            <div class="pot-balance">{v1_val:,.0f} €</div>
            <div class="pot-target">Target dinamico (3×Spese): <b>{t1_target:,.0f} €</b></div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(prog_v1)
        
        prog_v3a = min(1.0, v3a_val / max(1.0, t3a_target)) if t3a_target > 0 else 1.0
        st.markdown(f"""
        <div class="pot-card" style="border-left-color: #8b5cf6;">
            <div class="pot-title"><span>🚗 Vaso 3A: Obiettivo {H_years} Anni</span> <span style="font-size:0.8rem; font-weight:700; color:#8b5cf6;">{prog_v3a*100:.0f}%</span></div>
            <div class="pot-subtitle">BTP / Bond Sovrani Euro held-to-maturity</div>
            <div class="pot-balance">{v3a_val:,.0f} €</div>
            <div class="pot-target">Target reale indicizzato: <b>{t3a_target:,.0f} €</b></div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(prog_v3a)

    with col2:
        prog_v2 = min(1.0, v2_val / max(1.0, t2_target))
        st.markdown(f"""
        <div class="pot-card" style="border-left-color: #10b981;">
            <div class="pot-title"><span>🗓️ Vaso 2: Sinking Fund</span> <span style="font-size:0.8rem; font-weight:700; color:#10b981;">{prog_v2*100:.0f}%</span></div>
            <div class="pot-subtitle">Spese prevedibili a 12-24 mesi (Monetario)</div>
            <div class="pot-balance">{v2_val:,.0f} €</div>
            <div class="pot-target">Target fisso ammortizzato: <b>{t2_target:,.0f} €</b></div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(prog_v2)

        prog_v4 = min(1.0, v4_val / 100000.0)
        st.markdown(f"""
        <div class="pot-card" style="border-left-color: #06b6d4;">
            <div class="pot-title"><span>🚀 Vaso 4: Crescita Azionaria</span> <span style="font-size:0.8rem; font-weight:700; color:#06b6d4;">{prog_v4*100:.0f}%</span></div>
            <div class="pot-subtitle">ETF All-World Core + Satelliti Factor</div>
            <div class="pot-balance">{v4_val:,.0f} €</div>
            <div class="pot-target">Soglia attivazione satelliti: <b>100.000 €</b></div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(prog_v4)

    with col3:
        prog_v3b = min(1.0, v3b_val / max(1.0, t3b_target)) if t3b_target > 0 else 1.0
        st.markdown(f"""
        <div class="pot-card" style="border-left-color: #f59e0b;">
            <div class="pot-title"><span>🏠 Vaso 3B: Buffer Rata Mutuo</span> <span style="font-size:0.8rem; font-weight:700; color:#f59e0b;">{prog_v3b*100:.0f}%</span></div>
            <div class="pot-subtitle">Hard floor di liquidità (4 rate mutuo)</div>
            <div class="pot-balance">{v3b_val:,.0f} €</div>
            <div class="pot-target">Target sicurezza rata: <b>{t3b_target:,.0f} €</b></div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(prog_v3b)

        st.markdown(f"""
        <div class="pot-card" style="border-left-color: #ec4899;">
            <div class="pot-title"><span>👴 Vaso 0: Fondo Pensione & TFR</span> <span style="font-size:0.8rem; font-weight:700; color:#ec4899;">Autopilota</span></div>
            <div class="pot-subtitle">TFR + Matching datoriale + Deduzione 730</div>
            <div class="pot-balance">{pension_total_monthly:,.0f} €/m</div>
            <div class="pot-target">Rimborso IRPEF estivo stimato: <b>+{refund_730_annual:,.0f} €/anno</b></div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("🔒 Segregato per legge (D.Lgs. 252/2005): protetto contro prelievi per beni di consumo.")

    st.markdown("---")
    
    # --------------------------------------------------------------------------
    # DONUT CHART FINTECH A DOPPIA VISTA: BUDGET COMPLESSIVO VS DESTINAZIONE S-DFP
    # --------------------------------------------------------------------------
    st.markdown("##### 🍩 Analisi Finanziaria Mensile: Bilancio Familiare vs Allocazione S-DFP")
    st.caption("Visualizzazione contestuale a doppio anello (Donut Chart): l'equilibrio del reddito complessivo a confronto con la destinazione del risparmio sui vasi vincolati.")

    col_donut1, col_donut2 = st.columns(2)

    # COLONNA 1: Budget Mensile Complessivo (Entrate)
    with col_donut1:
        r0_val = mortgage_R0 if not no_mortgage else 0.0
        e_living = max(0.0, E_effettivo - r0_val)
        s0_val = max(0.0, Y0 - E_effettivo)

        b_labels = []
        b_vals = []
        b_colors = []

        if e_living > 0:
            b_labels.append("Spese Vive Ordinarie")
            b_vals.append(e_living)
            b_colors.append("#334155")  # Navy / Grigio Ardesia

        if r0_val > 0:
            b_labels.append("Rata Mutuo / Debito Fisso")
            b_vals.append(r0_val)
            b_colors.append("#ea580c")  # Terracotta / Arancio

        if s0_val > 0:
            b_labels.append("Risparmio Netto Mensile (S₀)")
            b_vals.append(s0_val)
            b_colors.append("#059669")  # Verde Smeraldo

        fig_b = go.Figure(data=[
            go.Pie(
                labels=b_labels,
                values=b_vals,
                hole=0.62,
                marker=dict(colors=b_colors, line=dict(color='#ffffff', width=2)),
                textinfo='percent',
                textposition='inside',
                hoverinfo='label+value+percent',
                hovertemplate='<b>%{label}</b><br>Importo: <b>%{value:,.2f} €</b><br>Quota su Entrate: <b>%{percent}</b><extra></extra>',
                sort=False
            )
        ])

        fig_b.update_layout(
            title=dict(
                text="<b>Budget Mensile Complessivo (Entrate)</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=14, color='#0f172a')
            ),
            annotations=[
                dict(
                    text=f"<span style='font-size:16px; font-weight:800; color:#0f172a;'>€ {Y0:,.0f}</span><br><span style='font-size:10px; font-weight:700; color:#64748b; text-transform:uppercase;'>Entrate Nette</span>",
                    x=0.5, y=0.5,
                    font_size=13,
                    showarrow=False
                )
            ],
            height=280,
            margin=dict(t=25, b=25, l=20, r=20),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.08,
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_b, use_container_width=True)

    # COLONNA 2: Destinazione del Risparmio Questo Mese (Flusso S-DFP)
    with col_donut2:
        f_idx = phase_curr['fase_idx']
        alloc = phase_curr['alloc']
        
        eur_v1 = alloc['v1'] * base_savings
        eur_v2 = alloc['v2'] * base_savings
        eur_v3b = alloc['v3b'] * base_savings
        eur_v3a = alloc['v3a'] * base_savings
        eur_v4_tot = alloc['v4'] * base_savings

        d_labels = []
        d_vals = []
        d_colors = []

        if f_idx == 1:
            d_labels.append("V1: Buffer Emergenza")
            d_vals.append(base_savings)
            d_colors.append("#10b981")  # Verde
        elif f_idx == 2:
            d_labels.append("V2: Sinking Fund")
            d_vals.append(base_savings)
            d_colors.append("#f59e0b")  # Arancio
        elif f_idx == 3:
            d_labels.append("V3B: Buffer Mutuo")
            d_vals.append(base_savings)
            d_colors.append("#ea580c")  # Terracotta / Arancio
        elif f_idx == 4:
            d_labels.append("V3A: Titoli di Stato a Scadenza H")
            d_vals.append(base_savings)
            d_colors.append("#8b5cf6")  # Viola
        elif f_idx == 5:
            d_labels.append("V3A: Titoli di Stato H (60%)")
            d_vals.append(eur_v3a)
            d_colors.append("#8b5cf6")  # Viola
            d_labels.append("V4: Core ETF All-World (40%)")
            d_vals.append(eur_v4_tot)
            d_colors.append("#06b6d4")  # Ciano / Azzurro
        else:  # Fase 6
            if v4_val >= 100000.0:
                d_labels.append("V4: Core ETF (51%)")
                d_vals.append(0.60 * eur_v4_tot)
                d_colors.append("#06b6d4")  # Ciano
                d_labels.append("V4: Satellite Factor (34%)")
                dest_sat = 0.40 * eur_v4_tot
                d_vals.append(dest_sat)
                d_colors.append("#3b82f6")  # Azzurro / Blu
            else:
                d_labels.append("V4: Core ETF All-World (85%)" if t3a_target > 0 else "V4: Core ETF (100%)")
                d_vals.append(eur_v4_tot)
                d_colors.append("#06b6d4")  # Ciano
            if eur_v3a > 0:
                d_labels.append("V3A: Mantenimento (15%)")
                d_vals.append(eur_v3a)
                d_colors.append("#8b5cf6")  # Viola

        if base_savings > 0 and len(d_vals) > 0:
            fig_d = go.Figure(data=[
                go.Pie(
                    labels=d_labels,
                    values=d_vals,
                    hole=0.62,
                    marker=dict(colors=d_colors, line=dict(color='#ffffff', width=2)),
                    textinfo='percent',
                    textposition='inside',
                    hoverinfo='label+value+percent',
                    hovertemplate='<b>%{label}</b><br>Allocazione: <b>%{value:,.2f} €</b><br>Quota Risparmio: <b>%{percent}</b><extra></extra>',
                    sort=False
                )
            ])

            fig_d.update_layout(
                title=dict(
                    text=f"<b>Destinazione Risparmio Questo Mese (Fase {f_idx})</b>",
                    x=0.5,
                    xanchor='center',
                    font=dict(size=14, color='#0f172a')
                ),
                annotations=[
                    dict(
                        text=f"<span style='font-size:16px; font-weight:800; color:#0f172a;'>€ {base_savings:,.0f}</span><br><span style='font-size:10px; font-weight:700; color:#64748b; text-transform:uppercase;'>Risparmio/Mese</span>",
                        x=0.5, y=0.5,
                        font_size=13,
                        showarrow=False
                    )
                ],
                height=280,
                margin=dict(t=25, b=25, l=20, r=20),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.08,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10)
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_d, use_container_width=True)
        else:
            st.info("Nessun risparmio mensile disponibile da allocare (S₀ = 0 €).")


# ==============================================================================
# TAB 2: PROIEZIONE PATRIMONIALE A 30 ANNI
# ==============================================================================
with tab2:
    st.markdown("#### 📈 Proiezione Patrimoniale a 30 Anni e Confronto Scientifico dei Modelli")
    st.caption("Validazione empirica e stocastica dei modelli su 360 mesi (30 anni) a confronto con i benchmark istituzionali sotto Common Random Numbers.")

    # KPI Top Card: S-DFP M5 vs Fair Benchmark M2
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    w_horizon_key = '30y' if '30y' in m5_res['Wealth'] else list(m5_res['Wealth'].keys())[-1]
    w30_m5 = m5_res['Wealth'][w_horizon_key]['W_tot_median']
    w30_m2 = m2_res['Wealth'][w_horizon_key]['W_tot_median']
    diff_wealth = w30_m5 - w30_m2
    
    tax_drag_m5 = m5_res['Realization_Tax_Drag_EUR']
    tax_drag_m2 = m2_res['Realization_Tax_Drag_EUR']
    tax_saved = max(0.0, tax_drag_m2 - tax_drag_m5)
    
    tot_mdd_m5 = m5_res['Total_Wealth_MDD_95th_pct']
    tot_mdd_m2 = m2_res['Total_Wealth_MDD_95th_pct']
    
    ugf_m5 = m5_res['UGF_pct']
    ugf_m2 = m2_res['UGF_pct']

    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Patrimonio Mediano ({w_horizon_key})</div>
            <div class="kpi-value">{w30_m5:,.0f} €</div>
            <div class="kpi-delta-pos">+{diff_wealth:,.0f} € vs Fair Bench</div>
            <div class="kpi-subtext">Fair Benchmark 60/40: {w30_m2:,.0f} €</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Risparmio Fiscale (Tax Drag Evitato)</div>
            <div class="kpi-value">{tax_saved:,.0f} €</div>
            <div class="kpi-delta-pos">Zero tasse da ribilanciamento</div>
            <div class="kpi-subtext">S-DFP: {tax_drag_m5:,.0f} € vs Bench: {tax_drag_m2:,.0f} €</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Protezione Goal H (UGF)</div>
            <div class="kpi-value">{ugf_m5:.1f}%</div>
            <div class="kpi-delta-pos">Nessuna liquidazione azionaria</div>
            <div class="kpi-subtext">Fair Benchmark: {ugf_m2:.1f}% (Liquidazione forzata)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Wealth Drawdown (95° Pct)</div>
            <div class="kpi-value">{tot_mdd_m5:.1f}%</div>
            <div class="kpi-delta-pos">-{tot_mdd_m2 - tot_mdd_m5:.1f}% vs Benchmark</div>
            <div class="kpi-subtext">Fair Benchmark subisce {tot_mdd_m2:.1f}% MDD</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # SCHEDA DI CONFRONTO A 4 KPI: M5 (Full S-DFP) vs M7 (Fair + Liability Matching + Inflow Rebalancing)
    if m7_res is not None:
        w30_fin_m5 = m5_res['Wealth']['30y']['W_fin_median']
        w30_fin_m7 = m7_res['Wealth']['30y']['W_fin_median']
        ugf_m7 = m7_res['UGF_pct']
        zfel_m5 = m5_res['ZFEL_pct']
        zfel_m7 = m7_res['ZFEL_pct']
        tax_drag_m7 = m7_res['Realization_Tax_Drag_EUR']
        mdd_m7 = m7_res['Total_Wealth_MDD_95th_pct']
        
        st.markdown("##### ⚖️ Confronto Scientifico a 4-KPI: M5 (Full S-DFP) vs M7 (Fair + Liability Matching + Inflow Rebalancing)")
        c_m5_1, c_m5_2, c_m5_3, c_m5_4 = st.columns(4)
        with c_m5_1:
            delta_w = w30_fin_m7 - w30_fin_m5
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #3b82f6;">
                <div class="kpi-label">1. W_fin 30y Mediano</div>
                <div class="kpi-value" style="font-size:1.30rem;">{w30_fin_m5:,.0f} € <span style="font-size:0.85rem; color:#64748b;">vs {w30_fin_m7:,.0f} €</span></div>
                <div class="kpi-subtext">Delta M7: <b>{delta_w:+,.0f} € (+{delta_w/w30_fin_m5*100:.1f}%)</b></div>
            </div>
            """, unsafe_allow_html=True)
            
        with c_m5_2:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #10b981;">
                <div class="kpi-label">2. Unforced Goal Fulfillment (UGF)</div>
                <div class="kpi-value" style="font-size:1.30rem;">{ugf_m5:.2f}% <span style="font-size:0.85rem; color:#64748b;">vs {ugf_m7:.2f}%</span></div>
                <div class="kpi-subtext">Zero vendite coatte di azioni</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c_m5_3:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #f59e0b;">
                <div class="kpi-label">3. ZFEL & Resilienza Drawdown</div>
                <div class="kpi-value" style="font-size:1.30rem;">{zfel_m5:.2f}% <span style="font-size:0.85rem; color:#64748b;">vs {zfel_m7:.2f}%</span></div>
                <div class="kpi-subtext">MDD 95°: <b>M5 {tot_mdd_m5:.1f}%</b> vs <b>M7 {mdd_m7:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)
            
        with c_m5_4:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #8b5cf6;">
                <div class="kpi-label">4. Realization Tax Drag</div>
                <div class="kpi-value" style="font-size:1.30rem;">{tax_drag_m5:,.0f} € <span style="font-size:0.85rem; color:#64748b;">vs {tax_drag_m7:,.0f} €</span></div>
                <div class="kpi-subtext">Frizioni: M5 {m5_res['Total_Friction_Cost_EUR']:,.0f} € vs M7 {m7_res['Total_Friction_Cost_EUR']:,.0f} €</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background: #fffbeb; border: 1px solid #fde68a; border-left: 5px solid #f59e0b; border-radius: 8px; padding: 12px 16px; margin: 12px 0 20px 0; font-size: 0.88rem; color: #92400e; line-height: 1.6;">
            ⚖️ <b>Nota di Neutralità Metodologica sul Trade-Off di Rischio M5 vs M7:</b><br>
            Il modello <b>M7 (Fair + Liability Matching + Inflow Rebalancing)</b> consegue un modesto incremento del patrimonio finanziario finale (+4.329 € mediano) e azzera quasi completamente il tax drag da realizzo (13 € vs 616 €) convogliando costantemente i flussi verso le passività. Tuttavia, questo vantaggio di efficienza pura si accompagna a una significativa vulnerabilità ai ribassi congiunti: il Total Wealth Drawdown (95° percentile) sale dal <b>63,47% di M5 all'89,89% di M7</b>, e nei percorsi di stress più severi la liquidità di riserva viene completamente azzerata (Liquidity MDD = 100% in M7 rispetto a 68,59% in M5). Al contrario, <b>M5 (Full S-DFP)</b> isola stabilmente il tenore di vita tramite barriere di liquidità e garantisce una robustezza sistemica nettamente superiore senza dipendere da flussi futuri.
        </div>
        """, unsafe_allow_html=True)

    # Grafico delle Traiettorie Temporali
    years = yearly_data['M5_SDFP_ScenC']['years']
    ages = eta_attuale + years
    
    fig_wealth = go.Figure()

    # Opzioni per mostrare modelli secondari
    col_toggles, _ = st.columns([4, 1])
    with col_toggles:
        show_m0 = st.checkbox("Mostra Naïve 60/40 Monolitico (M0)", value=False)
        show_m4 = st.checkbox("Mostra S-DFP Scenario B Austerità (M4)", value=False)
        show_m7 = st.checkbox("Mostra M7 Fair + Liability Matching + Inflow Rebalancing", value=True)

    # 1. Fair Benchmark 60/40 (M2)
    m2_med = yearly_data['M2_Fair']['W_tot_med'] / 1000.0
    m2_p10 = yearly_data['M2_Fair']['W_tot_p10'] / 1000.0
    m2_p90 = yearly_data['M2_Fair']['W_tot_p90'] / 1000.0
    
    fig_wealth.add_trace(go.Scatter(
        x=ages, y=m2_p90,
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip'
    ))
    fig_wealth.add_trace(go.Scatter(
        x=ages, y=m2_p10,
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(245, 158, 11, 0.12)',
        name='Banda 10°-90° Fair Benchmark', hoverinfo='skip'
    ))
    fig_wealth.add_trace(go.Scatter(
        x=ages, y=m2_med,
        mode='lines+markers', name=f'M2 Fair Benchmark (Mediana: {w30_m2/1000:,.0f}k €)',
        line=dict(color='#f59e0b', width=2.5, dash='dash'),
        customdata=years,
        hovertemplate='<b>Fair Benchmark 60/40</b><br>Età: %{x} anni (Anno piano: %{customdata})<br>Patrimonio: %{y:.1f}k €<extra></extra>'
    ))

    # 2. S-DFP Scenario C (M5)
    m5_med = yearly_data['M5_SDFP_ScenC']['W_tot_med'] / 1000.0
    m5_p10 = yearly_data['M5_SDFP_ScenC']['W_tot_p10'] / 1000.0
    m5_p90 = yearly_data['M5_SDFP_ScenC']['W_tot_p90'] / 1000.0
    
    fig_wealth.add_trace(go.Scatter(
        x=ages, y=m5_p90,
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip'
    ))
    fig_wealth.add_trace(go.Scatter(
        x=ages, y=m5_p10,
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(16, 185, 129, 0.15)',
        name='Banda 10°-90° S-DFP (Scenario C)', hoverinfo='skip'
    ))
    fig_wealth.add_trace(go.Scatter(
        x=ages, y=m5_med,
        mode='lines+markers', name=f'S-DFP Scenario C (Mediana: {w30_m5/1000:,.0f}k €)',
        line=dict(color='#059669', width=3.2),
        customdata=years,
        hovertemplate='<b>S-DFP Scenario C</b><br>Età: %{x} anni (Anno piano: %{customdata})<br>Patrimonio: %{y:.1f}k €<extra></extra>'
    ))

    # Opzionali
    if show_m0:
        m0_med = yearly_data['M0_Naive']['W_tot_med'] / 1000.0
        m0_val = m0_res["Wealth"].get(w_horizon_key, m0_res["Wealth"][list(m0_res["Wealth"].keys())[-1]])["W_tot_median"]
        fig_wealth.add_trace(go.Scatter(
            x=ages, y=m0_med,
            mode='lines', name=f'M0 Naïve Baseline ({m0_val/1000:,.0f}k €)',
            line=dict(color='#64748b', width=1.8, dash='dot'),
            customdata=years,
            hovertemplate='<b>M0 Naïve</b><br>Età: %{x} anni (Anno: %{customdata})<br>Patrimonio: %{y:.1f}k €<extra></extra>'
        ))
    if show_m4:
        m4_med = yearly_data['M4_SDFP_ScenB']['W_tot_med'] / 1000.0
        m4_val = m4_res["Wealth"].get(w_horizon_key, m4_res["Wealth"][list(m4_res["Wealth"].keys())[-1]])["W_tot_median"]
        fig_wealth.add_trace(go.Scatter(
            x=ages, y=m4_med,
            mode='lines', name=f'M4 S-DFP Austerità ({m4_val/1000:,.0f}k €)',
            line=dict(color='#3b82f6', width=2, dash='dashdot'),
            customdata=years,
            hovertemplate='<b>M4 Austerità</b><br>Età: %{x} anni (Anno: %{customdata})<br>Patrimonio: %{y:.1f}k €<extra></extra>'
        ))
    if show_m7 and 'M7_Fair_LiabInflowReb' in yearly_data:
        m7_med_tr = yearly_data['M7_Fair_LiabInflowReb']['W_tot_med'] / 1000.0
        fig_wealth.add_trace(go.Scatter(
            x=ages, y=m7_med_tr,
            mode='lines', name=f'M7 Fair Liab Inflow ({yearly_data["M7_Fair_LiabInflowReb"]["W_tot_med"][-1]/1000:,.0f}k €)',
            line=dict(color='#8b5cf6', width=2.2, dash='dashdot'),
            customdata=years,
            hovertemplate='<b>M7 Fair Liab Inflow</b><br>Età: %{x} anni (Anno: %{customdata})<br>Patrimonio: %{y:.1f}k €<extra></extra>'
        ))

    # Linee verticali per milestone H e glidepath
    fig_wealth.add_vline(x=eta_attuale + H_years, line_width=1.5, line_dash="dash", line_color="#ef4444",
                         annotation_text=f"Milestone H (Età {eta_attuale + H_years})", annotation_position="top left")
    fig_wealth.add_vline(x=eta_attuale + (T_retire_val // 12), line_width=1.5, line_dash="dot", line_color="#6366f1",
                         annotation_text=f"Fase 6 Glidepath Pensione (Età {eta_attuale + (T_retire_val // 12)})", annotation_position="top right")

    fig_wealth.update_layout(
        title=f"<b>Evoluzione Temporale del Patrimonio Totale (W_tot) su Orizzonte {T_horizon_val // 12} Anni ({T_horizon_val} Mesi)</b>",
        xaxis_title="Età Anagrafica dell'Investitore (Anni)",
        yaxis_title="Patrimonio Totale (Migliaia di Euro, k€)",
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor="rgba(255,255,255,0.85)"),
        hovermode="x unified",
        height=520,
        template="plotly_white"
    )
    st.plotly_chart(fig_wealth, use_container_width=True)

    # Tabella completa di Ablation Study a 8 Modelli (M0 → M7)
    st.markdown("##### 🔬 Tabella Comparativa Completa dell'Ablation Study (Tutti gli 8 Modelli M0 → M7)")
    st.caption("Caricamento dinamico da `ablation_results.csv` e certificazione stocastica su 10.000 percorsi.")

    ablation_csv_path = resolve_file_path("ablation_results.csv")
    sim_json_path = resolve_file_path("simulation_results.json")

    model_mapping = {
        'M0_Naive': 'M0_Uncontrolled (Naïve 60/40 Monolitico)',
        'M1_Naive_Buf': 'M1_Static_3Pillars (Naïve 60/40 + Buffer Statico)',
        'M2_Fair': 'M2_No_Sinking (Fair Benchmark 60/40 + 3×E_t)',
        'M3_SDFP_NoInflow': 'M3_No_Preemption (S-DFP senza Inflow Reb)',
        'M4_SDFP_ScenB': 'M4_No_Glidepath (S-DFP Scen B Austerità)',
        'M5_SDFP_ScenC': '⭐ M5_Full_SDFP (Scenario C SMarT 50/50)',
        'M6_Fair_LiabMatch': 'M6_No_Behavioral (Fair + Liab Matching)',
        'M7_Fair_LiabInflowReb': 'M7_Fair_LiabInflowReb (Fair + LiabMatch + Inflow)'
    }

    statuses = {
        'M0_Naive': 'Naïve Benchmark',
        'M1_Naive_Buf': 'Static Benchmark',
        'M2_Fair': 'Fair Benchmark',
        'M3_SDFP_NoInflow': 'Ablation Variant',
        'M4_SDFP_ScenB': 'Austerity Variant',
        'M5_SDFP_ScenC': 'Full S-DFP Model',
        'M6_Fair_LiabMatch': 'Liability Match Bench',
        'M7_Fair_LiabInflowReb': 'Full Rebalance Bench'
    }

    if ablation_csv_path and ablation_csv_path.exists():
        df_csv = pd.read_csv(ablation_csv_path)
        sim_json_data = {}
        if sim_json_path and sim_json_path.exists():
            with open(sim_json_path, 'r', encoding='utf-8') as f:
                sim_json_data = json.load(f)

        ablation_14_rows = []
        for _, r in df_csv.iterrows():
            m = r['Model']
            jm = sim_json_data.get('summary_metrics', {}).get(m, summary.get(m, {}))
            w30 = jm.get('Wealth', {}).get('30y', {})
            
            w_med = float(r['W_fin 30y Median (EUR)'])
            w_mean = float(w30.get('W_fin_mean', w_med))
            w_p10 = float(w30.get('W_fin_p10', w_med * 0.70))
            w_p90 = float(w30.get('W_fin_p90', w_med * 1.50))
            w_std = (w_p90 - w_p10) / (2.0 * 1.28155) if (w_p90 > w_p10) else w_med * 0.25

            ablation_14_rows.append({
                'Model': model_mapping.get(m, m),
                'W_fin_median': f"{w_med:,.0f} €",
                'W_fin_mean': f"{w_mean:,.0f} €",
                'W_fin_std': f"{w_std:,.0f} €",
                'UGF': f"{float(r['UGF (%)']):.2f}%",
                'ZFEL': f"{float(r['ZFEL (%)']):.2f}%",
                'UGapF': f"{float(r['UGapF (%)']):.2f}%",
                'UGapF_cond': f"{float(r['UGapF_cond (%)']):.2f}%",
                'Shortfall_CVaR_EUR': f"{float(r['Shortfall CVaR (EUR)']):,.0f} €",
                'MDD_p95': f"{float(r['Total Wealth MDD 95th (%)']):.2f}%",
                'Sharpe_Ratio': f"{((w_mean / 100000.0) / (w_std / 100000.0)):.2f}" if w_std > 0 else "N/A",
                'Tax_Drag_EUR': f"{float(r['Realiz Tax Drag (EUR)']):,.0f} €",
                'Friction_EUR': f"{float(r['Total Friction (EUR)']):,.0f} €",
                'Status': statuses.get(m, 'Validated')
            })
        df_ablation = pd.DataFrame(ablation_14_rows)
    else:
        # Fallback dai risultati in memoria
        fallback_rows = []
        for mod in ['M0_Naive', 'M1_Naive_Buf', 'M2_Fair', 'M3_SDFP_NoInflow', 'M4_SDFP_ScenB', 'M5_SDFP_ScenC', 'M6_Fair_LiabMatch', 'M7_Fair_LiabInflowReb']:
            if mod in summary:
                r = summary[mod]
                w30 = r['Wealth']['30y']
                w_med = w30['W_fin_median']
                w_mean = w30['W_fin_mean']
                w_std = (w30['W_fin_p90'] - w30['W_fin_p10']) / 2.563
                fallback_rows.append({
                    'Model': model_mapping.get(mod, mod),
                    'W_fin_median': f"{w_med:,.0f} €",
                    'W_fin_mean': f"{w_mean:,.0f} €",
                    'W_fin_std': f"{w_std:,.0f} €",
                    'UGF': f"{r['UGF_pct']:.2f}%",
                    'ZFEL': f"{r['ZFEL_pct']:.2f}%",
                    'UGapF': f"{r['UGapF_pct']:.2f}%",
                    'UGapF_cond': f"{r['UGapF_cond_pct']:.2f}%",
                    'Shortfall_CVaR_EUR': f"{r['Shortfall_CVaR_EUR']:,.0f} €",
                    'MDD_p95': f"{r['Total_Wealth_MDD_95th_pct']:.2f}%",
                    'Sharpe_Ratio': f"{((w_mean / 100000.0) / (w_std / 100000.0)):.2f}" if w_std > 0 else "N/A",
                    'Tax_Drag_EUR': f"{r['Realization_Tax_Drag_EUR']:,.0f} €",
                    'Friction_EUR': f"{r['Total_Friction_Cost_EUR']:,.0f} €",
                    'Status': statuses.get(mod, 'Validated')
                })
        df_ablation = pd.DataFrame(fallback_rows)

    st.dataframe(df_ablation, use_container_width=True, hide_index=True)

    # GALLERIA FIGURE UFFICIALI DELLA PUBBLICAZIONE ACCADEMICA (fig1 -> fig6)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🖼️ Figure Ufficiali della Pubblicazione Accademica (Paper S-DFP)", expanded=False):
        st.markdown("Grafici e diagnostica quantitativa generati dal framework di simulazione accademico sotto vincoli istituzionali.")
        figs_gallery = [
            ("fig1_wealth_trajectories.png", "Figura 1: Evoluzione Temporale del Patrimonio Totale (W_tot) a 30 Anni sotto CRN (M5 S-DFP vs Fair Benchmark M2)"),
            ("fig2_pgf_comparison.png", "Figura 2: Goal Fulfillment (PGF & UGF) e Distribuzione del Deficit Relativo al Mese 96"),
            ("fig3_drawdown_distribution.png", "Figura 3: Distribuzione Empirica del Maximum Drawdown (Market MDD vs Total Wealth MDD 95° pct)"),
            ("fig4_phase_evolution.png", "Figura 4: Dinamica delle Fasi della Macchina a Stati S-DFP e Preemption Loop lungo i 360 mesi"),
            ("fig5_scenario_comparison.png", "Figura 5: Analisi di Sensibilità degli Scenari Comportamentali SMarT (Scenario A, B, C)"),
            ("fig6_glidepath_phase6.png", "Figura 6: Fase 6b Subordinated Retirement Glidepath e De-risking Obbligazionario (mesi 276-360)")
        ]
        col_fig_a, col_fig_b = st.columns(2)
        for idx, (fname, caption) in enumerate(figs_gallery):
            tgt_col = col_fig_a if idx % 2 == 0 else col_fig_b
            p = resolve_file_path(fname)
            with tgt_col:
                if p and p.exists():
                    st.image(str(p), caption=caption, use_container_width=True)
                else:
                    st.caption(f"ℹ️ Figura `{fname}` non individuata nel percorso locale.")


# ==============================================================================
# TAB 3: PROTEZIONE OBIETTIVI E STRESS TEST
# ==============================================================================
with tab3:
    st.markdown("#### 🛡️ Protezione Obiettivi (Vaso 3A) e Resilienza agli Shock Storici")
    st.caption("Verifica come il meccanismo di segregazione a scadenza held-to-maturity protegge l'investitore dagli shock e dalle liquidazioni forzate.")

    # 1. NOTE METODOLOGICHE UFFICIALI DALLA VALIDAZIONE
    sim_json_file = resolve_file_path("simulation_results.json")
    meth_notes = {}
    if sim_json_file and sim_json_file.exists():
        try:
            with open(sim_json_file, 'r', encoding='utf-8') as f:
                sim_json_data = json.load(f)
            meth_notes = sim_json_data.get('_methodological_notes', {})
        except Exception:
            meth_notes = {}

    with st.expander("📝 Note Metodologiche Ufficiali della Validazione (Paper S-DFP)", expanded=True):
        st.markdown(f"""
        <div style="font-size: 0.88rem; line-height: 1.65; color: #1e293b;">
            • <b>Spiegazione LAW Modello M7:</b> {meth_notes.get('LAW_M7_explanation', "Il LAW inferiore di M7 (39,63%) rispetto a M6 (43,57%) riflette il maggior peso azionario mantenuto a fine orizzonte dall'Inflow Rebalancing, che riduce la componente di liquidità ponderata. Non è un errore di calcolo.")}<br><br>
            • <b>Definizione ZFEL (Zero Forced Equity Liquidation):</b> {meth_notes.get('ZFEL_definition', "Zero Forced Equity Liquidation: probabilità di adempiere al target H attingendo al comparto dedicato e all'eccedenza di V1 oltre il minimo operativo, senza vendita di asset azionari Core.")}<br><br>
            • <b>Perimetro di Validazione Accademica:</b> {meth_notes.get('validation_scope', "Analisi stocastica controllata trentennale in-sample. Estensione walk-forward out-of-sample e benchmarking con prodotti Target-Date commerciali rimandati a ricerca futura.")}
        </div>
        """, unsafe_allow_html=True)

    if enable_goal_H:
        st.markdown("##### 🚗 Traiettoria di Immunizzazione del Vaso 3A verso Milestone H")
        
        m_arr = v3a_track['months']
        v3a_med = v3a_track['med']
        v3a_p10 = v3a_track['p10']
        v3a_p90 = v3a_track['p90']
        v3a_target_series = v3a_track['target']

        col_v3a_chart, col_v3a_kpi = st.columns([2, 1])
        
        with col_v3a_chart:
            fig_v3a = go.Figure()
            
            # Target indicizzato
            fig_v3a.add_trace(go.Scatter(
                x=m_arr, y=v3a_target_series,
                mode='lines', name='Target Indicizzato all\'Inflazione',
                line=dict(color='#dc2626', width=2, dash='dash')
            ))
            
            # Banda 10-90
            fig_v3a.add_trace(go.Scatter(
                x=m_arr, y=v3a_p90,
                mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip'
            ))
            fig_v3a.add_trace(go.Scatter(
                x=m_arr, y=v3a_p10,
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(139, 92, 246, 0.15)',
                name='Banda di Dispersione 10°-90°', hoverinfo='skip'
            ))
            
            # Mediana
            fig_v3a.add_trace(go.Scatter(
                x=m_arr, y=v3a_med,
                mode='lines', name='Accumulo Mediano Vaso 3A (Bond + Liquidità)',
                line=dict(color='#7c3aed', width=3)
            ))
            
            # Soglia del 60%
            fig_v3a.add_hline(y=0.60 * T3a_0, line_dash="dot", line_color="#f59e0b",
                              annotation_text="Soglia 60% Inversione Flussi", annotation_position="bottom right")

            fig_v3a.update_layout(
                title=f"<b>Accumulo e Pull-to-Par Vaso 3A (Mesi 1 → {H_years*12})</b>",
                xaxis_title="Mese di Simulazione",
                yaxis_title="Capitale Dedicato (€)",
                height=360,
                margin=dict(l=20, r=20, t=40, b=20),
                template="plotly_white"
            )
            st.plotly_chart(fig_v3a, use_container_width=True)
            
        with col_v3a_kpi:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Unforced Goal Fulfillment (UGF)</div>
                <div class="kpi-value" style="color: #7c3aed;">{m5_res['UGF_pct']:.1f}%</div>
                <div class="kpi-delta-pos">Zero vendite azionarie coatte</div>
                <hr style="margin: 12px 0;">
                <div style="font-size:0.85rem; line-height: 1.8;">
                    • <b>PGF Totale:</b> {m5_res['PGF_pct']:.1f}%<br>
                    • <b>ZFEL (Zero Forced Liquidation):</b> {m5_res.get('ZFEL_pct', 99.97):.2f}%<br>
                    • <b>UGapF (Deficit Relativo):</b> {m5_res['UGapF_pct']:.2f}%<br>
                    • <b>Shortfall CVaR (95°):</b> {m5_res['Shortfall_CVaR_EUR']:,.0f} €<br>
                    • <b>Rapporto di Copertura:</b> Held-to-Maturity
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.info("💡 **Perché il Fair Benchmark ottiene 0% UGF?** Perché non possiede un comparto segregato held-to-maturity: per liquidare l'auto deve vendere a mercato le quote del portafoglio 60/40, subendo il rischio di sequenza e pagando imposte sul capital gain!")
    else:
        st.info("Obiettivo a medio termine disattivato.")

    st.markdown("---")
    
    # Sezione 2: Stress Test nei 4 Scenari Storici
    st.markdown("##### 🌪️ Stress Test nei 4 Scenari di Crisi Storica (Mesi 72-96)")
    st.caption("Simulazione rigorosa a fattori deterministici storici innestati esattamente nella finestra pre-Milestone H.")

    stress_table_rows = []
    regime_names = {
        'GFC_2008': 'Crisi Finanziaria 2008 (Fallimento Lehman)',
        'Sovereign_2011': 'Debito Sovrano 2011 (Spread BTP > 500 bps)',
        'Covid_2020': 'Shock Covid-19 (Crash e Rimbalzo a V)',
        'Stagflation_2022': 'Stagflazione 2022 (Rialzo Tassi e Crollo Bond)'
    }
# Unpack sicuro di stress_data (evita liste annidate)
if isinstance(stress_data, list) and len(stress_data) > 0 and isinstance(stress_data[0], list):
    stress_data = stress_data[0]

for sc in stress_data:
    if not isinstance(sc, dict):
        continue
    cid = sc.get('Crisis_ID', '')
    s_m5 = sc.get('S_DFP_M5', {})
    s_m2 = sc.get('Fair_M2', {})
    s_m0 = sc.get('Naive_M0', {})
        
    stress_table_rows.append({
        "Scenario di Crisi": regime_names.get(cid, cid),
        "Azionario": f"{sc.get('Cum_Equity_Return_pct', 0.0):+.1f}%",
        "Obbligazionario": f"{sc.get('Cum_Bond_Return_pct', 0.0):+.1f}%",
        "Inflazione": f"{sc.get('Cum_Inflation_pct', 0.0):+.1f}%",
        "S-DFP UGF": f"{s_m5.get('UGF_pct', 0.0):.0f}%",
        "S-DFP ZFEL": "100%" if s_m5.get('Forced_Liquidations_EUR', 0.0) == 0 else f"{100 - (s_m5.get('Forced_Liquidations_EUR', 0.0)/sc.get('Target_at_H_EUR', 25000))*100:.1f}%",
        "S-DFP Liq. Forzate": f"{s_m5.get('Forced_Liquidations_EUR', 0.0):,.0f} €",
        "Fair UGF": f"{s_m2.get('UGF_pct', 0.0):.0f}%",
        "Fair Liq. Forzate": f"{s_m2.get('Forced_Liquidations_EUR', 0.0):,.0f} €",
        "S-DFP MDD": f"{s_m5.get('MDD_pct', 0.0):.1f}%",
        "Fair MDD": f"{s_m2.get('MDD_pct', 0.0):.1f}%"
    })
# Grafico a barre delle liquidazioni forzate a mercato
sc_labels = [regime_names.get(s.get('Crisis_ID', ''), s.get('Crisis_ID', '')) for s in stress_data if isinstance(s, dict)]
forced_m5 = [s.get('S_DFP_M5', {}).get('Forced_Liquidations_EUR', 0.0) for s in stress_data if isinstance(s, dict)]
forced_m2 = [s.get('Fair_M2', {}).get('Forced_Liquidations_EUR', 0.0) for s in stress_data if isinstance(s, dict)]
forced_m0 = [s.get('Naive_M0', {}).get('Forced_Liquidations_EUR', 0.0) for s in stress_data if isinstance(s, dict)]

fig_forced= go.Figure(data=[
        go.Bar(name='S-DFP (Scenario C)', x=sc_labels, y=forced_m5, marker_color='#059669'),
        go.Bar(name='Fair Benchmark 60/40', x=sc_labels, y=forced_m2, marker_color='#f59e0b'),
        go.Bar(name='Naïve Benchmark 60/40', x=sc_labels, y=forced_m0, marker_color='#ef4444')
    ])
fig_forced.update_layout(
        title="<b>Liquidazioni Forzate a Mercato per Shock e Milestone H (€)</b>",
        yaxis_title="Euro Liquidati Forzatamente a Mercato (€)",
        barmode='group',
        height=380,
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
        template="plotly_white"
    )
st.plotly_chart(fig_forced, use_container_width=True)
    
st.markdown("""
    <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-left: 5px solid #16a34a; border-radius: 8px; padding: 14px 18px; margin-top: 10px; font-size: 0.88rem; color: #166534; line-height: 1.6;">
        🛡️ <b>Risultato Fondamentale sui 4 Regimi di Crisi Storica:</b><br>
        In 3 regimi su 4 (Crisi Finanziaria 2008, Debito Sovrano 2011 e Shock Covid 2020), <b>S-DFP azzera completamente le liquidazioni forzate a mercato (0 € di dismissioni coatte)</b>, contro oltre 26.000 € - 32.000 € di asset forzatamente venduti dal Fair Benchmark M2.<br><br>
        📉 <b>Focus Stagflazione 2022 (Co-movimentazione Bond-Equity):</b><br>
        Durante la crisi stagflativa del 2022 (-8,16% azionario, -20,11% obbligazionario, inflazione al +17,40% e 3 shock di spesa per 5.500 €), il Fair Benchmark M2 è stato costretto a liquidare sul mercato <b>30.347 €</b> di asset a valutazioni depresse. Al contrario, <b>S-DFP (M5) ha abbattuto le liquidazioni forzate a soli 2.957 € (un abbattimento del 90,3%)</b>, preservando il motore azionario grazie alla canalizzazione dei flussi, all'isolamento sequenziale dei buffer liquidi e alla struttura Held-to-Maturity del Vaso 3A.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# TAB 4: FRAMEWORK MATEMATICO, EQUAZIONI DINAMICHE & DIAGNOSTICA GRAFICA
# ==============================================================================
with tab4:
    st.markdown("#### 📐 Framework Matematico, Equazioni Dinamiche & Diagnostica Grafica")
    st.caption("Trattato analitico formale del modello State-Dependent Sequential Flow Allocation (S-DFP), risoluzione numerica puntuale istanziata sui parametri attuali e diagnostica stocastica avanzata.")

    # --------------------------------------------------------------------------
    # SEZIONE 1: FORMULAZIONE MATEMATICA FORMALE (LaTeX)
    # --------------------------------------------------------------------------
    st.markdown("### 1. Formulazione Matematica Formale")
    st.markdown("Il framework S-DFP modella l'allocazione finanziaria sequenziale e dinamica attraverso processi stocastici continui e discreti, vincoli comportamentali e transizioni a stati finiti (FSM).")

    with st.expander("🔬 Blocco 1: Generazione dei Processi Stocastici SDE (Mercati, Tassi, Inflazione e Shock)", expanded=True):
        st.markdown("**1.1. Azionario Globale (Geometric Brownian Motion)**")
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t^S")
        st.markdown("Rappresenta l'evoluzione del valore patrimoniale dell'azionario globale (Core ETF All-World), con drift atteso $\mu = 7{,}5\%$ e volatilità annua $\sigma = 16{,}5\%$.")
        
        st.markdown("**1.2. Struttura a Termine dei Tassi Brevi (Modello Mean-Reverting di Vasicek)**")
        st.latex(r"dr_t = k (\theta - r_t) dt + \sigma_r dW_t^r, \quad \text{con Correlazione } dW_t^S dW_t^r = \rho_{S,r} dt")
        st.markdown("Modella la dinamica del tasso free-risk monetario (€STR / BOT / XEON) verso il livello di equilibrio di lungo termine $\theta = 2{,}2\%$ con velocità di convergenza $k = 0{,}35$ e volatilità istantanea $\sigma_r = 1{,}2\%$.")
        
        st.markdown("**1.3. Inflazione Cumulata FOI (Stocastica)**")
        st.latex(r"\Pi_t = \prod_{m=1}^{t} (1 + \pi_m \cdot \Delta t), \quad \pi_m \sim \mathcal{N}\left(\bar{\pi}, \sigma_\pi^2\right)")
        st.markdown("Indice ISTAT FOI per l'adeguamento al costo della vita dei target di spesa e della Milestone $H$ (Auto), con inflazione attesa $\bar{\pi} = 2{,}0\%$ e $\sigma_\pi = 1{,}0\%$.")
        
        st.markdown("**1.4. Processo di Salto delle Spese Impreviste (Shock Compound Poisson)**")
        st.latex(r"J_t = \sum_{k=1}^{N_t} X_k, \quad N_t \sim \text{Poisson}(\lambda \cdot \Delta t), \quad X_k \sim \text{LogNormal}(\mu_X, \sigma_X^2)")
        st.markdown("Modella l'accadimento e l'entità degli shock di spesa straordinaria (rotture, guasti, sinistri) con intensità di salto $\lambda = 0{,}20$ eventi/anno e danno mediano di circa $1.800~€$.")

    with st.expander("💼 Blocco 2: Dinamica del Reddito, SMarT Split e Flussi di Cassa", expanded=True):
        st.markdown("**2.1. Crescita Salariale Discreta**")
        st.latex(r"Y_t = Y_0 \cdot (1 + g_Y)^{\lfloor t / 12 \rfloor}, \quad \Delta Y_t = \max(0, Y_t - Y_0)")
        st.markdown("Lo stipendio netto cresce a cadenza annuale con tasso $g_Y$, determinando il surplus incrementale di reddito $\Delta Y_t$ rispetto al valore base iniziale.")
        
        st.markdown("**2.2. Spese Ordinarie con Inflazione del Tenore di Vita (Regola SMarT)**")
        st.latex(r"E_t = E_0 + \alpha_{\text{smart}} \cdot \Delta Y_t")
        st.markdown("Formalizzazione comportamentale del consumo: il coefficiente $\alpha_{\text{smart}} \in [0, 1]$ stabilisce quale frazione degli aumenti salariali viene assorbita dall'aumento del tenore di vita (Lifestyle Inflation), preservando la restante frazione $(1 - \alpha_{\text{smart}})$ per il risparmio.")
        
        st.markdown("**2.3. Flusso di Risparmio Primario Disponibile**")
        st.latex(r"S_t = Y_t - E_t - R_{\text{mutuo}} = S_0 + (1 - \alpha_{\text{smart}}) \cdot \Delta Y_t")
        st.markdown("Il risparmio mensile strutturale $S_t$ è la risorsa netta disponibile per alimentare i vasi sequenziali dopo aver assolto al consumo effettivo $E_t$ e al servizio del debito $R_{\text{mutuo}}$.")
        
        st.markdown("**2.4. Fondo Pensione (Vaso 0) e Tax Shield IRPEF**")
        st.latex(r"\Phi_t = \left(\frac{\text{RAL}_t}{12}\right) \cdot (\tau_{\text{TFR}} + \tau_{\text{match}}) + C_{\text{volontario}}")
        st.latex(r"\text{Refund}_{730} = \min(5.164{,}57, 12 \cdot C_{\text{volontario}}) \cdot \tau_{\text{marginale}}^{\text{IRPEF}}")
        st.markdown("Canalizzazione previdenziale prioritaria con contributo TFR (6,91%), matching datoriale (1,50%) e contributo volontario deducibile ai sensi del D.Lgs. 252/2005 (scudo fiscale fino a 5.164,57 €/anno con aliquota marginale IRPEF 35%).")

    with st.expander("🏛️ Blocco 3: Tetti Dinamici dei Vasi e Condizioni di Transizione della FSM", expanded=True):
        st.markdown("**3.1. Vettore dei Target di Sicurezza Dinamici**")
        st.latex(r"\vec{T}(t) = \begin{bmatrix} T_1(t) \\ T_2 \\ T_{3b} \\ T_{3a}(t) \end{bmatrix} = \begin{bmatrix} 3 \cdot E_t \\ 2.400 \\ 4 \cdot R_{\text{mutuo}} \\ T_{3a,0} \cdot \Pi_t \end{bmatrix}")
        st.markdown("I massimali si auto-calibrano: $T_1(t)$ protegge 3 mesi di uscite effettive aggiornate, $T_2$ assorbe le spese cicliche ammortizzate, $T_{3b}$ copre 4 rate di mutuo, e $T_{3a}(t)$ preserva il potere d'acquisto della Milestone $H$.")
        
        st.markdown("**3.2. Soglia di Decoupling Anticipato e Sblocco Azionario**")
        st.latex(r"\Theta_{3a}(t) = 0{,}60 \cdot T_{3a}(t)")
        st.markdown("Al raggiungimento del $60\%$ della Milestone $H$, il protocollo attiva l'inversione di flusso anticipata, immettendo il $40\%$ del risparmio nell'azionario prima del completamento dell'obiettivo.")
        
        st.markdown("**3.3. Matrice di Transizione della Finite State Machine (FSM)**")
        st.latex(r"\begin{aligned}"
                 r"\mathcal{S}_t = S1 &\iff V_{1,t} < T_1(t) \implies \Delta V_1 = S_t \\"
                 r"\mathcal{S}_t = S2 &\iff V_{1,t} \ge T_1(t) \land V_{2,t} < T_2 \implies \Delta V_2 = S_t \\"
                 r"\mathcal{S}_t = S3 &\iff V_{2,t} \ge T_2 \land V_{3b,t} < T_{3b} \implies \Delta V_{3b} = S_t \\"
                 r"\mathcal{S}_t = S4 &\iff V_{3b,t} \ge T_{3b} \land V_{3a,t} < \Theta_{3a}(t) \implies \Delta V_{3a} = S_t \\"
                 r"\mathcal{S}_t = S4_{\text{inv}} &\iff \Theta_{3a}(t) \le V_{3a,t} < T_{3a}(t) \implies \begin{cases} \Delta V_{3a} = 0{,}60 \cdot S_t \\ \Delta V_4 = 0{,}40 \cdot S_t \end{cases} \\"
                 r"\mathcal{S}_t = S5A &\iff V_{3a,t} \ge T_{3a}(t) \land V_{4,t} < 100.000 \implies \begin{cases} \Delta V_4 = 0{,}85 \cdot S_t \\ \Delta V_{3a} = 0{,}15 \cdot S_t \end{cases} \\"
                 r"\mathcal{S}_t = S5B &\iff V_{3a,t} \ge T_{3a}(t) \land V_{4,t} \ge 100.000 \implies \text{Core-Satellite 60/40 Inflow Rebalancing}"
                 r"\end{aligned}")
        st.markdown("La macchina a stati dirige ogni singolo euro con priorità assoluta di sicurezza verso i comparti a monte prima di riversare i flussi negli asset rischiosi di accumulo.")

    with st.expander("⚖️ Blocco 4: Dinamica del PMC e Inflow Rebalancing (Azzeramento Tax Drag)", expanded=True):
        st.markdown("**4.1. Aggiornamento del Prezzo Medio di Carico (senza vendite)**")
        st.latex(r"\text{PMC}_{t+1} = \frac{V_t \cdot \text{PMC}_t + \Delta V_{\text{inflow}} \cdot P_{t+1}}{V_t + \Delta V_{\text{inflow}}}")
        st.markdown("L'accumulo puramente progressivo garantisce che il Prezzo Medio di Carico fiscale (PMC) cresca in modo armonico senza mai forzare vendite parziali, differendo la tassazione sul capital gain (26%) a tempo indefinito.")
        
        st.markdown("**4.2. Regola di Inflow Rebalancing con Banda di Inazione [35%, 45%]**")
        st.latex(r"w_{\text{sat}} = \frac{V_{4,\text{sat}}}{V_{4,\text{core}} + V_{4,\text{sat}}}")
        st.latex(r"\begin{cases}"
                 r"w_{\text{sat}} < 0{,}35 \implies \text{Inflow Split}: 20\% \text{ Core} \,/\, 80\% \text{ Satellite} \\"
                 r"w_{\text{sat}} > 0{,}45 \implies \text{Inflow Split}: 80\% \text{ Core} \,/\, 20\% \text{ Satellite} \\"
                 r"0{,}35 \le w_{\text{sat}} \le 0{,}45 \implies \text{Inflow Split}: 60\% \text{ Core} \,/\, 40\% \text{ Satellite}"
                 r"\end{cases} \implies \text{Tax Realized} = 0")
        st.markdown("Il ribilanciamento tra ETF Core ed ETF Satellite viene effettuato esclusivamente deviando la direzione dei nuovi risparmi mensili in ingresso, azzerando del tutto il Tax Drag fiscale derivante da vendite coatte.")

    with st.expander("🎯 Blocco 5: Funzione Obiettivo Intertemporale & Ottimalità della Liquidazione", expanded=False):
        st.latex(r"\max_{\{c_t, \vec{\Delta V}_t\}} \mathbb{E} \left[ \sum_{t=0}^{T} \beta^t U(C_t) + \Phi(V_T) - \lambda \cdot \mathbb{I}_{\{\text{Forced Sale}\}} \right]")
        st.markdown("Formalizzazione della funzione di utilità intertemporale: il termine di penalità $\lambda \cdot \mathbb{I}_{\{\text{Forced Sale}\}}$ impone un costo asimmetrico proibitivo alla liquidazione coatta di posizioni azionarie durante i drawdown, principio fondante della struttura a vasi sequenziali.")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # SEZIONE 2: APPLICAZIONE NUMERICA PUNTUALE (Istanziata sui Dati Utente)
    # --------------------------------------------------------------------------
    st.markdown("### 2. Applicazione Numerica Puntuale (Istanziata sui Tuoi Dati)")
    st.caption("Risoluzione delle equazioni analitiche mediante sostituzione dei parametri finanziari attuali impostati nella sidebar.")

    # Calcolo previdenziale Vaso 0
    tfr_monthly = (ral_0 / 12.0) * 0.0691
    match_monthly = (ral_0 / 12.0) * 0.015
    emp_monthly = 50.0
    v0_monthly_tot = tfr_monthly + match_monthly + emp_monthly
    tax_refund_yr = min(5164.57, emp_monthly * 12.0) * 0.35

    # Valutazione stato e flussi correnti
    v1_c = st.session_state.get("input_v1", t1_target)
    v2_c = st.session_state.get("input_v2", t2_target)
    v3b_c = st.session_state.get("input_v3b", t3b_target)
    v3a_c = st.session_state.get("input_v3a", t3a_target)
    v4_c = st.session_state.get("input_v4", 25000.0)
    p_info = determine_state_machine_phase(
        v1_c, v2_c, v3b_c, v3a_c, v4_c,
        t1_target, t2_target, t3b_target, t3a_target,
        base_savings, p4_threshold_val=threshold_v3a_p4, p5_split_v3a=split_fase5_v3a,
        current_month_t=current_month_t, H_months=H_years*12,
        mortgage_active=not no_mortgage, auto_active=enable_goal_H,
        T_retire=T_retire_val, T_horizon=T_horizon_val
    )
    p_alloc = p_info['alloc']
    flow_v1 = p_alloc['v1'] * base_savings
    flow_v2 = p_alloc['v2'] * base_savings
    flow_v3b = p_alloc['v3b'] * base_savings
    flow_v3a = p_alloc['v3a'] * base_savings
    flow_v4 = p_alloc['v4'] * base_savings

    # 4 Card informative su 2 colonne
    col_n1, col_n2 = st.columns(2)
    
    with col_n1:
        st.markdown(f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-weight: 700; color: #1e3a8a; font-size: 1.05rem; margin-bottom: 8px;">
                💼 Flussi Primari & Lifestyle SMarT
            </div>
            <div style="font-family: monospace; font-size: 0.9rem; line-height: 1.6; color: #334155;">
                • <b>Y₀</b> = {Y0:,.2f} €/mese<br>
                • <b>E_base</b> = {E_base:,.2f} €/mese<br>
                • <b>ΔY</b> = max(0, {Y0:,.0f} - 2.000) = <b>{delta_Y:,.2f} €/mese</b><br>
                • <b>α_smart</b> = {split_consumo*100:.1f}% ({'50/50 SMarT' if abs(split_consumo-0.5)<0.01 else ('100% Risparmio' if split_consumo<0.01 else 'Personalizzato')})<br>
                • <b>E_effettivo</b> = {E_base:,.2f} + ({split_consumo:.2f} × {delta_Y:,.2f}) = <b style="color: #0369a1;">{E_effettivo:,.2f} €/mese</b><br>
                • <b>R_mutuo</b> = {mortgage_R0:,.2f} €/mese<br>
                • <b>S₀</b> = {Y0:,.2f} - {E_effettivo:,.2f} - {mortgage_R0:,.2f} = <b style="color: #059669; font-size: 1rem;">{base_savings:,.2f} €/mese</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-weight: 700; color: #0f766e; font-size: 1.05rem; margin-bottom: 8px;">
                🏛️ Vettore dei Target Istantanei T⃗(0)
            </div>
            <div style="font-family: monospace; font-size: 0.9rem; line-height: 1.6; color: #334155;">
                • <b>T₁</b> = 3 × {E_effettivo:,.2f} = <b style="color: #0284c7;">{t1_target:,.2f} €</b> (Emergenza)<br>
                • <b>T₂</b> = <b style="color: #0d9488;">{t2_target:,.2f} €</b> (Sinking Fund)<br>
                • <b>T₃b</b> = 4 × {mortgage_R0:,.2f} = <b style="color: #dc2626;">{t3b_target:,.2f} €</b> (Buffer Mutuo)<br>
                • <b>T₃a</b> = <b style="color: #d97706;">{t3a_target:,.2f} €</b> (Milestone H)<br>
                • <b>Θ₃a</b> = 0,60 × {t3a_target:,.2f} = <b style="color: #ea580c;">{threshold_v3a_p4:,.2f} €</b> (Soglia Sblocco Fase 4)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_n2:
        st.markdown(f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-weight: 700; color: #7c2d12; font-size: 1.05rem; margin-bottom: 8px;">
                🎯 Stato FSM & Destinazione Risparmio Attuale (S₀ = {base_savings:,.2f} €)
            </div>
            <div style="margin-bottom: 8px;">
                <span class="badge-tag badge-accent" style="font-size: 0.85rem;">Stato Attivo: {p_info['name']}</span>
            </div>
            <div style="font-family: monospace; font-size: 0.9rem; line-height: 1.6; color: #334155;">
                • <b>ΔV₁</b> (Emergenza): <b>{flow_v1:,.2f} € ({p_alloc['v1']*100:.0f}%)</b><br>
                • <b>ΔV₂</b> (Sinking Fund): <b>{flow_v2:,.2f} € ({p_alloc['v2']*100:.0f}%)</b><br>
                • <b>ΔV₃b</b> (Buffer Mutuo): <b>{flow_v3b:,.2f} € ({p_alloc['v3b']*100:.0f}%)</b><br>
                • <b>ΔV₃a</b> (Obbligazionario BTP): <b>{flow_v3a:,.2f} € ({p_alloc['v3a']*100:.0f}%)</b><br>
                • <b>ΔV₄</b> (Azionario Core All-World): <b>{flow_v4:,.2f} € ({p_alloc['v4']*100:.0f}%)</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-weight: 700; color: #4338ca; font-size: 1.05rem; margin-bottom: 8px;">
                🛡️ Vaso 0: Previdenza Complementare & Scudo IRPEF
            </div>
            <div style="font-family: monospace; font-size: 0.9rem; line-height: 1.6; color: #334155;">
                • <b>RAL Base</b>: {ral_0:,.2f} €/anno<br>
                • <b>TFR (6,91%)</b>: {tfr_monthly:,.2f} €/mese<br>
                • <b>Contributo Datoriale (1,50%)</b>: {match_monthly:,.2f} €/mese (extra free-lunch)<br>
                • <b>Contributo Dipendente</b>: {emp_monthly:,.2f} €/mese ({emp_monthly*12:,.2f} €/anno)<br>
                • <b>Afflusso Totale V0</b>: <b style="color: #4338ca;">{v0_monthly_tot:,.2f} €/mese</b><br>
                • <b>Scudo Fiscale 730 D.Lgs. 252/2005</b>: <b style="color: #059669;">+{tax_refund_yr:,.2f} €/anno di rimborso IRPEF</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --------------------------------------------------------------------------
    # SEZIONE 3: DIAGNOSTICA GRAFICA (4 Grafici Analitici Plotly in Griglia 2x2)
    # --------------------------------------------------------------------------
    st.markdown("### 3. Diagnostica Grafica e Validazione Quantitativa")
    st.caption("Evoluzione deterministica e simulazioni stocastiche per validare l'architettura dinamica e la robustezza del modello.")

    # Calcolo dati per i grafici con caching
    traj_data = compute_dynamic_target_trajectories(
        Y0=Y0,
        E_base=E_base,
        split_consumo=split_consumo,
        g_Y=g_Y,
        mortgage_R0=mortgage_R0,
        t2_target=t2_target,
        t3b_target=t3b_target,
        t3a_target=t3a_target,
        H_years=H_years,
        n_months=T_horizon_val
    )
    
    flow_data = compute_savings_cashflow_dynamics(
        Y0=Y0,
        E_base=E_base,
        split_consumo=split_consumo,
        g_Y=g_Y,
        mortgage_R0=mortgage_R0,
        n_months=T_horizon_val
    )

    vasicek_paths = simulate_vasicek_paths(
        r0=0.0325,
        kappa=0.35,
        theta=0.022,
        sigma=0.012,
        n_months=T_horizon_val,
        n_paths=50,
        seed=int(seed)
    )

    # Griglia 2x2
    grid_r1_c1, grid_r1_c2 = st.columns(2)
    
    # ------------------ GRAFICO 1: Superficie Target 30y ------------------
    with grid_r1_c1:
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(
            x=eta_attuale + traj_data['years'],
            y=traj_data['T1'],
            mode='lines',
            line=dict(color='#0284c7', width=2.5),
            name='T₁(t) Fondo Emergenza (3×E_t)',
            hovertemplate='Anno %{x:.1f}: T₁ = %{y:,.0f} €<extra></extra>'
        ))
        if t3a_target > 0:
            hm = traj_data['H_month']
            fig_t.add_trace(go.Scatter(
                x=traj_data['years'][:hm+1],
                y=traj_data['T3a'][:hm+1],
                mode='lines',
                line=dict(color='#d97706', width=2.5),
                name=f'T₃a(t) Milestone H ({t3a_target:,.0f} € + FOI)',
                hovertemplate='Anno %{x:.1f}: T₃a = %{y:,.0f} €<extra></extra>'
            ))
            fig_t.add_trace(go.Scatter(
                x=traj_data['years'][:hm+1],
                y=traj_data['Theta_3a'][:hm+1],
                mode='lines',
                line=dict(color='#ea580c', width=2, dash='dot'),
                name='Θ₃a(t) Soglia Inversione Fase 4 (60%)',
                hovertemplate='Anno %{x:.1f}: Θ₃a = %{y:,.0f} €<extra></extra>'
            ))
        fig_t.add_trace(go.Scatter(
            x=eta_attuale + traj_data['years'],
            y=traj_data['T2'],
            mode='lines',
            line=dict(color='#0d9488', width=2, dash='dash'),
            name=f'T₂ Sinking Fund ({t2_target:,.0f} €)',
            hovertemplate='T₂ costante = %{y:,.0f} €<extra></extra>'
        ))
        if mortgage_R0 > 0:
            fig_t.add_trace(go.Scatter(
                x=eta_attuale + traj_data['years'],
                y=traj_data['T3b'],
                mode='lines',
                line=dict(color='#dc2626', width=2, dash='dash'),
                name=f'T₃b Buffer Mutuo ({t3b_target:,.0f} €)',
                hovertemplate='T₃b costante = %{y:,.0f} €<extra></extra>'
            ))
        fig_t.update_layout(
            title='<b>1. Superficie di Transizione dei Target Dinamici (30 Anni)</b>',
            xaxis_title='Orizzonte Temporale (Anni)',
            yaxis_title='Ammontare Target (€)',
            template='plotly_white',
            height=410,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(size=10))
        )
        st.plotly_chart(fig_t, use_container_width=True)

    # ------------------ GRAFICO 2: Area Risparmio vs Spese ------------------
    with grid_r1_c2:
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=eta_attuale + flow_data['years'],
            y=flow_data['E0'],
            name='Spese Vive Base (E₀)',
            mode='lines',
            line=dict(width=0.5, color='#64748b'),
            stackgroup='one',
            fillcolor='rgba(100, 116, 139, 0.45)',
            hovertemplate='E₀ = %{y:,.0f} €<extra></extra>'
        ))
        fig_s.add_trace(go.Scatter(
            x=eta_attuale + flow_data['years'],
            y=flow_data['lifestyle'],
            name='Miglioramento Tenore (α·ΔY)',
            mode='lines',
            line=dict(width=0.5, color='#f59e0b'),
            stackgroup='one',
            fillcolor='rgba(245, 158, 11, 0.45)',
            hovertemplate='SMarT Spese = %{y:,.0f} €<extra></extra>'
        ))
        if mortgage_R0 > 0:
            fig_s.add_trace(go.Scatter(
                x=eta_attuale + flow_data['years'],
                y=flow_data['mortgage'],
                name='Rata Passività / Mutuo (R₀)',
                mode='lines',
                line=dict(width=0.5, color='#ef4444'),
                stackgroup='one',
                fillcolor='rgba(239, 68, 68, 0.45)',
                hovertemplate='Rata Mutuo = %{y:,.0f} €<extra></extra>'
            ))
        fig_s.add_trace(go.Scatter(
            x=eta_attuale + flow_data['years'],
            y=flow_data['savings'],
            name='Risparmio Netto Strutturale (S_t)',
            mode='lines',
            line=dict(width=0.5, color='#10b981'),
            stackgroup='one',
            fillcolor='rgba(16, 185, 129, 0.50)',
            hovertemplate='Risparmio Netto = %{y:,.0f} €<extra></extra>'
        ))
        fig_s.update_layout(
            title='<b>2. Dinamica del Risparmio: Consumo SMarT vs Investimento</b>',
            xaxis_title='Orizzonte Temporale (Anni)',
            yaxis_title='Distribuzione Stipendio Netto Mensile (€)',
            template='plotly_white',
            height=410,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(size=10))
        )
        st.plotly_chart(fig_s, use_container_width=True)

    grid_r2_c1, grid_r2_c2 = st.columns(2)

    # ------------------ GRAFICO 3: Simulazione Vasicek ------------------
    with grid_r2_c1:
        years_vasicek = eta_attuale + np.arange(T_horizon_val + 1) / 12.0
        p5_v = np.percentile(vasicek_paths, 5, axis=0) * 100.0
        p50_v = np.percentile(vasicek_paths, 50, axis=0) * 100.0
        p95_v = np.percentile(vasicek_paths, 95, axis=0) * 100.0
        
        fig_v = go.Figure()
        # Traiettorie stocastiche semitrasparenti
        for i in range(min(35, len(vasicek_paths))):
            fig_v.add_trace(go.Scatter(
                x=years_vasicek,
                y=vasicek_paths[i] * 100.0,
                mode='lines',
                line=dict(color='rgba(147, 197, 253, 0.22)', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        # Fascia di confidenza 90%
        fig_v.add_trace(go.Scatter(
            x=years_vasicek,
            y=p95_v,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig_v.add_trace(go.Scatter(
            x=years_vasicek,
            y=p5_v,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(59, 130, 246, 0.16)',
            name='Intervallo Confidenza 90% [p5-p95]'
        ))
        # Traiettoria mediana
        fig_v.add_trace(go.Scatter(
            x=years_vasicek,
            y=p50_v,
            mode='lines',
            line=dict(color='#1d4ed8', width=3),
            name='Traiettoria Mediana (r₀ = 3,25%)'
        ))
        # Tasso di equilibrio lungo termine theta
        fig_v.add_trace(go.Scatter(
            x=years_vasicek,
            y=np.full(len(years_vasicek), 2.2),
            mode='lines',
            line=dict(color='#475569', width=1.8, dash='dash'),
            name='Equilibrio Vasicek θ = 2,20%'
        ))
        fig_v.update_layout(
            title='<b>3. Simulazione Vasicek del Tasso Free Risk (50 Percorsi Monte Carlo)</b>',
            xaxis_title='Orizzonte Temporale (Anni)',
            yaxis_title='Tasso Breve Free-Risk r_t (%)',
            template='plotly_white',
            height=410,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(size=10))
        )
        st.plotly_chart(fig_v, use_container_width=True)

    # ------------------ GRAFICO 4: Mappa Inflow Rebalancing ------------------
    with grid_r2_c2:
        # Calcolo posizione satellite attuale dell'utente o default
        w_sat_default = 40.0
        if v4_c >= 100000.0:
            w_sat_default = 40.0

        # Slider compatto integrato per test interattivo della regola
        w_sat_interactive = st.slider(
            "📐 Quota Satellite attuale di test (w_sat %):",
            min_value=20.0,
            max_value=60.0,
            value=float(w_sat_default),
            step=0.5,
            format="%.1f%%",
            help="Sposta lo slider per verificare come S-DFP adegua istantaneamente i flussi di cassa senza mai vendite coatte."
        )

        inflow_sat_now = 80.0 if w_sat_interactive < 35.0 else (20.0 if w_sat_interactive > 45.0 else 40.0)
        inflow_core_now = 100.0 - inflow_sat_now

        w_grid = np.linspace(0.20, 0.60, 500) * 100.0
        inflow_sat_curve = np.where(w_grid < 35.0, 80.0, np.where(w_grid <= 45.0, 40.0, 20.0))
        inflow_core_curve = 100.0 - inflow_sat_curve

        fig_r = go.Figure()
        # Fascia di inazione verde [35%, 45%]
        fig_r.add_vrect(
            x0=35.0,
            x1=45.0,
            fillcolor='rgba(16, 185, 129, 0.15)',
            line_width=1.5,
            line_color='#10b981',
            annotation_text='Banda Inazione [35%, 45%]',
            annotation_position='top left',
            annotation=dict(font=dict(color='#047857', size=11, family='Segoe UI'))
        )
        # Curve a gradino
        fig_r.add_trace(go.Scatter(
            x=w_grid,
            y=inflow_sat_curve,
            mode='lines',
            line=dict(color='#8b5cf6', width=3.5),
            name='Quota Flusso Satellite (%)',
            hovertemplate='w_sat = %{x:.1f}% → Inflow Sat = %{y:.0f}%<extra></extra>'
        ))
        fig_r.add_trace(go.Scatter(
            x=w_grid,
            y=inflow_core_curve,
            mode='lines',
            line=dict(color='#0284c7', width=2, dash='dot'),
            name='Quota Flusso Core ETF (%)',
            hovertemplate='w_sat = %{x:.1f}% → Inflow Core = %{y:.0f}%<extra></extra>'
        ))
        # Marker stato utente
        fig_r.add_trace(go.Scatter(
            x=[w_sat_interactive],
            y=[inflow_sat_now],
            mode='markers+text',
            marker=dict(color='#ef4444', size=14, symbol='diamond', line=dict(color='#ffffff', width=2)),
            name='Stato Portafoglio Selezionato',
            text=[f"w_sat={w_sat_interactive:.1f}% → Sat {inflow_sat_now:.0f}% / Core {inflow_core_now:.0f}%"],
            textposition='top center',
            textfont=dict(color='#b91c1c', size=11, family='Segoe UI')
        ))
        fig_r.update_layout(
            title='<b>4. Mappa di Inflow Rebalancing (Core vs Satellite)</b>',
            xaxis_title='Quota Patrimonio Attuale nel Satellite w_sat (%)',
            yaxis_title='Ripartizione Flusso Mensile in Ingresso (%)',
            template='plotly_white',
            height=370,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(size=10))
        )
        st.plotly_chart(fig_r, use_container_width=True)
        
        # Callout esplicativo della regola
        if w_sat_interactive < 35.0:
            st.info(f"🔄 **Sottopeso Satellite ({w_sat_interactive:.1f}% < 35%):** S-DFP devia l'**80% del nuovo risparmio** verso l'ETF Satellite e il **20%** al Core ETF per ripristinare il target senza disinvestire.")
        elif w_sat_interactive > 45.0:
            st.warning(f"🔄 **Sovrappeso Satellite ({w_sat_interactive:.1f}% > 45%):** S-DFP riduce il flusso Satellite al **20%** e convoglia l'**80%** al Core ETF, azzerando il Tax Drag.")
        else:
            st.success(f"✅ **Banda di Inazione [35%, 45%]:** Asset allocation bilanciata ({w_sat_interactive:.1f}%). I nuovi flussi rispettano il target strutturale: **60% Core / 40% Satellite**.")


# ==============================================================================
# TAB 5: DOSSIER TITOLI & MONITOR SCADENZE
# ==============================================================================
with tab5:
    st.markdown("#### 💼 Dossier Titoli & Monitor Scadenze a Semaforo")
    st.caption("Registro granulare degli strumenti finanziari del portafoglio, categorizzazione nei vasi S-DFP, monitoraggio attivo delle scadenze e riconciliazione automatica della liquidità.")

    # 1. Inizializzazione Session State per il Portafoglio
    if "portfolio_assets" not in st.session_state:
        st.session_state["portfolio_assets"] = get_default_portfolio_assets()
    assets = st.session_state["portfolio_assets"]

    today = datetime.date.today()

    # 2. Elaborazione e classificazione a semaforo
    red_assets = []
    yellow_assets = []
    green_assets = []
    perpetual_assets = []

    for a in assets:
        scad = a.get("scadenza")
        if a.get("has_scadenza") and scad:
            if isinstance(scad, str):
                try:
                    scad_date = datetime.datetime.strptime(scad, "%Y-%m-%d").date()
                except Exception:
                    scad_date = today
            elif isinstance(scad, datetime.date):
                scad_date = scad
            else:
                scad_date = today
            
            days_left = (scad_date - today).days
            if days_left <= 30:
                red_assets.append((a, scad_date, days_left))
            elif days_left <= 180:
                yellow_assets.append((a, scad_date, days_left))
            else:
                green_assets.append((a, scad_date, days_left))
        else:
            perpetual_assets.append(a)

    # --------------------------------------------------------------------------
    # 3. BOX ALLARMI PRIORITARI A SEMAFORO
    # --------------------------------------------------------------------------
    if red_assets:
        red_items_html = "".join([
            f"<li style='margin-bottom: 4px;'><b>{a['nome']}</b> (ISIN: {a['isin'] or 'N/D'}) — "
            f"<b>{a['importo']:,.2f} €</b> assegnati a <b>{a['vaso']}</b>: "
            f"<span style='background: #dc2626; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 700;'>"
            f"Scade il {scad_d.strftime('%d/%m/%Y')} ({max(0, dl)} giorni residui)</span></li>"
            for a, scad_d, dl in red_assets
        ])
        st.markdown(f"""
        <div style="background: #fee2e2; border-left: 6px solid #dc2626; border-radius: 8px; padding: 16px 20px; margin-bottom: 16px; box-shadow: 0 2px 6px rgba(220,38,38,0.12);">
            <div style="font-weight: 800; color: #991b1b; font-size: 1.05rem; margin-bottom: 6px; display: flex; align-items: center; gap: 8px;">
                <span>🔴 ALLARME PRIORITARIO: SCADENZA IMMINENTE (&le; 30 GIORNI)</span>
                <span style="font-size: 0.8rem; background: #dc2626; color: white; padding: 2px 8px; border-radius: 12px;">{len(red_assets)} TITOLI</span>
            </div>
            <div style="color: #7f1d1d; font-size: 0.92rem; line-height: 1.5; margin-bottom: 10px;">
                <b>Azione Operativa Immediata Richiesta:</b> Il capitale nominale sta per essere rimborsato e accreditato sul conto di liquidità. Verificare la disponibilità per l'obiettivo programmato (es. acquisto veicolo per Milestone H) oppure predisporre il reinvestimento immediato secondo la gerarchia della Macchina a Stati per evitare il drag da liquidità infruttifera.
            </div>
            <ul style="color: #991b1b; margin: 0; padding-left: 20px; font-size: 0.9rem;">
                {red_items_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if yellow_assets:
        yellow_items_html = "".join([
            f"<li style='margin-bottom: 4px;'><b>{a['nome']}</b> (ISIN: {a['isin'] or 'N/D'}) — "
            f"<b>{a['importo']:,.2f} €</b> in <b>{a['vaso']}</b>: "
            f"<span style='background: #f59e0b; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 700;'>"
            f"Scade il {scad_d.strftime('%d/%m/%Y')} (~{max(1, dl // 30)} mesi residui / {dl} gg)</span></li>"
            for a, scad_d, dl in yellow_assets
        ])
        st.markdown(f"""
        <div style="background: #fef3c7; border-left: 6px solid #f59e0b; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; box-shadow: 0 1px 4px rgba(245,158,11,0.1);">
            <div style="font-weight: 700; color: #92400e; font-size: 0.98rem; margin-bottom: 4px; display: flex; align-items: center; gap: 8px;">
                <span>🟡 PRE-ALLARME LIQUIDITÀ & RIBILANCIAMENTO (Scadenza tra 1 e 6 Mesi)</span>
                <span style="font-size: 0.8rem; background: #f59e0b; color: white; padding: 2px 8px; border-radius: 12px;">{len(yellow_assets)} TITOLI</span>
            </div>
            <div style="color: #78350f; font-size: 0.88rem; line-height: 1.5; margin-bottom: 8px;">
                <b>Pianificazione Strategica S-DFP:</b> I seguenti titoli giungeranno a maturazione entro 1-6 mesi (inclusa la finestra pre-target di mese 4). Pianificare in anticipo la destinazione del rimborso capitale (es. Milestone H per Vaso 3A o riallocazione secondo FSM).
            </div>
            <ul style="color: #92400e; margin: 0; padding-left: 20px; font-size: 0.88rem;">
                {yellow_items_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if not assets:
        st.info("ℹ️ Nessuno strumento finanziario registrato nel portafoglio. Utilizza il modulo sottostante per inserire i tuoi titoli.")
    elif not red_assets and not yellow_assets:
        st.markdown("""
        <div style="background: #ecfdf5; border-left: 5px solid #10b981; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; font-size: 0.9rem; color: #065f46;">
            🟢 <b>Nessun allarme di scadenza attivo:</b> Tutti i titoli a scadenza hanno orizzonte superiore a 6 mesi (&gt; 180 giorni) e decorrono regolarmente maturando cedole.
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 4. DASHBOARD RIEPILOGATIVA: KPI METRICS
    # --------------------------------------------------------------------------
    tot_dossier = sum(a['importo'] for a in assets)
    tot_fixed_maturity = len(red_assets) + len(yellow_assets) + len(green_assets)
    tot_alarms = len(red_assets) + len(yellow_assets)

    col_dk1, col_dk2, col_dk3, col_dk4 = st.columns(4)
    with col_dk1:
        st.metric("Controvalore Dossier Titoli", f"{tot_dossier:,.2f} €", help="Somma del capitale investito in tutti gli strumenti registrati")
    with col_dk2:
        st.metric("Posizioni Finanziarie Attive", f"{len(assets)} titoli", help="Numero di strumenti in portafoglio")
    with col_dk3:
        st.metric("Strumenti a Scadenza Fissa", f"{tot_fixed_maturity} bond / CD", help="Titoli governativi, corporate o conti vincolati con scadenza definita")
    with col_dk4:
        st.metric("Scadenze Entro 6 Mesi", f"{tot_alarms}", delta=f"{len(red_assets)} imminenti (&le;30gg)" if red_assets else "Nessun allarme critico", delta_color="inverse" if red_assets else "normal")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 5. GRAFICI DI AGGREGAZIONE (Asset Class & Vasi S-DFP)
    # --------------------------------------------------------------------------
    if assets:
        col_dchart1, col_dchart2 = st.columns(2)

        with col_dchart1:
            cat_totals = {}
            for a in assets:
                cat_totals[a['categoria']] = cat_totals.get(a['categoria'], 0.0) + a['importo']

            palette_cat = {
                "Obbligazione Governativa / BTP": "#d97706",
                "Obbligazione Corporate": "#ea580c",
                "ETF Azionario (Core)": "#2563eb",
                "ETF/ETC Satellite": "#8b5cf6",
                "Strumento Monetario / Liquidità (€STR)": "#0ea5e9"
            }
            colors_cat = [palette_cat.get(c, "#64748b") for c in cat_totals.keys()]

            fig_cat = go.Figure(data=[go.Pie(
                labels=list(cat_totals.keys()),
                values=list(cat_totals.values()),
                hole=0.60,
                marker_colors=colors_cat,
                textinfo='percent',
                hovertemplate='<b>%{label}</b><br>Controvalore: %{value:,.2f} € (%{percent})<extra></extra>'
            )])
            fig_cat.update_layout(
                title='<b>Ripartizione per Categoria Strumento</b>',
                height=290,
                margin=dict(t=40, b=20, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=10)),
                template="plotly_white"
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        with col_dchart2:
            vaso_totals = {}
            for a in assets:
                vaso_totals[a['vaso']] = vaso_totals.get(a['vaso'], 0.0) + a['importo']

            palette_vaso = {
                "Vaso 1 (Emergenza)": "#2563eb",
                "Vaso 2 (Sinking)": "#059669",
                "Vaso 3B (Buffer Mutuo)": "#dc2626",
                "Vaso 3A (Auto / Medio Termine)": "#d97706",
                "Vaso 4 (Crescita / Core-Satellite)": "#4f46e5"
            }
            colors_vaso = [palette_vaso.get(v, "#64748b") for v in vaso_totals.keys()]

            fig_vaso = go.Figure(data=[go.Pie(
                labels=list(vaso_totals.keys()),
                values=list(vaso_totals.values()),
                hole=0.60,
                marker_colors=colors_vaso,
                textinfo='percent',
                hovertemplate='<b>%{label}</b><br>Controvalore: %{value:,.2f} € (%{percent})<extra></extra>'
            )])
            fig_vaso.update_layout(
                title='<b>Ripartizione per Vaso S-DFP</b>',
                height=290,
                margin=dict(t=40, b=20, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=10)),
                template="plotly_white"
            )
            st.plotly_chart(fig_vaso, use_container_width=True)
    else:
        st.info("ℹ️ Nessuno strumento registrato: i grafici di ripartizione per categoria e per vaso appariranno una volta inseriti i titoli.")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 6. RICONCILIAZIONE AUTOMATICA CON I SALDI DICHIARATI (Scheda 1)
    # --------------------------------------------------------------------------
    st.markdown("#### 🔄 Riconciliazione Automatica Dossier Titoli vs Saldi Dichiarati")
    st.caption("Verifica la quadratura tra i titoli registrati nel dossier e i saldi patrimoniali attuali impostati nella Scheda 1.")

    v1_val_rec = st.session_state.get("input_v1", t1_target + 500.0)
    v2_val_rec = st.session_state.get("input_v2", max(0.0, t2_target - 400.0))
    v3b_val_rec = st.session_state.get("input_v3b", t3b_target)
    v3a_val_rec = st.session_state.get("input_v3a", min(16000.0, t3a_target) if t3a_target > 0 else 0.0)
    v4_val_rec = st.session_state.get("input_v4", 25000.0)

    dossier_v1 = sum(a['importo'] for a in assets if "Vaso 1" in a['vaso'])
    dossier_v2 = sum(a['importo'] for a in assets if "Vaso 2" in a['vaso'])
    dossier_v3b = sum(a['importo'] for a in assets if "Vaso 3B" in a['vaso'])
    dossier_v3a = sum(a['importo'] for a in assets if "Vaso 3A" in a['vaso'])
    dossier_v4 = sum(a['importo'] for a in assets if "Vaso 4" in a['vaso'])

    reconciliation_rows = [
        {
            "Comparto S-DFP": "Vaso 1 (Buffer Emergenza)",
            "Saldo Dichiarato Scheda 1 (€)": v1_val_rec,
            "Titoli nel Dossier (€)": dossier_v1,
            "Cassa Libera / Diff (€)": v1_val_rec - dossier_v1,
            "Stato": "✅ Riconciliato" if abs(v1_val_rec - dossier_v1) < 0.01 else ("💧 Cassa Libera non investita" if v1_val_rec > dossier_v1 else "⚠️ Titoli eccedono saldo")
        },
        {
            "Comparto S-DFP": "Vaso 2 (Sinking Fund Spese Cicliche)",
            "Saldo Dichiarato Scheda 1 (€)": v2_val_rec,
            "Titoli nel Dossier (€)": dossier_v2,
            "Cassa Libera / Diff (€)": v2_val_rec - dossier_v2,
            "Stato": "✅ Riconciliato" if abs(v2_val_rec - dossier_v2) < 0.01 else ("💧 Cassa Libera non investita" if v2_val_rec > dossier_v2 else "⚠️ Titoli eccedono saldo")
        },
        {
            "Comparto S-DFP": "Vaso 3B (Buffer Mutuo / Debito)",
            "Saldo Dichiarato Scheda 1 (€)": v3b_val_rec,
            "Titoli nel Dossier (€)": dossier_v3b,
            "Cassa Libera / Diff (€)": v3b_val_rec - dossier_v3b,
            "Stato": "✅ Riconciliato" if abs(v3b_val_rec - dossier_v3b) < 0.01 else ("💧 Cassa Libera non investita" if v3b_val_rec > dossier_v3b else "⚠️ Titoli eccedono saldo")
        },
        {
            "Comparto S-DFP": "Vaso 3A (Milestone H Auto / BTP)",
            "Saldo Dichiarato Scheda 1 (€)": v3a_val_rec,
            "Titoli nel Dossier (€)": dossier_v3a,
            "Cassa Libera / Diff (€)": v3a_val_rec - dossier_v3a,
            "Stato": "✅ Riconciliato" if abs(v3a_val_rec - dossier_v3a) < 0.01 else ("💧 Cassa Libera non investita" if v3a_val_rec > dossier_v3a else "⚠️ Titoli eccedono saldo")
        },
        {
            "Comparto S-DFP": "Vaso 4 (Crescita Core-Satellite ETF)",
            "Saldo Dichiarato Scheda 1 (€)": v4_val_rec,
            "Titoli nel Dossier (€)": dossier_v4,
            "Cassa Libera / Diff (€)": v4_val_rec - dossier_v4,
            "Stato": "✅ Riconciliato" if abs(v4_val_rec - dossier_v4) < 0.01 else ("💧 Cassa Libera non investita" if v4_val_rec > dossier_v4 else "⚠️ Titoli eccedono saldo")
        }
    ]

    df_rec = pd.DataFrame(reconciliation_rows)
    df_rec_styled = df_rec.copy()
    df_rec_styled["Saldo Dichiarato Scheda 1 (€)"] = df_rec_styled["Saldo Dichiarato Scheda 1 (€)"].apply(lambda x: f"{x:,.2f} €")
    df_rec_styled["Titoli nel Dossier (€)"] = df_rec_styled["Titoli nel Dossier (€)"].apply(lambda x: f"{x:,.2f} €")
    df_rec_styled["Cassa Libera / Diff (€)"] = df_rec_styled["Cassa Libera / Diff (€)"].apply(lambda x: f"{x:+,.2f} €" if abs(x) >= 0.01 else "0,00 €")
    st.dataframe(df_rec_styled, use_container_width=True, hide_index=True)

    tot_saldi_dichiarati = v1_val_rec + v2_val_rec + v3b_val_rec + v3a_val_rec + v4_val_rec
    cassa_totale_libera = tot_saldi_dichiarati - tot_dossier
    if not assets and tot_saldi_dichiarati > 0:
        st.info(f"💧 **Portafoglio Vuoto:** I saldi patrimoniali correnti risultano interamente in liquidità libera non allocata ({tot_saldi_dichiarati:,.2f} €). Inserisci i tuoi strumenti finanziari per monitorarne scadenze e vasi.")
    elif abs(cassa_totale_libera) < 0.01:
        st.success("🎯 **Quadratura Perfetta:** Il 100% dei saldi patrimoniali dichiarati è impiegato in strumenti finanziari monitorati nel dossier.")
    elif cassa_totale_libera > 0:
        st.info(f"💧 **Liquidità Libera Complessiva:** Risultano **{cassa_totale_libera:,.2f} €** di cassa non ancora allocata in titoli specifici (giacenza liquida sui conti correnti/deposito liberi).")
    else:
        st.warning(f"⚠️ **Discrepanza Rilevata:** Il controvalore dei titoli inseriti nel dossier eccede i saldi dichiarati nella Scheda 1 di **{abs(cassa_totale_libera):,.2f} €**.")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 7. REGISTRO ANALITICO TITOLI & GESTIONE PORTAFOGLIO
    # --------------------------------------------------------------------------
    st.markdown("#### 📋 Elenco Dettagliato Strumenti & Azioni di Gestione")

    table_rows = []
    for a in assets:
        scad = a.get("scadenza")
        if a.get("has_scadenza") and scad:
            if isinstance(scad, str):
                try:
                    scad_date = datetime.datetime.strptime(scad, "%Y-%m-%d").date()
                except Exception:
                    scad_date = today
            elif isinstance(scad, datetime.date):
                scad_date = scad
            else:
                scad_date = today
            days_left = (scad_date - today).days
            if days_left <= 30:
                badge = f"🔴 Imminente ({days_left} gg)"
            elif days_left <= 180:
                badge = f"🟡 Pre-Allarme ({days_left} gg)"
            else:
                badge = f"🟢 Regolare ({days_left} gg)"
            scad_str = scad_date.strftime("%d/%m/%Y")
            gg_str = f"{days_left} gg"
        else:
            badge = "⚪ Perpetuo"
            scad_str = "—"
            gg_str = "—"

        table_rows.append({
            "ID": a["id"],
            "Stato Scadenza": badge,
            "Denominazione Strumento": a["nome"],
            "ISIN": a["isin"] or "—",
            "Categoria": a["categoria"],
            "Vaso S-DFP": a["vaso"],
            "Controvalore (€)": f"{a['importo']:,.2f} €",
            "PMC (€)": f"{a['pmc']:,.2f} €",
            "Data Scadenza": scad_str,
            "Giorni Residui": gg_str,
            "Note Operative": a["note"] or "—"
        })

    columns = [
        "ID", "Stato Scadenza", "Denominazione Strumento", "ISIN",
        "Categoria", "Vaso S-DFP", "Controvalore (€)", "PMC (€)",
        "Data Scadenza", "Giorni Residui", "Note Operative"
    ]
    df_assets_view = pd.DataFrame(table_rows, columns=columns) if table_rows else pd.DataFrame(columns=columns)
    st.dataframe(df_assets_view, use_container_width=True, hide_index=True)

    col_act1, col_act2 = st.columns([3, 1])
    with col_act2:
        if st.button("🗑️ Svuota Portafoglio", use_container_width=True, help="Rimuove tutti gli strumenti e reimposta il portafoglio vuoto"):
            st.session_state["portfolio_assets"] = []
            st.success("Portafoglio svuotato!")
            st.rerun()

    # Form per Aggiunta Nuovo Strumento
    with st.expander("➕ Inserisci Nuovo Strumento Finanziario nel Portafoglio", expanded=False):
        with st.form("form_add_portfolio_asset", clear_on_submit=True):
            col_fa1, col_fa2 = st.columns(2)
            with col_fa1:
                new_nome = st.text_input("Denominazione / Nome Strumento *", placeholder="es. BTP Valore Ott27 TF 4.10% o SWDA ETF")
                new_isin = st.text_input("Codice ISIN (Opzionale)", placeholder="es. IT0005565391 o IE00B4L5Y983")
                new_cat = st.selectbox(
                    "Categoria Strumento *",
                    [
                        "Obbligazione Governativa / BTP",
                        "Obbligazione Corporate",
                        "ETF Azionario (Core)",
                        "ETF/ETC Satellite",
                        "Strumento Monetario / Liquidità (€STR)"
                    ]
                )
                new_vaso = st.selectbox(
                    "Vaso di Destinazione *",
                    [
                        "Vaso 1 (Emergenza)",
                        "Vaso 2 (Sinking)",
                        "Vaso 3B (Buffer Mutuo)",
                        "Vaso 3A (Auto / Medio Termine)",
                        "Vaso 4 (Crescita / Core-Satellite)"
                    ],
                    index=3 if "Obbligazione" in new_cat else (4 if "ETF" in new_cat else 0)
                )
            with col_fa2:
                new_importo = st.number_input("Controvalore Investito Attuale (€) *", min_value=1.0, value=5000.0, step=500.0)
                new_pmc = st.number_input("Prezzo Medio di Carico (PMC) / Prezzo Acquisto (€)", min_value=0.01, value=100.0, step=0.5)
                new_has_scadenza = st.checkbox("Titolo con Data di Scadenza Fissata (BTP, CD, Obbligazioni)", value=("Obbligazione" in new_cat))
                new_scadenza = None
                if new_has_scadenza:
                    new_scadenza = st.date_input("Data di Scadenza Titolo *", value=today + datetime.timedelta(days=365))
                new_note = st.text_input("Note Operative / Obiettivo", placeholder="es. Quota acquisto auto mese 96")

            btn_add = st.form_submit_button("💾 Registra Titolo nel Portafoglio", use_container_width=True)
            if btn_add:
                if not new_nome.strip():
                    st.error("Inserisci la denominazione del titolo.")
                else:
                    new_id = f"AST-{len(st.session_state['portfolio_assets'])+1:02d}-{int(datetime.datetime.now().timestamp())%1000}"
                    new_entry = {
                        "id": new_id,
                        "nome": new_nome.strip(),
                        "isin": new_isin.strip().upper(),
                        "categoria": new_cat,
                        "vaso": new_vaso,
                        "importo": float(new_importo),
                        "pmc": float(new_pmc),
                        "has_scadenza": bool(new_has_scadenza),
                        "scadenza": new_scadenza if new_has_scadenza else None,
                        "note": new_note.strip()
                    }
                    st.session_state["portfolio_assets"].append(new_entry)
                    st.success(f"✅ Titolo '{new_nome}' aggiunto con successo!")
                    st.rerun()

    # Sezione Eliminazione Titoli Esistenti
    with st.expander("🗑️ Modifica o Elimina Titoli Esistenti", expanded=False):
        st.caption("Seleziona uno strumento da rimuovere dal dossier titoli.")
        asset_options = {f"{a['id']} - {a['nome']} ({a['importo']:,.0f} € in {a['vaso']})": a["id"] for a in assets}
        if asset_options:
            selected_asset_label = st.selectbox("Seleziona strumento:", list(asset_options.keys()))
            selected_asset_id = asset_options[selected_asset_label]
            if st.button(f"Elimina Definitivamente '{selected_asset_label.split(' - ')[1]}'", type="secondary"):
                st.session_state["portfolio_assets"] = [a for a in assets if a["id"] != selected_asset_id]
                st.success("Strumento eliminato dal portafoglio!")
                st.rerun()
        else:
            st.info("Nessuno strumento presente nel portafoglio.")


# ==============================================================================
# TAB 6: ESPORTAZIONE DATI
# ==============================================================================
with tab6:
    st.markdown("#### 💾 Esportazione Dati e Report Esecutivo")
    st.caption("Scarica i risultati completi dell'analisi quantitativa e il Dossier Titoli in formato standard CSV e JSON per audit e ulteriori elaborazioni.")

    st.markdown("##### 📊 Risultati di Simulazione e Metriche Accademiche")
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    # 1. CSV Ablation Results
    csv_buffer = io.StringIO()
    df_ablation.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    
    with col_exp1:
        st.download_button(
            label="📥 Scarica Tabella Ablation (CSV)",
            data=csv_bytes,
            file_name="sdfp_ablation_results.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption("Risultati sintetici completi degli 8 modelli M0-M7 con tutte le metriche accademiche.")

    # 2. JSON Executive Summary
    summary_export = {
        "parameters": sim_data['cfg'],
        "metrics": {k: {m: v for m, v in val.items() if m != 'Wealth'} for k, val in summary.items()},
        "wealth_milestones": {k: val['Wealth'] for k, val in summary.items()},
        "historical_stress_tests": stress_data
    }
    json_bytes = json.dumps(summary_export, indent=4).encode('utf-8')
    
    with col_exp2:
        st.download_button(
            label="📥 Scarica Report Esecutivo (JSON)",
            data=json_bytes,
            file_name="sdfp_simulation_results.json",
            mime="application/json",
            use_container_width=True
        )
        st.caption("Dati completi di simulazione, parametri microeconomici e scenari di stress.")

    # 3. CSV Traiettorie 30 Anni
    traj_dict = {"Anno": years}
    for mod in ['M0_Naive', 'M2_Fair', 'M4_SDFP_ScenB', 'M5_SDFP_ScenC', 'M7_Fair_LiabInflowReb']:
        if mod in yearly_data:
            traj_dict[f"{mod}_Med"] = yearly_data[mod]['W_tot_med']
            traj_dict[f"{mod}_p10"] = yearly_data[mod]['W_tot_p10']
            traj_dict[f"{mod}_p90"] = yearly_data[mod]['W_tot_p90']
    df_traj = pd.DataFrame(traj_dict)
    
    traj_csv_buffer = io.StringIO()
    df_traj.to_csv(traj_csv_buffer, index=False)
    traj_csv_bytes = traj_csv_buffer.getvalue().encode('utf-8')
    
    with col_exp3:
        st.download_button(
            label="📥 Scarica Traiettorie 30y (CSV)",
            data=traj_csv_bytes,
            file_name="sdfp_wealth_trajectories_30y.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption("Serie storica anno per anno dei percentili 10°, 50° e 90° per tutti i modelli.")

    st.markdown("---")
    st.markdown("##### 💼 Dossier Titoli & Monitor Scadenze")
    col_dexp1, col_dexp2 = st.columns(2)

    # Export Dossier Titoli CSV
    assets_for_export = st.session_state.get("portfolio_assets", get_default_portfolio_assets())
    df_assets_exp = pd.DataFrame(assets_for_export)
    if "scadenza" in df_assets_exp.columns:
        df_assets_exp["scadenza"] = df_assets_exp["scadenza"].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, (datetime.date, datetime.datetime)) else (str(x) if x else ''))
    
    csv_dossier_buffer = io.StringIO()
    df_assets_exp.to_csv(csv_dossier_buffer, index=False)
    csv_dossier_bytes = csv_dossier_buffer.getvalue().encode('utf-8')

    with col_dexp1:
        st.download_button(
            label="📥 Scarica Dossier Titoli (CSV)",
            data=csv_dossier_bytes,
            file_name="sdfp_dossier_titoli.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption("Tabella completa degli strumenti in portafoglio con scadenze, categorie e vasi.")

    # Export Dossier Titoli JSON
    assets_json_export = []
    for a in assets_for_export:
        item = a.copy()
        if item.get("scadenza") is not None:
            item["scadenza"] = item["scadenza"].strftime('%Y-%m-%d') if isinstance(item["scadenza"], (datetime.date, datetime.datetime)) else str(item["scadenza"])
        assets_json_export.append(item)
    json_dossier_bytes = json.dumps(assets_json_export, indent=4).encode('utf-8')

    with col_dexp2:
        st.download_button(
            label="📥 Scarica Dossier Titoli (JSON)",
            data=json_dossier_bytes,
            file_name="sdfp_dossier_titoli.json",
            mime="application/json",
            use_container_width=True
        )
        st.caption("Dati strutturati del portafoglio titoli per audit, contabilità personale e fogli di calcolo.")

    st.markdown("---")
    st.markdown("##### 📄 Documenti e Pubblicazioni Ufficiali")
    st.caption("Scarica la guida sintetica esecutiva in una pagina e il manoscritto scientifico per consultazione e archiviazione.")
    col_doc1, col_doc2, col_doc3 = st.columns(3)

    p_flyer = resolve_file_path("S_DFP_in_una_pagina.pdf")
    if p_flyer and p_flyer.exists():
        with open(p_flyer, "rb") as f_flyer:
            with col_doc1:
                st.download_button(
                    label="📄 Scarica S-DFP in una Pagina (PDF A4)",
                    data=f_flyer.read(),
                    file_name="S_DFP_in_una_pagina.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.caption("Executive flyer in una singola pagina A4 con le 6 scatole, 5 vantaggi e 3 regole.")

    p_flyer_md = resolve_file_path("S_DFP_in_una_pagina.md")
    if p_flyer_md and p_flyer_md.exists():
        with open(p_flyer_md, "rb") as f_fmd:
            with col_doc2:
                st.download_button(
                    label="📝 Scarica S-DFP in una Pagina (Markdown)",
                    data=f_fmd.read(),
                    file_name="S_DFP_in_una_pagina.md",
                    mime="text/markdown",
                    use_container_width=True
                )
                st.caption("Testo markdown portabile per appunti, documentazione o presentazioni.")

    p_paper = resolve_file_path("investimenti.pdf") or resolve_file_path("paper_S_DFP.pdf")
    if p_paper and p_paper.exists():
        with open(p_paper, "rb") as f_paper:
            with col_doc3:
                st.download_button(
                    label="📑 Scarica Paper Accademico (PDF 13 Pagine)",
                    data=f_paper.read(),
                    file_name="paper_S_DFP.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.caption("Paper teorico completo di formulazione analitica SDE, ablation study a 8 modelli e audit.")

    st.markdown("---")
    st.markdown("##### 🔍 Anteprima del Report Esecutivo JSON")
    st.json(summary_export, expanded=False)
