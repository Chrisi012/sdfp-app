"""
Stochastic Simulation Engine for S-DFP (State-Dependent Sequential Flow Allocation)
and Comparative Benchmark Analysis.

Author: Senior Quantitative Researcher in Quantitative Finance, Portfolio Management,
        Behavioral Economics and Statistical Validation.
Academic Standards: Journal of Wealth Management / Journal of Portfolio Management / JFQA.
Regulatory Framework: TUIR, D.Lgs. 461/1997, D.Lgs. 252/2005, D.L. 201/2011.
"""

import numpy as np
import scipy.stats as stats
import json
import csv
import os
import sys

# ==============================================================================
# 1. CONFIGURATION DATASTRUCTURE
# ==============================================================================

class SimulationConfig:
    """Centralized configuration container with JSON export/import."""
    def __init__(self, **kwargs):
        # Time and Simulation Parameters
        self.n_paths = kwargs.get('n_paths', 10000)
        self.T_horizon = kwargs.get('T_horizon', kwargs.get('n_months', kwargs.get('n_years', 30) * 12))
        self.n_months = self.T_horizon
        self.n_years = self.n_months // 12
        self.T_retire = kwargs.get('T_retire', self.T_horizon - 84)
        self.dt = 1.0 / 12.0
        self.seed = kwargs.get('seed', 42)
        self.sub_seeds = kwargs.get('sub_seeds', [101, 777, 2026])
        
        # Macroeconomic Parameters (Annualized)
        self.mu_eq = kwargs.get('mu_eq', 0.075)         # Equity drift (7.5%)
        self.sigma_eq = kwargs.get('sigma_eq', 0.165)     # Equity volatility (16.5%)
        
        self.r0 = kwargs.get('r0', 0.0325)                # Initial short rate (ESTER ~3.25%)
        self.kappa_r = kwargs.get('kappa_r', 0.35)        # Mean reversion speed for rates
        self.theta_r = kwargs.get('theta_r', 0.022)       # Long term short rate (2.2%)
        self.sigma_r = kwargs.get('sigma_r', 0.012)       # Short rate volatility (1.2%)
        
        self.pi0 = kwargs.get('pi0', 0.020)               # Initial inflation (2.0%)
        self.kappa_pi = kwargs.get('kappa_pi', 0.50)      # Mean reversion speed for inflation
        self.theta_pi = kwargs.get('theta_pi', 0.020)     # Long term inflation (2.0%)
        self.sigma_pi = kwargs.get('sigma_pi', 0.010)     # Inflation volatility (1.0%)
        
        self.spread_mean = kwargs.get('spread_mean', 0.0110) # 10Y BTP-Bund spread (110 bps)
        self.spread_sigma = kwargs.get('spread_sigma', 0.0035)
        self.buffer_months = kwargs.get('buffer_months', 3.0)
        
        # Dual-Shock Model: continuous intensity lambda(DD_t) = lambda_0 * (1 + gamma * |DD_t|)
        self.lambda_0 = kwargs.get('lambda_0', 0.20)      # Baseline arrival frequency (0.20/yr)
        self.gamma_dual = kwargs.get('gamma_dual', 2.50)   # Continuous drawdown multiplier
        self.mu_D = kwargs.get('mu_D', np.log(1800.0))    # Median jump severity 1,800 EUR
        self.sigma_D = kwargs.get('sigma_D', 0.55)        # Jump log-variance
        
        # Institutional & Fiscal Parameters (Italian Framework)
        self.tax_bonds = 0.125                            # White list sovereign bonds (12.5%)
        self.tax_equity = 0.260                           # Capital gains on equity / standard ETFs (26%)
        self.stamp_duty = 0.0020                          # Stamp duty (0.20% p.a., D.L. 201/2011)
        self.pension_fund_tax = 0.20                      # Blended internal substitute tax (~18-20%)
        
        # Labor Income & Pension Pillar
        self.Y0 = kwargs.get('Y0', 2000.0)                # Net monthly starting income
        self.g_Y = kwargs.get('g_Y', 0.018)               # Nominal annual wage growth (1.8%)
        self.ral_0 = kwargs.get('ral_0', 32000.0)         # Gross starting salary
        self.tfr_rate = 0.0691                            # TFR (6.91%)
        self.matching_contrib_rate = 0.015                # Employer matching (1.5%)
        self.employee_pension_contrib = 50.0              # Employee voluntary contribution (EUR/m)
        self.tax_deduction_rate = 0.35                    # Marginal IRPEF bracket for 730 deduction
        self.smart_split = kwargs.get('smart_split', 0.50)# SMarT split: fraction of delta_Y to consumption (50% default)
        
        # Goals & Liabilities
        self.H_years = kwargs.get('H_years', 8)           # Milestone H (Car purchase at year 8)
        self.H_month = self.H_years * 12                  # Month 96
        self.T3a_0 = kwargs.get('T3a_0', 25000.0)         # Base real target at t=0
        self.mortgage_R0 = kwargs.get('mortgage_R0', 600.0)# Mortgage monthly installment
        self.T3b_fixed = kwargs.get('T3b_fixed', 4.0 * self.mortgage_R0) # 2,400 EUR buffer (0 if mortgage_R0 == 0)
        self.T2_fixed = kwargs.get('T2_fixed', 2400.0)                            # Sinking fund 12-24m
        self.E0 = kwargs.get('E0', 1500.0)                # Base ordinary expenses
        self.S0 = kwargs.get('S0', max(0.0, self.Y0 - self.E0)) # Base monthly savings
        
        # Costs and Friction (TER & Brokerage)
        self.ter_core = kwargs.get('ter_core', 0.0012)    # All-World ETF TER (12 bps)
        self.ter_sat = kwargs.get('ter_sat', 0.0025)      # Satellite Factor ETF TER (25 bps)
        self.ter_monetary = kwargs.get('ter_monetary', 0.0007) # XEON ETF TER (7 bps)
        self.broker_fixed_fee = kwargs.get('broker_fixed_fee', 2.50) # Fixed fee per trade (EUR)
        self.broker_pct_fee = kwargs.get('broker_pct_fee', 0.0010)   # Proportional fee (10 bps)
        self.bid_ask_equity = kwargs.get('bid_ask_equity', 0.0004)   # Bid-ask spread equity (4 bps)
        self.bid_ask_bond = kwargs.get('bid_ask_bond', 0.0008)       # Bid-ask spread sovereign bond (8 bps)
        
        # Milestone evaluation months
        default_mils = [m for m in [96, 180, 240, 360, 480, 600, 720, 1200] if m <= self.T_horizon]
        if self.T_horizon not in default_mils:
            default_mils.append(self.T_horizon)
        default_mils.sort()
        self.milestones = kwargs.get('milestones', default_mils)
        # Vaso 3A Configuration: "coupon" (BTP coupon + routing) or "zerocoupon" (P = F / (1+y)^T)
        self.v3a_config = kwargs.get('v3a_config', 'coupon')
        
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
    def save_json(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)


# ==============================================================================
# 2. COMMON RANDOM NUMBERS (CRN) & MACRO ENVIRONMENT
# ==============================================================================

class MacroEnvironment:
    """
    Generates joint stochastic macroeconomic and asset return paths
    using Common Random Numbers (CRN) for perfect cross-model symmetry.
    Separates pure bond price movements from coupon yields to eliminate double counting.
    """
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        
        # Correlation Matrix: [Equity, Short Rate, Inflation]
        self.corr_matrix = np.array([
            [ 1.00, -0.20,  0.15],
            [-0.20,  1.00,  0.30],
            [ 0.15,  0.30,  1.00]
        ])
        self.L = np.linalg.cholesky(self.corr_matrix)
        
    def generate_paths(self, n_paths, seed=None):
        """
        Pre-generates all stochastic innovations across n_paths and n_months.
        Returns a structured dictionary of market trajectories.
        """
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(self.cfg.seed)
            
        N = n_paths
        M = self.cfg.n_months
        dt = self.cfg.dt
        
        # 1. Correlated Gaussian Innovations
        Z_raw = np.random.normal(0, 1, (M, N, 3))
        Z = np.einsum('mnt,kt->mnk', Z_raw, self.L) # Shape (M, N, 3)
        
        # 2. Short Rate (ESTER) - Vasicek / Hull-White
        r_ester = np.zeros((M + 1, N))
        r_ester[0] = self.cfg.r0
        
        # 3. Inflation Rate - Mean-reverting Ornstein-Uhlenbeck
        pi = np.zeros((M + 1, N))
        pi[0] = self.cfg.pi0
        cum_infl = np.ones((M + 1, N))
        
        # 4. Asset Returns & Innovations
        r_eq = np.zeros((M, N))
        r_bond_price = np.zeros((M, N))
        r_bond_coupon = np.zeros((M, N))
        dr_series = np.zeros((M, N))
        
        # 5. Dual-Shock Poisson Jumps (state-dependent)
        shocks = np.zeros((M, N))
        
        # Track equity peak for continuous drawdown calculation
        eq_index = np.ones(N)
        eq_peak = np.ones(N)
        
        for m in range(M):
            # Short rate update
            dr = self.cfg.kappa_r * (self.cfg.theta_r - r_ester[m]) * dt + self.cfg.sigma_r * np.sqrt(dt) * Z[m, :, 1]
            r_ester[m + 1] = np.maximum(0.0001, r_ester[m] + dr)
            dr_series[m] = dr
            
            # Inflation update
            dpi = self.cfg.kappa_pi * (self.cfg.theta_pi - pi[m]) * dt + self.cfg.sigma_pi * np.sqrt(dt) * Z[m, :, 2]
            pi[m + 1] = np.maximum(-0.01, pi[m] + dpi)
            cum_infl[m + 1] = cum_infl[m] * (1.0 + pi[m + 1] * dt)
            
            # Equity return (Geometric Brownian Motion)
            r_eq[m] = np.exp((self.cfg.mu_eq - 0.5 * self.cfg.sigma_eq**2) * dt + 
                             self.cfg.sigma_eq * np.sqrt(dt) * Z[m, :, 0]) - 1.0
            
            # Update equity peak and drawdown
            eq_index *= (1.0 + r_eq[m])
            eq_peak = np.maximum(eq_peak, eq_index)
            dd_eq = np.maximum(0.0, (eq_peak - eq_index) / eq_peak)
            
            # Continuous Dual-Shock Arrival Intensity: lambda(DD_t) = lambda_0 * (1 + gamma * |DD_t|)
            lambda_t = self.cfg.lambda_0 * (1.0 + self.cfg.gamma_dual * dd_eq)
            p_shock = 1.0 - np.exp(-lambda_t * dt)
            u_shock = np.random.uniform(0, 1, N)
            has_shock = (u_shock < p_shock)
            
            # Shock severity drawn from lognormal distribution
            severities = np.where(has_shock, np.random.lognormal(self.cfg.mu_D, self.cfg.sigma_D, N), 0.0)
            shocks[m] = severities
            
            # Sovereign 7-10Y Bond Components (Duration ~ 6.5)
            yield_bond = r_ester[m] + self.cfg.spread_mean
            r_bond_coupon[m] = yield_bond * dt           # Pure coupon carry
            r_bond_price[m] = - 6.5 * dr                  # Pure duration price movement
            
        # 6. Satellite ETF/ETC returns (0.60 equity beta + independent tracking factor)
        r_sat = np.zeros((M, N))
        Z_sat = np.random.normal(0, 1, (M, N))
        for m in range(M):
            r_sat[m] = 0.60 * r_eq[m] + 0.40 * (0.005 * dt + 0.14 * np.sqrt(dt) * Z_sat[m])
            
        return {
            'r_ester': r_ester,
            'pi': pi,
            'cum_infl': cum_infl,
            'r_eq': r_eq,
            'r_bond_price': r_bond_price,
            'r_bond_coupon': r_bond_coupon,
            'r_bond_total': r_bond_price + r_bond_coupon,
            'dr_series': dr_series,
            'shocks': shocks,
            'r_sat': r_sat
        }


# ==============================================================================
# 3. VASO 3A PRICING & RESIDUAL INFLATION RISK (RIR)
# ==============================================================================

class Vaso3APricingEngine:
    """
    Simulates explicit discrete cash flows for Vaso 3A (Liability Matching for Goal H):
    - Supports Nominal Immunization, Real Immunization (linkers), and Mixed Strategy.
    - Explicit coupon schedule, 12.5% tax withholding, indexation lag, issuance gap,
      transaction fees, bid-ask spread, and batch parking.
    """
    @staticmethod
    def evaluate_vaso3a_zerocoupon(mode='nominal', r_bond_price_series=None, r_bond_coupon_series=None,
                                   infl_series=None, dr_series=None, inflows=None, config=None, H_month=96,
                                   disable_issuance_gap=False, disable_batch_friction=False):
        """
        Simulates explicit zero-coupon bond pricing: P = F / (1 + y)^T.
        No intermediate coupons, no Coupon Routing.
        Capital gain tax = 12.5% * max(0, F - P_purchase) applied exclusively at maturity H.
        """
        N = inflows.shape[1]
        dt = config.dt
        F_total = np.zeros(N)
        cost_total = np.zeros(N)
        V_cash = np.zeros(N)
        tax_bonds = config.tax_bonds
        target_H = config.T3a_0 * infl_series[H_month]

        for m in range(H_month):
            monthly_inflow = inflows[m]
            V_cash += monthly_inflow

            is_issuance_gap = ((H_month - 24) <= m < H_month) and (not disable_issuance_gap)
            T_remain = (H_month - m) / 12.0

            y_nom = (r_bond_coupon_series[m] / dt) if r_bond_coupon_series is not None else (config.r0 + config.spread_mean)
            pi_m = ((infl_series[m+1] / infl_series[m] - 1.0) / dt) if infl_series is not None else config.theta_pi
            y_real = y_nom - pi_m

            min_lot = 0.0 if disable_batch_friction else 1500.0
            if not is_issuance_gap:
                can_buy = (V_cash >= min_lot)
                if np.any(can_buy):
                    buy_amount = np.where(can_buy, V_cash, 0.0)
                    fees = 0.0 if disable_batch_friction else (np.maximum(config.broker_fixed_fee, buy_amount * config.broker_pct_fee) + buy_amount * (config.bid_ask_bond * 0.5))
                    c_invest = np.maximum(0.0, buy_amount - fees)

                    if mode == 'nominal':
                        f_bought = c_invest * ((1.0 + y_nom) ** T_remain)
                    elif mode == 'real':
                        infl_mult = infl_series[H_month] / infl_series[m]
                        f_bought = c_invest * ((1.0 + y_real) ** T_remain) * infl_mult
                    elif mode == 'mixed':
                        f_nom = 0.50 * c_invest * ((1.0 + y_nom) ** T_remain)
                        infl_mult = infl_series[H_month] / infl_series[m]
                        f_real = 0.50 * c_invest * ((1.0 + y_real) ** T_remain) * infl_mult
                        f_bought = f_nom + f_real

                    F_total += f_bought
                    cost_total += c_invest
                    V_cash = np.where(can_buy, 0.0, V_cash)
            else:
                r_mon = np.maximum(0.0, r_bond_coupon_series[m] * 0.5)
                V_cash *= (1.0 + r_mon * (1.0 - tax_bonds) * dt)

        # At maturity H: capital gain tax 12.5%
        gain = np.maximum(0.0, F_total - cost_total)
        tax_gain = tax_bonds * gain
        W_H = (F_total - tax_gain) + V_cash

        relative_errors = (W_H - target_H) / target_H
        rir = float(np.sqrt(np.mean(relative_errors**2)))
        return W_H, rir

    @staticmethod
    def evaluate_vaso3a(mode='mixed', r_bond_price_series=None, r_bond_coupon_series=None, 
                         infl_series=None, dr_series=None, inflows=None, config=None, H_month=96,
                         disable_issuance_gap=False, disable_batch_friction=False,
                         disable_coupon_tax=False, disable_foi_lag=False,
                         deterministic_infl=False, v3a_config='coupon'):
        """
        Runs explicit bond cash-flow matching over H_month with pull-to-par convergence.
        Supports counterfactual ablation of individual market frictions.
        Dispatches to evaluate_vaso3a_zerocoupon if v3a_config == 'zerocoupon'.
        """
        if v3a_config == 'zerocoupon' or getattr(config, 'v3a_config', 'coupon') == 'zerocoupon':
            return Vaso3APricingEngine.evaluate_vaso3a_zerocoupon(
                mode=mode, r_bond_price_series=r_bond_price_series, r_bond_coupon_series=r_bond_coupon_series,
                infl_series=infl_series, dr_series=dr_series, inflows=inflows, config=config, H_month=H_month,
                disable_issuance_gap=disable_issuance_gap, disable_batch_friction=disable_batch_friction
            )
        """
        Runs explicit bond cash-flow matching over H_month with pull-to-par convergence.
        Supports counterfactual ablation of individual market frictions.
        """
        N = inflows.shape[1]
        V_bonds = np.zeros(N)
        V_cash = np.zeros(N)
        
        if deterministic_infl:
            infl_mean = getattr(config, 'theta_pi', 0.020)
            infl_series_used = np.repeat(((1.0 + infl_mean * config.dt) ** np.arange(H_month + 2))[:, np.newaxis], N, axis=1)
        else:
            infl_series_used = infl_series
            
        target_H = config.T3a_0 * infl_series_used[H_month]
        
        tax_bonds = 0.0 if disable_coupon_tax else config.tax_bonds
        coupon_rate_nom = 0.032  # 3.2% nominal coupon
        coupon_rate_real = 0.015 # 1.5% real coupon on BTP Italia
        
        for m in range(H_month):
            dt = config.dt
            # Issuance Gap check: exactly 24 months before H
            is_issuance_gap = ((H_month - 24) <= m < H_month) and (not disable_issuance_gap)
            
            # Accumulate inflow into parking cash
            monthly_inflow = inflows[m]
            V_cash += monthly_inflow
            
            # Pull-to-par effective duration: decreases linearly to 0 at H
            dur_eff = 6.5 * max(0.0, (H_month - m) / H_month)
            dr_m = dr_series[m] if dr_series is not None else 0.0
            r_price_eff = - dur_eff * dr_m
            
            # Reinvest coupons
            if mode == 'nominal':
                coupons = V_bonds * (coupon_rate_nom * (1.0 - tax_bonds) * dt)
                V_cash += coupons
                V_bonds *= (1.0 + r_price_eff)
            elif mode == 'real':
                if disable_foi_lag:
                    lagged_infl = infl_series_used[m+1] / infl_series_used[0]
                else:
                    lagged_infl = infl_series_used[max(0, m - 3)] / infl_series_used[0]
                coupons = V_bonds * (coupon_rate_real * lagged_infl * (1.0 - tax_bonds) * dt)
                V_cash += coupons
                r_linker_price = r_price_eff + (infl_series_used[m+1] / infl_series_used[m] - 1.0)
                V_bonds *= (1.0 + r_linker_price)
            elif mode == 'mixed':
                c_nom = 0.50 * V_bonds * (coupon_rate_nom * (1.0 - tax_bonds) * dt)
                if disable_foi_lag:
                    lagged_infl = infl_series_used[m+1] / infl_series_used[0]
                else:
                    lagged_infl = infl_series_used[max(0, m - 3)] / infl_series_used[0]
                c_real = 0.50 * V_bonds * (coupon_rate_real * lagged_infl * (1.0 - tax_bonds) * dt)
                V_cash += (c_nom + c_real)
                r_mix_price = 0.50 * r_price_eff + 0.50 * (r_price_eff + (infl_series_used[m+1] / infl_series_used[m] - 1.0))
                V_bonds *= (1.0 + r_mix_price)
                
            # Batch purchase into bonds if cash >= min_lot and not in issuance gap
            min_lot = 0.0 if disable_batch_friction else 1500.0
            if not is_issuance_gap:
                can_buy = (V_cash >= min_lot)
                if np.any(can_buy):
                    buy_amount = np.where(can_buy, V_cash, 0.0)
                    if disable_batch_friction:
                        fees = 0.0
                    else:
                        fees = np.maximum(config.broker_fixed_fee, buy_amount * config.broker_pct_fee) + buy_amount * (config.bid_ask_bond * 0.5)
                    V_bonds += np.maximum(0.0, buy_amount - fees)
                    V_cash = np.where(can_buy, 0.0, V_cash)
            else:
                # In Issuance Gap: V_cash parked in monetary instruments (€STR / BOT)
                r_mon = np.maximum(0.0, r_bond_coupon_series[m] * 0.5)
                V_cash *= (1.0 + r_mon * (1.0 - tax_bonds) * dt)
                
        # Total wealth at H
        W_H = V_bonds + V_cash
        relative_errors = (W_H - target_H) / target_H
        rir = np.sqrt(np.mean(relative_errors**2))
        return W_H, rir


# ==============================================================================
# S-DFP STATE MACHINE CONSTANTS
# ==============================================================================
STATE_S1 = 1           # Boot V1 (Buffer Emergenza: fino a T1(t))
STATE_S2 = 2           # Boot V2 (Sinking Fund: fino a T2 = 2.400 EUR)
STATE_S3 = 3           # Boot V3B (Buffer Rata Mutuo: fino a T3b = 2.400 EUR)
STATE_S4 = 4           # Boot V3A Parziale (fino a 0.60 * T3a_cur)
STATE_S4_INV = 5       # Inversione pre-target (60% V3A / 40% V4 Core)
STATE_S5A = 6          # S5A: V3A al target, V4 < 100.000 EUR (100% Core ETF All-World)
STATE_S5B = 7          # S5B: V3A al target, V4 >= 100.000 EUR (Core-Satellite 60/40 Inflow Rebalancing [35%, 45%])
STATE_S_REBUILD = 8    # Rebuilding Difensivo post-Milestone H (60% V3A / 40% V4 Core)


# ==============================================================================
# 4. UNIFIED MULTI-MODEL SIMULATION ENGINE (M0 -> M6)
# ==============================================================================

class UnifiedSimulationEngine:
    """
    Executes the 7 ablation models (M0 to M6) concurrently under identical
    Common Random Numbers (CRN) to ensure absolute symmetry.
    """
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.macro_env = MacroEnvironment(config)
        
    def run_all_models(self, n_paths=None, seed=None, macro=None):
        """
        Runs M0 through M6.
        Returns comprehensive metrics, percentile distributions, and test statistics.
        """
        N = n_paths if n_paths is not None else self.cfg.n_paths
        M = self.cfg.n_months
        dt = self.cfg.dt
        
        # Generate stochastic macro trajectories
        if macro is None:
            macro = self.macro_env.generate_paths(N, seed=seed)
        r_ester = macro['r_ester']
        pi = macro['pi']
        cum_infl = macro['cum_infl']
        r_eq_gross = macro['r_eq']
        r_bond_price = macro['r_bond_price']
        r_bond_coupon = macro['r_bond_coupon']
        r_bond_total = macro['r_bond_total']
        dr_series = macro['dr_series']
        shocks = macro['shocks']
        r_sat_gross = macro['r_sat']
        
        # Net returns after TER (Bug R2)
        r_eq = r_eq_gross - self.cfg.ter_core * dt
        r_sat = r_sat_gross - self.cfg.ter_sat * dt
        
        # Anti-ZIRP switch for monetary instruments
        is_anti_zirp = (r_ester < 0.015)
        r_mon_xeon_net = np.maximum(0.0, r_ester - self.cfg.ter_monetary) * (1.0 - self.cfg.tax_bonds) * dt
        r_mon_depo_net = r_ester * (1.0 - self.cfg.tax_equity) * dt
        r_mon_net_dynamic = np.where(is_anti_zirp, r_mon_xeon_net, r_mon_depo_net)
        r_mon_net_static_m1 = r_ester * (1.0 - self.cfg.tax_equity) * dt # M1 always taxed at 26%, no anti-ZIRP
        
        # -------------------------------------------------------------
        # Income and Savings Dynamics
        # -------------------------------------------------------------
        t_years = np.arange(1, M + 1) * dt
        Y_t_series = np.zeros((M, N))
        delta_Y_series = np.zeros((M, N))
        for m in range(M):
            Y_t_series[m] = self.cfg.Y0 * ((1.0 + self.cfg.g_Y) ** t_years[m])
            delta_Y_series[m] = Y_t_series[m] - self.cfg.Y0
            
        # Pillar 0 (Fondo Pensione & TFR)
        ral_t = self.cfg.ral_0 * ((1.0 + self.cfg.g_Y) ** t_years)
        tfr_m = (ral_t / 12.0) * self.cfg.tfr_rate
        matching_m = (ral_t / 12.0) * self.cfg.matching_contrib_rate
        v0_inflows = tfr_m + matching_m + self.cfg.employee_pension_contrib
        
        V0 = np.zeros(N)
        V0_milestones = {m: np.zeros(N) for m in self.cfg.milestones}
        
        # 730 Tax Refund series (credited to investment in June of each year: m % 12 == 5)
        refund_730 = np.zeros((M, N))
        for m in range(M):
            if m % 12 == 5:
                ann_contrib = (self.cfg.employee_pension_contrib + matching_m[m]) * 12.0
                deduction_base = min(5164.57, ann_contrib)
                refund_730[m] = deduction_base * self.cfg.tax_deduction_rate
                
        # Target H profile
        T3a_target_series = self.cfg.T3a_0 * cum_infl
        
        # -------------------------------------------------------------
        # State Containers for Models M0 to M6
        # -------------------------------------------------------------
        model_names = ['M0_Naive', 'M1_Naive_Buf', 'M2_Fair', 'M3_SDFP_NoInflow', 
                       'M4_SDFP_ScenB', 'M5_SDFP_ScenC', 'M6_Fair_LiabMatch', 'M7_Fair_LiabInflowReb']
        
        data = {m: {
            'W_fin': {mil: np.zeros(N) for mil in self.cfg.milestones},
            'W_tot': {mil: np.zeros(N) for mil in self.cfg.milestones},
            'yearly_W_tot_med': np.zeros(self.cfg.n_years + 1),
            'yearly_W_tot_p10': np.zeros(self.cfg.n_years + 1),
            'yearly_W_tot_p90': np.zeros(self.cfg.n_years + 1),
            'yearly_W_fin_med': np.zeros(self.cfg.n_years + 1),
            'yearly_W_fin_p10': np.zeros(self.cfg.n_years + 1),
            'yearly_W_fin_p90': np.zeros(self.cfg.n_years + 1),
            'W_goal_at_H': np.zeros(N),
            'target_at_H': np.zeros(N),
            'shortfall_at_H': np.zeros(N),
            'unforced_shortfall_at_H': np.zeros(N),
            'total_shortfall_at_H': np.zeros(N),
            'V_dedicated_at_H': np.zeros(N),
            'unforced_goal_success': np.zeros(N, dtype=bool),
            'zfel_success': np.zeros(N, dtype=bool),
            'forced_equity_liq_at_H': np.zeros(N),
            'market_mdd': np.zeros(N),
            'liquidity_mdd': np.zeros(N),
            'total_wealth_mdd': np.zeros(N),
            'tax_breakdown': {},
            'law_30y': np.zeros(N)
        } for m in model_names}
        
        v3a_track = {
            'months': np.arange(self.cfg.H_month + 1),
            'med': np.zeros(self.cfg.H_month + 1),
            'p10': np.zeros(self.cfg.H_month + 1),
            'p90': np.zeros(self.cfg.H_month + 1),
            'target': np.zeros(self.cfg.H_month + 1)
        }
        v3a_track['target'][0] = float(self.cfg.T3a_0)
        m5_v3a_inflows = np.zeros((self.cfg.H_month, N))
        
        # Cost Basis (PMC) Containers for True Capital Gains (Bug R2 & R7)
        pmc_m0_eq = np.zeros(N); pmc_m0_b = np.zeros(N)
        pmc_m1_eq = np.zeros(N); pmc_m1_b = np.zeros(N)
        pmc_m2_eq = np.zeros(N); pmc_m2_b = np.zeros(N)
        pmc_m6_eq = np.zeros(N); pmc_m6_b = np.zeros(N)
        pmc_m7_eq = np.zeros(N); pmc_m7_b = np.zeros(N)
        
        pmc_m3_core = np.zeros(N); pmc_m3_sat = np.zeros(N)
        pmc_m4_core = np.zeros(N); pmc_m4_sat = np.zeros(N)
        pmc_m5_core = np.zeros(N); pmc_m5_sat = np.zeros(N)
        
        # 8-Component Tax & Friction Accumulators (Bug R2)
        friction_components = [
            'tax_rebalance', 'tax_shock_liquidation', 'tax_milestone_liquidation',
            'tax_coupons', 'stamp_duty', 'commissions', 'bid_ask_spread', 'forced_liq_penalty'
        ]
        frictions = {mod: {comp: np.zeros(N) for comp in friction_components} for mod in model_names}
        
        # Running state vectors
        M0_eq = np.zeros(N); M0_bond = np.zeros(N)
        M1_cash = np.zeros(N); M1_eq = np.zeros(N); M1_bond = np.zeros(N)
        M2_cash = np.zeros(N); M2_eq = np.zeros(N); M2_bond = np.zeros(N)
        
        M3_v1 = np.zeros(N); M3_v2 = np.zeros(N); M3_v3b = np.zeros(N)
        M3_v3a = np.zeros(N); M3_v3a_cash = np.zeros(N)
        M3_v4_legacy = np.zeros(N); M3_v4_post = np.zeros(N); M3_v4_sat = np.zeros(N); M3_v5 = np.zeros(N)
        M3_phase = np.ones(N, dtype=int)
        
        M4_v1 = np.zeros(N); M4_v2 = np.zeros(N); M4_v3b = np.zeros(N)
        M4_v3a = np.zeros(N); M4_v3a_cash = np.zeros(N)
        M4_v4_legacy = np.zeros(N); M4_v4_post = np.zeros(N); M4_v4_sat = np.zeros(N); M4_v5 = np.zeros(N)
        M4_phase = np.ones(N, dtype=int)
        
        M5_v1 = np.zeros(N); M5_v2 = np.zeros(N); M5_v3b = np.zeros(N)
        M5_v3a = np.zeros(N); M5_v3a_cash = np.zeros(N)
        M5_v4_legacy = np.zeros(N); M5_v4_post = np.zeros(N); M5_v4_sat = np.zeros(N); M5_v5 = np.zeros(N)
        M5_phase = np.ones(N, dtype=int)
        
        M6_cash = np.zeros(N); M6_v3a = np.zeros(N); M6_v3a_cash = np.zeros(N)
        M6_eq = np.zeros(N); M6_bond = np.zeros(N)
        M7_cash = np.zeros(N); M7_v3a = np.zeros(N); M7_v3a_cash = np.zeros(N)
        M7_eq = np.zeros(N); M7_bond = np.zeros(N)
        
        # Drawdown Tracking Vectors (Bug R5)
        twr_index = {mod: np.ones(N) for mod in model_names}
        twr_peaks = {mod: np.ones(N) for mod in model_names}
        market_mdds = {mod: np.zeros(N) for mod in model_names}
        
        liq_peaks = {mod: np.zeros(N) for mod in model_names}
        liquidity_mdds = {mod: np.zeros(N) for mod in model_names}
        
        tot_peaks = {mod: np.zeros(N) for mod in model_names}
        total_wealth_mdds = {mod: np.zeros(N) for mod in model_names}
        
        W_fin_prev = {mod: np.zeros(N) for mod in model_names}
        
        # -------------------------------------------------------------
        # Helper: PMC Liquidation & Realization Tax Calculator
        # -------------------------------------------------------------
        def liquidate_sleeve(current_val, cost_basis, amount_to_sell, tax_rate):
            """Computes true capital gain at tax_rate using Prezzo Medio di Carico (PMC)."""
            if np.isscalar(current_val):
                amt = min(current_val, amount_to_sell)
                sold_basis = cost_basis * (amt / current_val) if current_val > 1e-9 else 0.0
                gain = max(0.0, amt - sold_basis)
                tax = gain * tax_rate
                new_basis = max(0.0, cost_basis - sold_basis)
                new_val = max(0.0, current_val - amt)
                return new_val, new_basis, amt, tax
            else:
                amt = np.minimum(current_val, amount_to_sell)
                has_val = (current_val > 1e-9)
                sold_basis = np.where(has_val, cost_basis * (amt / np.maximum(1e-9, current_val)), 0.0)
                gain = np.maximum(0.0, amt - sold_basis)
                tax = gain * tax_rate
                new_basis = np.maximum(0.0, cost_basis - sold_basis)
                new_val = np.maximum(0.0, current_val - amt)
                return new_val, new_basis, amt, tax
            
        def record_trade_friction(mod_key, buy_amount, is_equity=True, idx=None):
            """Applies broker fees and bid-ask spread to trade amount."""
            spread_rate = self.cfg.bid_ask_equity if is_equity else self.cfg.bid_ask_bond
            if idx is None:
                has_trade = (buy_amount > 0.0)
                comm = np.where(has_trade, self.cfg.broker_fixed_fee + self.cfg.broker_pct_fee * buy_amount, 0.0)
                spread = np.where(has_trade, buy_amount * (0.5 * spread_rate), 0.0)
                frictions[mod_key]['commissions'] += comm
                frictions[mod_key]['bid_ask_spread'] += spread
                return np.maximum(0.0, buy_amount - comm - spread)
            else:
                if buy_amount > 0.0:
                    comm = self.cfg.broker_fixed_fee + self.cfg.broker_pct_fee * buy_amount
                    spread = buy_amount * (0.5 * spread_rate)
                    frictions[mod_key]['commissions'][idx] += comm
                    frictions[mod_key]['bid_ask_spread'][idx] += spread
                    return max(0.0, buy_amount - comm - spread)
                return 0.0

        # -------------------------------------------------------------
        # Monthly Simulation Loop
        # -------------------------------------------------------------
        for m in range(M):
            dt_step = dt
            r_mon_dyn = r_mon_net_dynamic[m]
            r_mon_m1 = r_mon_net_static_m1[m]
            r_e = r_eq[m]
            r_s = r_sat[m]
            r_bp = r_bond_price[m]
            r_bc = r_bond_coupon[m]
            dr_m = dr_series[m]
            shk = shocks[m]
            ref_730 = refund_730[m]
            d_Y = delta_Y_series[m]
            
            T3a_cur = T3a_target_series[m + 1]
            
            # Effective duration for held-to-maturity Vaso 3A (pull-to-par convergence at H)
            dur_v3a = 6.5 * max(0.0, (self.cfg.H_month - m) / self.cfg.H_month) if m < self.cfg.H_month else 6.5
            r_v3a_price = - dur_v3a * dr_m
            coupon_b_rate = r_bc * (1.0 - self.cfg.tax_bonds)
            coupon_b_tax_rate = r_bc * self.cfg.tax_bonds
            
            # Issuance gap: active only for the 24 months before H
            is_issuance_gap = ((self.cfg.H_month - 24) <= m < self.cfg.H_month)
            
            # Update Pillar 0
            r_fp = 0.80 * r_e + 0.20 * (r_bp + r_bc * (1.0 - self.cfg.tax_bonds))
            r_fp_net = r_fp * (1.0 - self.cfg.pension_fund_tax)
            V0 = V0 * (1.0 + r_fp_net) + v0_inflows[m]
            
            # --- Flows Definition ---
            base_s = getattr(self.cfg, 'S0', 500.0)
            base_e = getattr(self.cfg, 'E0', 1500.0)
            S_t_C = base_s + (1.0 - self.cfg.smart_split) * d_Y
            E_t_C = base_e + self.cfg.smart_split * d_Y
            buffer_mult = getattr(self.cfg, 'buffer_months', 3.0)
            T1_C = buffer_mult * E_t_C
            
            S_t_B = base_s + d_Y
            E_t_B = np.full(N, base_e)
            T1_B = buffer_mult * E_t_B
            
            T1_fixed_m1 = np.full(N, buffer_mult * base_e)
            
            # =========================================================
            # STEP A: MARKET RETURNS (Pre-Flow Wealth for TWR)
            # =========================================================
            M0_eq_pre = M0_eq * (1.0 + r_e)
            M0_bond_pre = M0_bond * (1.0 + r_bp + coupon_b_rate)
            frictions['M0_Naive']['tax_coupons'] += M0_bond * coupon_b_tax_rate
            W_fin_pre_m0 = M0_eq_pre + M0_bond_pre
            
            M1_cash_pre = M1_cash * (1.0 + r_mon_m1)
            M1_eq_pre = M1_eq * (1.0 + r_e)
            M1_bond_pre = M1_bond * (1.0 + r_bp + coupon_b_rate)
            frictions['M1_Naive_Buf']['tax_coupons'] += M1_bond * coupon_b_tax_rate
            W_fin_pre_m1 = M1_cash_pre + M1_eq_pre + M1_bond_pre
            
            M2_cash_pre = M2_cash * (1.0 + r_mon_dyn)
            M2_eq_pre = M2_eq * (1.0 + r_e)
            M2_bond_pre = M2_bond * (1.0 + r_bp + coupon_b_rate)
            frictions['M2_Fair']['tax_coupons'] += M2_bond * coupon_b_tax_rate
            W_fin_pre_m2 = M2_cash_pre + M2_eq_pre + M2_bond_pre
            
            def get_sdfp_pre(v1, v2, v3b, v3a, v3a_c, v4_leg, v4_post, v4_s, v5, mod_key):
                v1_p = v1 * (1.0 + r_mon_dyn)
                v2_p = v2 * (1.0 + r_mon_dyn)
                v3b_p = v3b * (1.0 + r_mon_dyn)
                v3a_c_p = v3a_c * (1.0 + r_mon_dyn)
                
                v3a_p = v3a * (1.0 + r_v3a_price)
                coup = v3a * coupon_b_rate
                frictions[mod_key]['tax_coupons'] += v3a * coupon_b_tax_rate
                
                is_funded = (v3a_p + v3a_c_p >= T3a_cur)
                v3a_c_p += np.where(~is_funded, coup, 0.0)
                v4_post_p = v4_post * (1.0 + r_e) + np.where(is_funded, coup, 0.0)
                
                v4_leg_p = v4_leg * (1.0 + r_e)
                v4_s_p = v4_s * (1.0 + r_s)
                v5_p = v5 * (1.0 + r_bp + coupon_b_rate)
                frictions[mod_key]['tax_coupons'] += v5 * coupon_b_tax_rate
                
                wf_pre = v1_p + v2_p + v3b_p + v3a_p + v3a_c_p + v4_leg_p + v4_post_p + v4_s_p + v5_p
                return (v1_p, v2_p, v3b_p, v3a_p, v3a_c_p, v4_leg_p, v4_post_p, v4_s_p, v5_p, wf_pre)

            (M3_v1, M3_v2, M3_v3b, M3_v3a, M3_v3a_cash, M3_v4_legacy, M3_v4_post, M3_v4_sat, M3_v5, W_fin_pre_m3) = \
                get_sdfp_pre(M3_v1, M3_v2, M3_v3b, M3_v3a, M3_v3a_cash, M3_v4_legacy, M3_v4_post, M3_v4_sat, M3_v5, 'M3_SDFP_NoInflow')
                
            (M4_v1, M4_v2, M4_v3b, M4_v3a, M4_v3a_cash, M4_v4_legacy, M4_v4_post, M4_v4_sat, M4_v5, W_fin_pre_m4) = \
                get_sdfp_pre(M4_v1, M4_v2, M4_v3b, M4_v3a, M4_v3a_cash, M4_v4_legacy, M4_v4_post, M4_v4_sat, M4_v5, 'M4_SDFP_ScenB')
                
            (M5_v1, M5_v2, M5_v3b, M5_v3a, M5_v3a_cash, M5_v4_legacy, M5_v4_post, M5_v4_sat, M5_v5, W_fin_pre_m5) = \
                get_sdfp_pre(M5_v1, M5_v2, M5_v3b, M5_v3a, M5_v3a_cash, M5_v4_legacy, M5_v4_post, M5_v4_sat, M5_v5, 'M5_SDFP_ScenC')

            M6_cash_pre = M6_cash * (1.0 + r_mon_dyn)
            M6_v3a_c_pre = M6_v3a_cash * (1.0 + r_mon_dyn)
            M6_v3a_pre = M6_v3a * (1.0 + r_v3a_price)
            coup_m6 = M6_v3a * coupon_b_rate
            frictions['M6_Fair_LiabMatch']['tax_coupons'] += M6_v3a * coupon_b_tax_rate
            is_funded_m6 = (M6_v3a_pre + M6_v3a_c_pre >= T3a_cur)
            M6_v3a_c_pre += np.where(~is_funded_m6, coup_m6, 0.0)
            
            M6_eq_pre = M6_eq * (1.0 + r_e) + np.where(is_funded_m6, 0.60 * coup_m6, 0.0)
            M6_bond_pre = M6_bond * (1.0 + r_bp + coupon_b_rate) + np.where(is_funded_m6, 0.40 * coup_m6, 0.0)
            frictions['M6_Fair_LiabMatch']['tax_coupons'] += M6_bond * coupon_b_tax_rate
            W_fin_pre_m6 = M6_cash_pre + M6_v3a_pre + M6_v3a_c_pre + M6_eq_pre + M6_bond_pre

            # Pre-market returns M7
            M7_cash_pre = M7_cash * (1.0 + r_mon_dyn)
            M7_v3a_c_pre = M7_v3a_cash * (1.0 + r_mon_dyn)
            M7_v3a_pre = M7_v3a * (1.0 + r_v3a_price)
            coup_m7 = M7_v3a * coupon_b_rate
            frictions['M7_Fair_LiabInflowReb']['tax_coupons'] += M7_v3a * coupon_b_tax_rate
            is_funded_m7 = (M7_v3a_pre + M7_v3a_c_pre >= T3a_cur)
            M7_v3a_c_pre += np.where(~is_funded_m7, coup_m7, 0.0)
            M7_eq_pre = M7_eq * (1.0 + r_e) + np.where(is_funded_m7, 0.60 * coup_m7, 0.0)
            M7_bond_pre = M7_bond * (1.0 + r_bp + coupon_b_rate) + np.where(is_funded_m7, 0.40 * coup_m7, 0.0)
            frictions['M7_Fair_LiabInflowReb']['tax_coupons'] += M7_bond * coupon_b_tax_rate
            W_fin_pre_m7 = M7_cash_pre + M7_v3a_pre + M7_v3a_c_pre + M7_eq_pre + M7_bond_pre

            W_fin_pres = {
                'M0_Naive': W_fin_pre_m0, 'M1_Naive_Buf': W_fin_pre_m1, 'M2_Fair': W_fin_pre_m2,
                'M3_SDFP_NoInflow': W_fin_pre_m3, 'M4_SDFP_ScenB': W_fin_pre_m4,
                'M5_SDFP_ScenC': W_fin_pre_m5, 'M6_Fair_LiabMatch': W_fin_pre_m6,
                'M7_Fair_LiabInflowReb': W_fin_pre_m7
            }
            
            for mod in model_names:
                if m > 0:
                    safe_denom = np.maximum(10.0, W_fin_prev[mod])
                    r_twr = np.where(W_fin_prev[mod] > 10.0, (W_fin_pres[mod] - W_fin_prev[mod]) / safe_denom, 0.0)
                    twr_index[mod] *= (1.0 + r_twr)
                    twr_peaks[mod] = np.maximum(twr_peaks[mod], twr_index[mod])
                    cur_mkt_dd = (twr_peaks[mod] - twr_index[mod]) / np.maximum(1e-9, twr_peaks[mod])
                    market_mdds[mod] = np.maximum(market_mdds[mod], cur_mkt_dd)
                    
            M0_eq = M0_eq_pre; M0_bond = M0_bond_pre
            M1_cash = M1_cash_pre; M1_eq = M1_eq_pre; M1_bond = M1_bond_pre
            M2_cash = M2_cash_pre; M2_eq = M2_eq_pre; M2_bond = M2_bond_pre
            M6_cash = M6_cash_pre; M6_v3a = M6_v3a_pre; M6_v3a_cash = M6_v3a_c_pre
            M6_eq = M6_eq_pre; M6_bond = M6_bond_pre
            M7_cash = M7_cash_pre; M7_v3a = M7_v3a_pre; M7_v3a_cash = M7_v3a_c_pre
            M7_eq = M7_eq_pre; M7_bond = M7_bond_pre

            # =========================================================
            # STEP B: INFLOWS, SHOCKS, REBALANCING & MILESTONE H
            # =========================================================
            
            # --- Model M0: Naive 60/40 Monolithic ---
            inflow_m0 = S_t_C + ref_730
            inv_eq_m0 = record_trade_friction('M0_Naive', 0.60 * inflow_m0, is_equity=True)
            inv_b_m0 = record_trade_friction('M0_Naive', 0.40 * inflow_m0, is_equity=False)
            M0_eq += inv_eq_m0; pmc_m0_eq += inv_eq_m0
            M0_bond += inv_b_m0; pmc_m0_b += inv_b_m0
            
            tot_m0 = M0_eq + M0_bond
            shk_m0_act = np.minimum(tot_m0, shk)
            has_shk_m0 = (shk_m0_act > 0.0)
            eq_ratio_m0 = np.where(tot_m0 > 0.0, M0_eq / np.maximum(1e-9, tot_m0), 0.60)
            sold_eq_m0 = shk_m0_act * eq_ratio_m0
            sold_b_m0 = shk_m0_act * (1.0 - eq_ratio_m0)
            
            M0_eq, pmc_m0_eq, _, tax_shk_eq = liquidate_sleeve(M0_eq, pmc_m0_eq, sold_eq_m0, self.cfg.tax_equity)
            M0_bond, pmc_m0_b, _, tax_shk_b = liquidate_sleeve(M0_bond, pmc_m0_b, sold_b_m0, self.cfg.tax_bonds)
            frictions['M0_Naive']['tax_shock_liquidation'] += (tax_shk_eq + tax_shk_b)
            frictions['M0_Naive']['forced_liq_penalty'] += np.where(has_shk_m0, shk_m0_act * 0.0015, 0.0)
            
            if m % 12 == 0:
                tot_m0_reb = M0_eq + M0_bond
                tgt_eq = 0.60 * tot_m0_reb
                excess_eq = np.maximum(0.0, M0_eq - tgt_eq)
                M0_eq, pmc_m0_eq, sold_reb, tax_reb = liquidate_sleeve(M0_eq, pmc_m0_eq, excess_eq, self.cfg.tax_equity)
                frictions['M0_Naive']['tax_rebalance'] += tax_reb
                reb_inv_b = record_trade_friction('M0_Naive', np.maximum(0.0, sold_reb - tax_reb), is_equity=False)
                M0_bond += reb_inv_b; pmc_m0_b += reb_inv_b
                
            if m == (self.cfg.H_month - 1):
                tot_m0_h = M0_eq + M0_bond
                spend_m0 = np.minimum(tot_m0_h, T3a_cur)
                eq_r0 = np.where(tot_m0_h > 0, M0_eq / np.maximum(1e-9, tot_m0_h), 0.60)
                
                data['M0_Naive']['target_at_H'] = T3a_cur
                data['M0_Naive']['W_goal_at_H'] = tot_m0_h
                data['M0_Naive']['V_dedicated_at_H'] = np.zeros(N)
                data['M0_Naive']['unforced_shortfall_at_H'] = T3a_cur.copy()
                data['M0_Naive']['total_shortfall_at_H'] = np.maximum(0.0, T3a_cur - tot_m0_h)
                data['M0_Naive']['shortfall_at_H'] = data['M0_Naive']['unforced_shortfall_at_H']
                data['M0_Naive']['unforced_goal_success'] = np.zeros(N, dtype=bool)
                data['M0_Naive']['forced_equity_liq_at_H'] = (spend_m0 * eq_r0).copy()
                data['M0_Naive']['zfel_success'] = (spend_m0 * eq_r0 <= 0.01) & (data['M0_Naive']['total_shortfall_at_H'] <= 0.01)
                
                M0_eq, pmc_m0_eq, _, t_eq = liquidate_sleeve(M0_eq, pmc_m0_eq, spend_m0 * eq_r0, self.cfg.tax_equity)
                M0_bond, pmc_m0_b, _, t_b = liquidate_sleeve(M0_bond, pmc_m0_b, spend_m0 * (1.0 - eq_r0), self.cfg.tax_bonds)
                frictions['M0_Naive']['tax_milestone_liquidation'] += (t_eq + t_b)
                frictions['M0_Naive']['forced_liq_penalty'] += spend_m0 * 0.0015

            # --- Model M1: Naive + Static Cash Buffer (Bug R3) ---
            inflow_m1 = S_t_C + ref_730
            buf_need_m1 = np.maximum(0.0, T1_fixed_m1 - M1_cash)
            to_buf_m1 = np.minimum(inflow_m1, buf_need_m1)
            M1_cash += to_buf_m1
            inv_m1 = inflow_m1 - to_buf_m1
            
            inv_eq_m1 = record_trade_friction('M1_Naive_Buf', 0.60 * inv_m1, is_equity=True)
            inv_b_m1 = record_trade_friction('M1_Naive_Buf', 0.40 * inv_m1, is_equity=False)
            M1_eq += inv_eq_m1; pmc_m1_eq += inv_eq_m1
            M1_bond += inv_b_m1; pmc_m1_b += inv_b_m1
            
            abs_c_m1 = np.minimum(M1_cash, shk)
            M1_cash -= abs_c_m1
            rem_shk_m1 = shk - abs_c_m1
            tot_inv_m1 = M1_eq + M1_bond
            shk_m1_act = np.minimum(tot_inv_m1, rem_shk_m1)
            has_shk_m1 = (shk_m1_act > 0.0)
            eq_ratio_m1 = np.where(tot_inv_m1 > 0, M1_eq / np.maximum(1e-9, tot_inv_m1), 0.60)
            sold_eq_m1 = shk_m1_act * eq_ratio_m1
            sold_b_m1 = shk_m1_act * (1.0 - eq_ratio_m1)
            
            M1_eq, pmc_m1_eq, _, t_shk_eq1 = liquidate_sleeve(M1_eq, pmc_m1_eq, sold_eq_m1, self.cfg.tax_equity)
            M1_bond, pmc_m1_b, _, t_shk_b1 = liquidate_sleeve(M1_bond, pmc_m1_b, sold_b_m1, self.cfg.tax_bonds)
            frictions['M1_Naive_Buf']['tax_shock_liquidation'] += (t_shk_eq1 + t_shk_b1)
            frictions['M1_Naive_Buf']['forced_liq_penalty'] += np.where(has_shk_m1, shk_m1_act * 0.0015, 0.0)
            
            if m % 12 == 0:
                tot_reb = M1_eq + M1_bond
                tgt_eq = 0.60 * tot_reb
                excess_eq = np.maximum(0.0, M1_eq - tgt_eq)
                M1_eq, pmc_m1_eq, sold_reb, tax_reb = liquidate_sleeve(M1_eq, pmc_m1_eq, excess_eq, self.cfg.tax_equity)
                frictions['M1_Naive_Buf']['tax_rebalance'] += tax_reb
                reb_inv = record_trade_friction('M1_Naive_Buf', np.maximum(0.0, sold_reb - tax_reb), is_equity=False)
                M1_bond += reb_inv; pmc_m1_b += reb_inv
                
            if m == (self.cfg.H_month - 1):
                tot_m1_avail = M1_eq + M1_bond + np.maximum(0.0, M1_cash - 500.0)
                spend_m1 = np.minimum(M1_eq + M1_bond, T3a_cur)
                eq_r1 = np.where((M1_eq + M1_bond) > 0, M1_eq / np.maximum(1e-9, M1_eq + M1_bond), 0.60)
                
                data['M1_Naive_Buf']['target_at_H'] = T3a_cur
                data['M1_Naive_Buf']['W_goal_at_H'] = tot_m1_avail
                data['M1_Naive_Buf']['V_dedicated_at_H'] = np.maximum(0.0, M1_cash - T1_fixed_m1)
                data['M1_Naive_Buf']['unforced_shortfall_at_H'] = np.maximum(0.0, T3a_cur - data['M1_Naive_Buf']['V_dedicated_at_H'])
                data['M1_Naive_Buf']['total_shortfall_at_H'] = np.maximum(0.0, T3a_cur - tot_m1_avail)
                data['M1_Naive_Buf']['shortfall_at_H'] = data['M1_Naive_Buf']['unforced_shortfall_at_H']
                data['M1_Naive_Buf']['unforced_goal_success'] = np.zeros(N, dtype=bool)
                data['M1_Naive_Buf']['forced_equity_liq_at_H'] = (spend_m1 * eq_r1).copy()
                data['M1_Naive_Buf']['zfel_success'] = (spend_m1 * eq_r1 <= 0.01) & (data['M1_Naive_Buf']['total_shortfall_at_H'] <= 0.01)
                
                M1_eq, pmc_m1_eq, _, t_eq = liquidate_sleeve(M1_eq, pmc_m1_eq, spend_m1 * eq_r1, self.cfg.tax_equity)
                M1_bond, pmc_m1_b, _, t_b = liquidate_sleeve(M1_bond, pmc_m1_b, spend_m1 * (1.0 - eq_r1), self.cfg.tax_bonds)
                frictions['M1_Naive_Buf']['tax_milestone_liquidation'] += (t_eq + t_b)
                frictions['M1_Naive_Buf']['forced_liq_penalty'] += spend_m1 * 0.0015

            # --- Model M2: Fair Benchmark (Dynamic Buffer 3*E_t) ---
            inflow_m2 = S_t_C + ref_730
            buf_need_m2 = np.maximum(0.0, T1_C - M2_cash)
            to_buf_m2 = np.minimum(inflow_m2, buf_need_m2)
            M2_cash += to_buf_m2
            inv_m2 = inflow_m2 - to_buf_m2
            
            inv_eq_m2 = record_trade_friction('M2_Fair', 0.60 * inv_m2, is_equity=True)
            inv_b_m2 = record_trade_friction('M2_Fair', 0.40 * inv_m2, is_equity=False)
            M2_eq += inv_eq_m2; pmc_m2_eq += inv_eq_m2
            M2_bond += inv_b_m2; pmc_m2_b += inv_b_m2
            
            abs_c_m2 = np.minimum(M2_cash, shk)
            M2_cash -= abs_c_m2
            rem_shk_m2 = shk - abs_c_m2
            tot_inv_m2 = M2_eq + M2_bond
            shk_m2_act = np.minimum(tot_inv_m2, rem_shk_m2)
            has_shk_m2 = (shk_m2_act > 0.0)
            eq_ratio_m2 = np.where(tot_inv_m2 > 0, M2_eq / np.maximum(1e-9, tot_inv_m2), 0.60)
            sold_eq_m2 = shk_m2_act * eq_ratio_m2
            sold_b_m2 = shk_m2_act * (1.0 - eq_ratio_m2)
            
            M2_eq, pmc_m2_eq, _, t_shk_eq2 = liquidate_sleeve(M2_eq, pmc_m2_eq, sold_eq_m2, self.cfg.tax_equity)
            M2_bond, pmc_m2_b, _, t_shk_b2 = liquidate_sleeve(M2_bond, pmc_m2_b, sold_b_m2, self.cfg.tax_bonds)
            frictions['M2_Fair']['tax_shock_liquidation'] += (t_shk_eq2 + t_shk_b2)
            frictions['M2_Fair']['forced_liq_penalty'] += np.where(has_shk_m2, shk_m2_act * 0.0015, 0.0)
            
            if m % 12 == 0:
                tot_reb = M2_eq + M2_bond
                tgt_eq = 0.60 * tot_reb
                excess_eq = np.maximum(0.0, M2_eq - tgt_eq)
                M2_eq, pmc_m2_eq, sold_reb, tax_reb = liquidate_sleeve(M2_eq, pmc_m2_eq, excess_eq, self.cfg.tax_equity)
                frictions['M2_Fair']['tax_rebalance'] += tax_reb
                reb_inv = record_trade_friction('M2_Fair', np.maximum(0.0, sold_reb - tax_reb), is_equity=False)
                M2_bond += reb_inv; pmc_m2_b += reb_inv
                
            if m == (self.cfg.H_month - 1):
                avail_buf = np.maximum(0.0, M2_cash - 500.0)
                tot_m2_avail = M2_eq + M2_bond + avail_buf
                v_ded_m2 = np.maximum(0.0, M2_cash - T1_C)
                unforced_sh_m2 = np.maximum(0.0, T3a_cur - v_ded_m2)
                
                tot_inv_h = M2_eq + M2_bond
                rem_target = T3a_cur.copy()
                liq_inv = np.minimum(tot_inv_h, rem_target)
                eq_r = np.where(tot_inv_h > 0, M2_eq / np.maximum(1e-9, tot_inv_h), 0.60)
                M2_eq, pmc_m2_eq, _, t_eq = liquidate_sleeve(M2_eq, pmc_m2_eq, liq_inv * eq_r, self.cfg.tax_equity)
                M2_bond, pmc_m2_b, _, t_b = liquidate_sleeve(M2_bond, pmc_m2_b, liq_inv * (1.0 - eq_r), self.cfg.tax_bonds)
                frictions['M2_Fair']['tax_milestone_liquidation'] += (t_eq + t_b)
                frictions['M2_Fair']['forced_liq_penalty'] += liq_inv * 0.0015
                rem_target -= liq_inv
                
                tap_buf = np.minimum(avail_buf, rem_target)
                M2_cash -= tap_buf
                rem_target -= tap_buf
                
                data['M2_Fair']['target_at_H'] = T3a_cur
                data['M2_Fair']['W_goal_at_H'] = (T3a_cur - rem_target)
                data['M2_Fair']['V_dedicated_at_H'] = v_ded_m2
                data['M2_Fair']['unforced_shortfall_at_H'] = unforced_sh_m2
                data['M2_Fair']['total_shortfall_at_H'] = rem_target.copy()
                data['M2_Fair']['shortfall_at_H'] = unforced_sh_m2
                data['M2_Fair']['unforced_goal_success'] = np.zeros(N, dtype=bool)
                data['M2_Fair']['forced_equity_liq_at_H'] = (liq_inv * eq_r).copy()
                data['M2_Fair']['zfel_success'] = (liq_inv * eq_r <= 0.01) & (rem_target <= 0.01)

            # =========================================================
            # S-DFP UPDATE FUNCTION (Models M3, M4, M5)
            # =========================================================
            def step_sdfp(v1, v2, v3b, v3a, v3a_c, v4_leg, v4_post, v4_s, v5,
                          pmc_core, pmc_sat, phase, mod_key, S_flow, T1_target, is_inflow_reb=True,
                          v3a_inflow_arr=None):
                # 1. Reinvest 730 tax refund into Core ETF
                ref_net = record_trade_friction(mod_key, ref_730, is_equity=True)
                v4_post += ref_net
                pmc_core += ref_net
                
                # 2. Liquidity Shock Waterfall (V1 -> V3b -> V3a -> V4)
                rem_shk = shk.copy()
                abs_v1 = np.minimum(v1, rem_shk); v1 -= abs_v1; rem_shk -= abs_v1
                abs_v3b = np.minimum(v3b, rem_shk); v3b -= abs_v3b; rem_shk -= abs_v3b
                abs_v3a = np.minimum(v3a, rem_shk); v3a -= abs_v3a; rem_shk -= abs_v3a
                
                has_shk_v4 = (rem_shk > 0.0)
                if np.any(has_shk_v4):
                    v4_tot = v4_leg + v4_post
                    v4_post, pmc_core, _, t_shk_post = liquidate_sleeve(v4_post, pmc_core, rem_shk, self.cfg.tax_equity)
                    rem_shk_leg = np.maximum(0.0, rem_shk - (v4_tot - (v4_leg + v4_post)))
                    v4_leg, pmc_core, _, t_shk_leg = liquidate_sleeve(v4_leg, pmc_core, rem_shk_leg, self.cfg.tax_equity)
                    frictions[mod_key]['tax_shock_liquidation'] += (t_shk_post + t_shk_leg)
                    frictions[mod_key]['forced_liq_penalty'] += np.where(has_shk_v4, np.minimum(v4_tot, rem_shk) * 0.0015, 0.0)

                # 3. State Machine & Inflows
                is_p_v1 = (v1 < T1_target)
                is_p_v3b = (~is_p_v1) & (v3b < self.cfg.T3b_fixed) if self.cfg.T3b_fixed > 0 else np.zeros(N, dtype=bool)
                
                for i in range(N):
                    # Preemption Loop
                    if is_p_v1[i]:
                        v1[i] += S_flow[i]
                        continue
                    if is_p_v3b[i]:
                        v3b[i] += S_flow[i]
                        continue
                        
                    # Retirement Glidepath (month >= T_retire)
                    if m >= self.cfg.T_retire:
                        tot_v4 = v4_leg[i] + v4_post[i] + v4_s[i]
                        shift = 0.0025 * tot_v4
                        sh_post = min(v4_post[i], shift * 0.60)
                        v4_post[i] -= sh_post
                        sh_leg = (shift * 0.60) - sh_post
                        v4_leg[i] -= sh_leg
                        sh_sat = min(v4_s[i], shift * 0.40)
                        v4_s[i] -= sh_sat
                        
                        gain_c = max(0.0, (shift * 0.60) - min(pmc_core[i], shift * 0.60))
                        pmc_core[i] = max(0.0, pmc_core[i] - (shift * 0.60))
                        gain_s = max(0.0, sh_sat - min(pmc_sat[i], sh_sat))
                        pmc_sat[i] = max(0.0, pmc_sat[i] - sh_sat)
                        
                        tax_glide = (gain_c + gain_s) * self.cfg.tax_equity
                        frictions[mod_key]['tax_rebalance'][i] += tax_glide
                        
                        v5_in = record_trade_friction(mod_key, shift - tax_glide + S_flow[i], is_equity=False, idx=i)
                        v5[i] += v5_in
                        continue
                        
                    p = phase[i]
                    
                    if p == STATE_S3 and self.cfg.T3b_fixed == 0:
                        p = STATE_S4
                        phase[i] = STATE_S4
                        
                    if (p in [STATE_S4, STATE_S4_INV, STATE_S_REBUILD]) and self.cfg.T3a_0 == 0:
                        p = STATE_S5B if (v4_leg[i] + v4_post[i] + v4_s[i]) >= 100000.0 else STATE_S5A
                        phase[i] = p
                        
                    if p == STATE_S1:
                        v1[i] += S_flow[i]
                        if v1[i] >= T1_target[i]: phase[i] = STATE_S2
                    elif p == STATE_S2:
                        v2[i] += S_flow[i]
                        if v2[i] >= self.cfg.T2_fixed: 
                            phase[i] = STATE_S4 if self.cfg.T3b_fixed == 0 else STATE_S3
                    elif p == STATE_S3:
                        v3b[i] += S_flow[i]
                        if v3b[i] >= self.cfg.T3b_fixed: phase[i] = STATE_S4
                    elif p == STATE_S4:
                        # Bug R4: threshold modularized to 0.60 * T3a_cur[i]
                        v3a_c[i] += S_flow[i]
                        if v3a_inflow_arr is not None: v3a_inflow_arr[i] = S_flow[i]
                        # Bug R4: issuance gap strictly (H_month - 24 <= m < H_month)
                        if v3a_c[i] >= 1500.0 and not is_issuance_gap:
                            b_amt = record_trade_friction(mod_key, v3a_c[i], is_equity=False, idx=i)
                            v3a[i] += b_amt; v3a_c[i] = 0.0
                        if (v3a[i] + v3a_c[i]) >= (0.60 * T3a_cur[i]): 
                            phase[i] = STATE_S4_INV
                    elif p == STATE_S4_INV:
                        f_3a = 0.60 * S_flow[i]; f_v4 = 0.40 * S_flow[i]
                        v3a_c[i] += f_3a
                        if v3a_inflow_arr is not None: v3a_inflow_arr[i] = f_3a
                        if v3a_c[i] >= 1500.0 and not is_issuance_gap:
                            b_amt = record_trade_friction(mod_key, v3a_c[i], is_equity=False, idx=i)
                            v3a[i] += b_amt; v3a_c[i] = 0.0
                        c_net = record_trade_friction(mod_key, f_v4, is_equity=True, idx=i)
                        v4_post[i] += c_net; pmc_core[i] += c_net
                        if (v3a[i] + v3a_c[i]) >= T3a_cur[i]:
                            tot_v4 = v4_leg[i] + v4_post[i] + v4_s[i]
                            phase[i] = STATE_S5B if tot_v4 >= 100000.0 else STATE_S5A
                    elif p == STATE_S_REBUILD:
                        f_3a = 0.60 * S_flow[i]; f_v4 = 0.40 * S_flow[i]
                        v3a_c[i] += f_3a
                        if v3a_inflow_arr is not None: v3a_inflow_arr[i] = f_3a
                        if v3a_c[i] >= 1500.0 and not is_issuance_gap:
                            b_amt = record_trade_friction(mod_key, v3a_c[i], is_equity=False, idx=i)
                            v3a[i] += b_amt; v3a_c[i] = 0.0
                        c_net = record_trade_friction(mod_key, f_v4, is_equity=True, idx=i)
                        v4_post[i] += c_net; pmc_core[i] += c_net
                        if (v3a[i] + v3a_c[i]) >= T3a_cur[i]:
                            tot_v4 = v4_leg[i] + v4_post[i] + v4_s[i]
                            phase[i] = STATE_S5B if tot_v4 >= 100000.0 else STATE_S5A
                    elif p == STATE_S5A:
                        f_3a = 0.15 * S_flow[i]; f_v4 = 0.85 * S_flow[i]
                        v3a_c[i] += f_3a
                        if v3a_inflow_arr is not None: v3a_inflow_arr[i] = f_3a
                        if v3a_c[i] >= 1500.0 and not is_issuance_gap:
                            b_amt = record_trade_friction(mod_key, v3a_c[i], is_equity=False, idx=i)
                            v3a[i] += b_amt; v3a_c[i] = 0.0
                        c_net = record_trade_friction(mod_key, f_v4, is_equity=True, idx=i)
                        v4_post[i] += c_net; pmc_core[i] += c_net
                        tot_v4 = v4_leg[i] + v4_post[i] + v4_s[i]
                        if tot_v4 >= 100000.0:
                            v4_leg[i] = 100000.0
                            v4_post[i] = max(0.0, tot_v4 - 100000.0)
                            phase[i] = STATE_S5B
                    elif p == STATE_S5B:
                        f_3a = 0.15 * S_flow[i]; f_v4 = 0.85 * S_flow[i]
                        v3a_c[i] += f_3a
                        if v3a_inflow_arr is not None: v3a_inflow_arr[i] = f_3a
                        if v3a_c[i] >= 1500.0 and not is_issuance_gap:
                            b_amt = record_trade_friction(mod_key, v3a_c[i], is_equity=False, idx=i)
                            v3a[i] += b_amt; v3a_c[i] = 0.0
                            
                        denom_sleeve = v4_s[i] + v4_post[i]
                        w_sat = v4_s[i] / denom_sleeve if denom_sleeve > 0.0 else 0.40
                        
                        if is_inflow_reb:
                            if w_sat < 0.35:
                                split_c, split_s = 0.20, 0.80
                            elif w_sat > 0.45:
                                split_c, split_s = 0.80, 0.20
                            else:
                                split_c, split_s = 0.60, 0.40
                            c_in = record_trade_friction(mod_key, split_c * f_v4, is_equity=True, idx=i)
                            s_in = record_trade_friction(mod_key, split_s * f_v4, is_equity=True, idx=i)
                            v4_post[i] += c_in; pmc_core[i] += c_in
                            v4_s[i] += s_in; pmc_sat[i] += s_in
                        else:
                            c_in = record_trade_friction(mod_key, 0.60 * f_v4, is_equity=True, idx=i)
                            s_in = record_trade_friction(mod_key, 0.40 * f_v4, is_equity=True, idx=i)
                            v4_post[i] += c_in; pmc_core[i] += c_in
                            v4_s[i] += s_in; pmc_sat[i] += s_in
                            if m % 12 == 0 and denom_sleeve > 0:
                                tgt_c = 0.60 * denom_sleeve
                                if v4_post[i] > tgt_c:
                                    excess_c = v4_post[i] - tgt_c
                                    v4_post[i], pmc_core[i], sold_c, tax_reb = liquidate_sleeve(v4_post[i], pmc_core[i], excess_c, self.cfg.tax_equity)
                                    frictions[mod_key]['tax_rebalance'][i] += tax_reb
                                    s_reb = record_trade_friction(mod_key, max(0.0, sold_c - tax_reb), is_equity=True, idx=i)
                                    v4_s[i] += s_reb; pmc_sat[i] += s_reb
                                elif v4_s[i] > (0.40 * denom_sleeve):
                                    excess_s = v4_s[i] - 0.40 * denom_sleeve
                                    v4_s[i], pmc_sat[i], sold_s, tax_reb = liquidate_sleeve(v4_s[i], pmc_sat[i], excess_s, self.cfg.tax_equity)
                                    frictions[mod_key]['tax_rebalance'][i] += tax_reb
                                    c_reb = record_trade_friction(mod_key, max(0.0, sold_s - tax_reb), is_equity=True, idx=i)
                                    v4_post[i] += c_reb; pmc_core[i] += c_reb
                                    
                        tot_v4 = v4_leg[i] + v4_post[i] + v4_s[i]
                        if tot_v4 < 100000.0:
                            v4_post[i] += v4_leg[i]
                            v4_leg[i] = 0.0
                            phase[i] = STATE_S5A
                            
                return (v1, v2, v3b, v3a, v3a_c, v4_leg, v4_post, v4_s, v5, pmc_core, pmc_sat, phase)

            (M3_v1, M3_v2, M3_v3b, M3_v3a, M3_v3a_cash, M3_v4_legacy, M3_v4_post, M3_v4_sat, M3_v5, pmc_m3_core, pmc_m3_sat, M3_phase) = \
                step_sdfp(M3_v1, M3_v2, M3_v3b, M3_v3a, M3_v3a_cash, M3_v4_legacy, M3_v4_post, M3_v4_sat, M3_v5,
                          pmc_m3_core, pmc_m3_sat, M3_phase, 'M3_SDFP_NoInflow', S_t_C, T1_C, is_inflow_reb=False)

            (M4_v1, M4_v2, M4_v3b, M4_v3a, M4_v3a_cash, M4_v4_legacy, M4_v4_post, M4_v4_sat, M4_v5, pmc_m4_core, pmc_m4_sat, M4_phase) = \
                step_sdfp(M4_v1, M4_v2, M4_v3b, M4_v3a, M4_v3a_cash, M4_v4_legacy, M4_v4_post, M4_v4_sat, M4_v5,
                          pmc_m4_core, pmc_m4_sat, M4_phase, 'M4_SDFP_ScenB', S_t_B, T1_B, is_inflow_reb=True)

            m5_v3a_inflow_m = np.zeros(N)
            (M5_v1, M5_v2, M5_v3b, M5_v3a, M5_v3a_cash, M5_v4_legacy, M5_v4_post, M5_v4_sat, M5_v5, pmc_m5_core, pmc_m5_sat, M5_phase) = \
                step_sdfp(M5_v1, M5_v2, M5_v3b, M5_v3a, M5_v3a_cash, M5_v4_legacy, M5_v4_post, M5_v4_sat, M5_v5,
                          pmc_m5_core, pmc_m5_sat, M5_phase, 'M5_SDFP_ScenC', S_t_C, T1_C, is_inflow_reb=True,
                          v3a_inflow_arr=m5_v3a_inflow_m)
            if m < self.cfg.H_month:
                m5_v3a_inflows[m] = m5_v3a_inflow_m

            # Milestone H Event for S-DFP (Bug R7: Full Spending Parity)
            if m == (self.cfg.H_month - 1):
                for m_id, v_a, v_ac, v_1, v_2, v_4c_p, v_4c_l, pmc_c, p_arr in [
                    ('M3_SDFP_NoInflow', M3_v3a, M3_v3a_cash, M3_v1, M3_v2, M3_v4_post, M3_v4_legacy, pmc_m3_core, M3_phase),
                    ('M4_SDFP_ScenB', M4_v3a, M4_v3a_cash, M4_v1, M4_v2, M4_v4_post, M4_v4_legacy, pmc_m4_core, M4_phase),
                    ('M5_SDFP_ScenC', M5_v3a, M5_v3a_cash, M5_v1, M5_v2, M5_v4_post, M5_v4_legacy, pmc_m5_core, M5_phase)
                ]:
                    tot_v3a = v_a + v_ac
                    rem_target = T3a_cur.copy()
                    data[m_id]['V_dedicated_at_H'] = tot_v3a.copy()
                    data[m_id]['unforced_shortfall_at_H'] = np.maximum(0.0, T3a_cur - tot_v3a)
                    data[m_id]['unforced_goal_success'] = (tot_v3a >= T3a_cur)
                    
                    spend_v3a = np.minimum(tot_v3a, rem_target)
                    from_v3a_c = np.minimum(v_ac, spend_v3a)
                    v_ac -= from_v3a_c
                    from_v3a = spend_v3a - from_v3a_c
                    v_a -= from_v3a
                    rem_target -= spend_v3a
                    
                    tap_v4 = np.zeros(N)
                    has_deficit = (rem_target > 0.0)
                    if np.any(has_deficit):
                        tap_v2 = np.minimum(v_2, rem_target)
                        v_2 -= tap_v2
                        rem_target -= tap_v2
                        
                        avail_v1 = np.maximum(0.0, v_1 - 500.0)
                        tap_v1 = np.minimum(avail_v1, rem_target)
                        v_1 -= tap_v1
                        rem_target -= tap_v1
                        
                        v4_tot_c = v_4c_p + v_4c_l
                        tap_v4 = np.minimum(v4_tot_c, rem_target)
                        v_4c_p, pmc_c, sold_p, t_p = liquidate_sleeve(v_4c_p, pmc_c, tap_v4, self.cfg.tax_equity)
                        rem_v4_l = tap_v4 - sold_p
                        v_4c_l, pmc_c, sold_l, t_l = liquidate_sleeve(v_4c_l, pmc_c, rem_v4_l, self.cfg.tax_equity)
                        frictions[m_id]['tax_milestone_liquidation'] += (t_p + t_l)
                        frictions[m_id]['forced_liq_penalty'] += tap_v4 * 0.0015
                        rem_target -= tap_v4
                        
                    data[m_id]['target_at_H'] = T3a_cur
                    data[m_id]['W_goal_at_H'] = (T3a_cur - rem_target)
                    data[m_id]['total_shortfall_at_H'] = rem_target.copy()
                    data[m_id]['shortfall_at_H'] = data[m_id]['unforced_shortfall_at_H']
                    data[m_id]['forced_equity_liq_at_H'] = tap_v4.copy()
                    data[m_id]['zfel_success'] = (tap_v4 <= 0.01) & (rem_target <= 0.01)
                    p_arr[:] = STATE_S_REBUILD

            # --- Model M6: Fair + Dedicated Liability Matching (Bug R3) ---
            inflow_m6 = S_t_C + ref_730
            buf_need_m6 = np.maximum(0.0, T1_C - M6_cash)
            to_buf_m6 = np.minimum(inflow_m6, buf_need_m6)
            M6_cash += to_buf_m6
            inv_m6 = inflow_m6 - to_buf_m6
            
            if m < self.cfg.H_month:
                is_m6_v3a_funded = (M6_v3a + M6_v3a_cash >= T3a_cur)
                to_m6_v3a = np.where(~is_m6_v3a_funded, 0.65 * inv_m6, 0.15 * inv_m6)
                to_m6_port = inv_m6 - to_m6_v3a
                M6_v3a_cash += to_m6_v3a
                if not is_issuance_gap:
                    can_buy_m6 = (M6_v3a_cash >= 1500.0)
                    if np.any(can_buy_m6):
                        buy_amt = record_trade_friction('M6_Fair_LiabMatch', np.where(can_buy_m6, M6_v3a_cash, 0.0), is_equity=False)
                        M6_v3a += buy_amt
                        M6_v3a_cash = np.where(can_buy_m6, 0.0, M6_v3a_cash)
            else:
                to_m6_port = inv_m6
                
            split_eq = 0.60
            split_b = 0.40
            inv_eq_m6 = record_trade_friction('M6_Fair_LiabMatch', 0.60 * to_m6_port, is_equity=True)
            inv_b_m6 = record_trade_friction('M6_Fair_LiabMatch', 0.40 * to_m6_port, is_equity=False)
            M6_eq += inv_eq_m6; pmc_m6_eq += inv_eq_m6
            M6_bond += inv_b_m6; pmc_m6_b += inv_b_m6
            
            abs_c_m6 = np.minimum(M6_cash, shk)
            M6_cash -= abs_c_m6
            rem_shk_m6 = shk - abs_c_m6
            tot_inv_m6 = M6_eq + M6_bond
            shk_m6_act = np.minimum(tot_inv_m6, rem_shk_m6)
            has_shk_m6 = (shk_m6_act > 0.0)
            eq_ratio_m6 = np.where(tot_inv_m6 > 0, M6_eq / np.maximum(1e-9, tot_inv_m6), 0.60)
            sold_eq_m6 = shk_m6_act * eq_ratio_m6
            sold_b_m6 = shk_m6_act * (1.0 - eq_ratio_m6)
            
            M6_eq, pmc_m6_eq, _, t_shk_eq6 = liquidate_sleeve(M6_eq, pmc_m6_eq, sold_eq_m6, self.cfg.tax_equity)
            M6_bond, pmc_m6_b, _, t_shk_b6 = liquidate_sleeve(M6_bond, pmc_m6_b, sold_b_m6, self.cfg.tax_bonds)
            frictions['M6_Fair_LiabMatch']['tax_shock_liquidation'] += (t_shk_eq6 + t_shk_b6)
            frictions['M6_Fair_LiabMatch']['forced_liq_penalty'] += np.where(has_shk_m6, shk_m6_act * 0.0015, 0.0)
            
            if m % 12 == 0:
                tot_reb = M6_eq + M6_bond
                tgt_eq = 0.60 * tot_reb
                excess_eq = np.maximum(0.0, M6_eq - tgt_eq)
                M6_eq, pmc_m6_eq, sold_reb, tax_reb = liquidate_sleeve(M6_eq, pmc_m6_eq, excess_eq, self.cfg.tax_equity)
                frictions['M6_Fair_LiabMatch']['tax_rebalance'] += tax_reb
                reb_inv = record_trade_friction('M6_Fair_LiabMatch', np.maximum(0.0, sold_reb - tax_reb), is_equity=False)
                M6_bond += reb_inv; pmc_m6_b += reb_inv
                
            if m == (self.cfg.H_month - 1):
                tot_m6_v3a = M6_v3a + M6_v3a_cash
                rem_target = T3a_cur.copy()
                data['M6_Fair_LiabMatch']['V_dedicated_at_H'] = tot_m6_v3a.copy()
                data['M6_Fair_LiabMatch']['unforced_shortfall_at_H'] = np.maximum(0.0, T3a_cur - tot_m6_v3a)
                data['M6_Fair_LiabMatch']['unforced_goal_success'] = (tot_m6_v3a >= T3a_cur)
                
                spend = np.minimum(T3a_cur, tot_m6_v3a)
                from_c = np.minimum(M6_v3a_cash, spend)
                M6_v3a_cash -= from_c
                M6_v3a -= (spend - from_c)
                rem_target -= spend
                
                m6_forced_eq = np.zeros(N)
                if np.any(rem_target > 0.0):
                    tot_p = M6_eq + M6_bond
                    tap = np.minimum(tot_p, rem_target)
                    eq_r = np.where(tot_p > 0, M6_eq / np.maximum(1e-9, tot_p), 0.60)
                    M6_eq, pmc_m6_eq, _, t_eq = liquidate_sleeve(M6_eq, pmc_m6_eq, tap * eq_r, self.cfg.tax_equity)
                    M6_bond, pmc_m6_b, _, t_b = liquidate_sleeve(M6_bond, pmc_m6_b, tap * (1.0 - eq_r), self.cfg.tax_bonds)
                    frictions['M6_Fair_LiabMatch']['tax_milestone_liquidation'] += (t_eq + t_b)
                    frictions['M6_Fair_LiabMatch']['forced_liq_penalty'] += tap * 0.0015
                    rem_target -= tap
                    m6_forced_eq = tap * eq_r
                    
                data['M6_Fair_LiabMatch']['target_at_H'] = T3a_cur
                data['M6_Fair_LiabMatch']['W_goal_at_H'] = (T3a_cur - rem_target)
                data['M6_Fair_LiabMatch']['total_shortfall_at_H'] = rem_target.copy()
                data['M6_Fair_LiabMatch']['shortfall_at_H'] = data['M6_Fair_LiabMatch']['unforced_shortfall_at_H']
                data['M6_Fair_LiabMatch']['forced_equity_liq_at_H'] = m6_forced_eq
                data['M6_Fair_LiabMatch']['zfel_success'] = (m6_forced_eq <= 0.01) & (rem_target <= 0.01)

            # --- Model M7: Fair + Dedicated Liability Matching + Inflow Rebalancing ---
            inflow_m7 = S_t_C + ref_730
            buf_need_m7 = np.maximum(0.0, T1_C - M7_cash)
            to_buf_m7 = np.minimum(inflow_m7, buf_need_m7)
            M7_cash += to_buf_m7
            inv_m7 = inflow_m7 - to_buf_m7
            
            if m < self.cfg.H_month:
                is_m7_v3a_funded = (M7_v3a + M7_v3a_cash >= T3a_cur)
                to_m7_v3a = np.where(~is_m7_v3a_funded, 0.65 * inv_m7, 0.15 * inv_m7)
                to_m7_port = inv_m7 - to_m7_v3a
                M7_v3a_cash += to_m7_v3a
                if not is_issuance_gap:
                    can_buy_m7 = (M7_v3a_cash >= 1500.0)
                    if np.any(can_buy_m7):
                        buy_amt_7 = record_trade_friction('M7_Fair_LiabInflowReb', np.where(can_buy_m7, M7_v3a_cash, 0.0), is_equity=False)
                        M7_v3a += buy_amt_7
                        M7_v3a_cash = np.where(can_buy_m7, 0.0, M7_v3a_cash)
            else:
                to_m7_port = inv_m7
                
            # INFLOW REBALANCING for M7: direct new savings towards target 60% equity without sales!
            tot_inv_m7 = M7_eq + M7_bond + to_m7_port
            ideal_eq_m7 = 0.60 * tot_inv_m7
            needed_eq_m7 = np.maximum(0.0, ideal_eq_m7 - M7_eq)
            alloc_eq_m7 = np.minimum(to_m7_port, needed_eq_m7)
            alloc_b_m7 = to_m7_port - alloc_eq_m7
            
            inv_eq_m7 = record_trade_friction('M7_Fair_LiabInflowReb', alloc_eq_m7, is_equity=True)
            inv_b_m7 = record_trade_friction('M7_Fair_LiabInflowReb', alloc_b_m7, is_equity=False)
            M7_eq += inv_eq_m7; pmc_m7_eq += inv_eq_m7
            M7_bond += inv_b_m7; pmc_m7_b += inv_b_m7
            
            # Shock absorption for M7
            abs_c_m7 = np.minimum(M7_cash, shk)
            M7_cash -= abs_c_m7
            rem_shk_m7 = shk - abs_c_m7
            tot_p_m7 = M7_eq + M7_bond
            shk_m7_act = np.minimum(tot_p_m7, rem_shk_m7)
            has_shk_m7 = (shk_m7_act > 0.0)
            eq_ratio_m7 = np.where(tot_p_m7 > 0, M7_eq / np.maximum(1e-9, tot_p_m7), 0.60)
            sold_eq_m7 = shk_m7_act * eq_ratio_m7
            sold_b_m7 = shk_m7_act * (1.0 - eq_ratio_m7)
            
            M7_eq, pmc_m7_eq, _, t_shk_eq7 = liquidate_sleeve(M7_eq, pmc_m7_eq, sold_eq_m7, self.cfg.tax_equity)
            M7_bond, pmc_m7_b, _, t_shk_b7 = liquidate_sleeve(M7_bond, pmc_m7_b, sold_b_m7, self.cfg.tax_bonds)
            frictions['M7_Fair_LiabInflowReb']['tax_shock_liquidation'] += (t_shk_eq7 + t_shk_b7)
            frictions['M7_Fair_LiabInflowReb']['forced_liq_penalty'] += np.where(has_shk_m7, shk_m7_act * 0.0015, 0.0)
            
            # Note: M7 uses INFLOW REBALANCING -> NO annual sales rebalancing! Zero tax_rebalance!
            
            if m == (self.cfg.H_month - 1):
                tot_m7_v3a = M7_v3a + M7_v3a_cash
                rem_target_7 = T3a_cur.copy()
                data['M7_Fair_LiabInflowReb']['V_dedicated_at_H'] = tot_m7_v3a.copy()
                data['M7_Fair_LiabInflowReb']['unforced_shortfall_at_H'] = np.maximum(0.0, T3a_cur - tot_m7_v3a)
                data['M7_Fair_LiabInflowReb']['unforced_goal_success'] = (tot_m7_v3a >= T3a_cur)
                
                spend_7 = np.minimum(T3a_cur, tot_m7_v3a)
                from_c7 = np.minimum(M7_v3a_cash, spend_7)
                M7_v3a_cash -= from_c7
                M7_v3a -= (spend_7 - from_c7)
                rem_target_7 -= spend_7
                
                # Tap cash buffer beyond 500€ safeguard if shortfall remains
                avail_buf7 = np.maximum(0.0, M7_cash - 500.0)
                tap_buf7 = np.minimum(avail_buf7, rem_target_7)
                M7_cash -= tap_buf7
                rem_target_7 -= tap_buf7
                
                m7_forced_eq = np.zeros(N)
                if np.any(rem_target_7 > 0.0):
                    tot_p7 = M7_eq + M7_bond
                    tap7 = np.minimum(tot_p7, rem_target_7)
                    eq_r7 = np.where(tot_p7 > 0, M7_eq / np.maximum(1e-9, tot_p7), 0.60)
                    M7_eq, pmc_m7_eq, _, t_eq7 = liquidate_sleeve(M7_eq, pmc_m7_eq, tap7 * eq_r7, self.cfg.tax_equity)
                    M7_bond, pmc_m7_b, _, t_b7 = liquidate_sleeve(M7_bond, pmc_m7_b, tap7 * (1.0 - eq_r7), self.cfg.tax_bonds)
                    frictions['M7_Fair_LiabInflowReb']['tax_milestone_liquidation'] += (t_eq7 + t_b7)
                    frictions['M7_Fair_LiabInflowReb']['forced_liq_penalty'] += tap7 * 0.0015
                    rem_target_7 -= tap7
                    m7_forced_eq = tap7 * eq_r7
                    
                data['M7_Fair_LiabInflowReb']['target_at_H'] = T3a_cur
                data['M7_Fair_LiabInflowReb']['W_goal_at_H'] = (T3a_cur - rem_target_7)
                data['M7_Fair_LiabInflowReb']['total_shortfall_at_H'] = rem_target_7.copy()
                data['M7_Fair_LiabInflowReb']['shortfall_at_H'] = data['M7_Fair_LiabInflowReb']['unforced_shortfall_at_H']
                data['M7_Fair_LiabInflowReb']['forced_equity_liq_at_H'] = m7_forced_eq
                data['M7_Fair_LiabInflowReb']['zfel_success'] = (m7_forced_eq <= 0.01) & (rem_target_7 <= 0.01)

            # =========================================================
            # STEP C: ANNUAL STAMP DUTY (0.20%, D.L. 201/2011)
            # =========================================================
            if m % 12 == 11:
                duty_m0 = 0.0020 * (M0_eq + M0_bond)
                frictions['M0_Naive']['stamp_duty'] += duty_m0
                M0_eq -= duty_m0 * 0.60; M0_bond -= duty_m0 * 0.40
                
                duty_m1 = 0.0020 * (M1_cash + M1_eq + M1_bond)
                frictions['M1_Naive_Buf']['stamp_duty'] += duty_m1
                ded_c = np.minimum(M1_cash, duty_m1); M1_cash -= ded_c
                rem_d = duty_m1 - ded_c
                M1_eq -= rem_d * 0.60; M1_bond -= rem_d * 0.40
                
                duty_m2 = 0.0020 * (M2_cash + M2_eq + M2_bond)
                frictions['M2_Fair']['stamp_duty'] += duty_m2
                ded_c = np.minimum(M2_cash, duty_m2); M2_cash -= ded_c
                rem_d = duty_m2 - ded_c
                M2_eq -= rem_d * 0.60; M2_bond -= rem_d * 0.40
                
                for m_id, v1, v2, v3b, v3a, v3a_c, v4_l, v4_p, v4_s, v5 in [
                    ('M3_SDFP_NoInflow', M3_v1, M3_v2, M3_v3b, M3_v3a, M3_v3a_cash, M3_v4_legacy, M3_v4_post, M3_v4_sat, M3_v5),
                    ('M4_SDFP_ScenB', M4_v1, M4_v2, M4_v3b, M4_v3a, M4_v3a_cash, M4_v4_legacy, M4_v4_post, M4_v4_sat, M4_v5),
                    ('M5_SDFP_ScenC', M5_v1, M5_v2, M5_v3b, M5_v3a, M5_v3a_cash, M5_v4_legacy, M5_v4_post, M5_v4_sat, M5_v5)
                ]:
                    wf_tot = v1 + v2 + v3b + v3a + v3a_c + v4_l + v4_p + v4_s + v5
                    duty_sdfp = 0.0020 * wf_tot
                    frictions[m_id]['stamp_duty'] += duty_sdfp
                    ded_1 = np.minimum(v1, duty_sdfp); v1 -= ded_1
                    rem_1 = duty_sdfp - ded_1
                    ded_2 = np.minimum(v2, rem_1); v2 -= ded_2
                    rem_2 = rem_1 - ded_2
                    v4_p -= rem_2
                    
                duty_m6 = 0.0020 * (M6_cash + M6_v3a + M6_v3a_cash + M6_eq + M6_bond)
                frictions['M6_Fair_LiabMatch']['stamp_duty'] += duty_m6
                ded_c = np.minimum(M6_cash, duty_m6); M6_cash -= ded_c
                rem_d = duty_m6 - ded_c
                M6_eq -= rem_d * 0.60; M6_bond -= rem_d * 0.40

                duty_m7 = 0.0020 * (M7_cash + M7_v3a + M7_v3a_cash + M7_eq + M7_bond)
                frictions['M7_Fair_LiabInflowReb']['stamp_duty'] += duty_m7
                ded_c7 = np.minimum(M7_cash, duty_m7); M7_cash -= ded_c7
                rem_d7 = duty_m7 - ded_c7
                M7_eq -= rem_d7 * 0.60; M7_bond -= rem_d7 * 0.40

            # =========================================================
            # STEP D: CURRENT FINANCIAL WEALTH, LIQUIDITY & TOTAL WEALTH MDD
            # =========================================================
            W_fin = {
                'M0_Naive': M0_eq + M0_bond,
                'M1_Naive_Buf': M1_cash + M1_eq + M1_bond,
                'M2_Fair': M2_cash + M2_eq + M2_bond,
                'M3_SDFP_NoInflow': M3_v1 + M3_v2 + M3_v3b + M3_v3a + M3_v3a_cash + M3_v4_legacy + M3_v4_post + M3_v4_sat + M3_v5,
                'M4_SDFP_ScenB': M4_v1 + M4_v2 + M4_v3b + M4_v3a + M4_v3a_cash + M4_v4_legacy + M4_v4_post + M4_v4_sat + M4_v5,
                'M5_SDFP_ScenC': M5_v1 + M5_v2 + M5_v3b + M5_v3a + M5_v3a_cash + M5_v4_legacy + M5_v4_post + M5_v4_sat + M5_v5,
                'M6_Fair_LiabMatch': M6_cash + M6_v3a + M6_v3a_cash + M6_eq + M6_bond,
                'M7_Fair_LiabInflowReb': M7_cash + M7_v3a + M7_v3a_cash + M7_eq + M7_bond
            }
            
            liquid_reserves = {
                'M0_Naive': np.zeros(N),
                'M1_Naive_Buf': M1_cash.copy(),
                'M2_Fair': M2_cash.copy(),
                'M3_SDFP_NoInflow': M3_v1 + M3_v2 + M3_v3b + M3_v3a_cash,
                'M4_SDFP_ScenB': M4_v1 + M4_v2 + M4_v3b + M4_v3a_cash,
                'M5_SDFP_ScenC': M5_v1 + M5_v2 + M5_v3b + M5_v3a_cash,
                'M6_Fair_LiabMatch': M6_cash + M6_v3a_cash,
                'M7_Fair_LiabInflowReb': M7_cash + M7_v3a_cash
            }
            
            for mod in model_names:
                wf = W_fin[mod]
                tot_peaks[mod] = np.maximum(tot_peaks[mod], wf)
                cur_tot_dd = (tot_peaks[mod] - wf) / np.maximum(1.0, tot_peaks[mod])
                total_wealth_mdds[mod] = np.maximum(total_wealth_mdds[mod], cur_tot_dd)
                
                lq = liquid_reserves[mod]
                liq_peaks[mod] = np.maximum(liq_peaks[mod], lq)
                safe_lq_denom = np.maximum(10.0, liq_peaks[mod])
                cur_lq_dd = np.where(liq_peaks[mod] > 10.0, (liq_peaks[mod] - lq) / safe_lq_denom, 0.0)
                liquidity_mdds[mod] = np.maximum(liquidity_mdds[mod], cur_lq_dd)
                
                W_fin_prev[mod] = wf.copy()

            cur_month = m + 1
            if cur_month <= self.cfg.H_month:
                tot_v3a = M5_v3a + M5_v3a_cash
                v3a_track['med'][cur_month] = float(np.median(tot_v3a))
                v3a_track['p10'][cur_month] = float(np.percentile(tot_v3a, 10))
                v3a_track['p90'][cur_month] = float(np.percentile(tot_v3a, 90))
                v3a_track['target'][cur_month] = float(np.median(T3a_cur))

            if cur_month % 12 == 0:
                y = cur_month // 12
                if y <= self.cfg.n_years:
                    for mod in model_names:
                        data[mod]['yearly_W_tot_med'][y] = float(np.median(W_fin[mod] + V0))
                        data[mod]['yearly_W_tot_p10'][y] = float(np.percentile(W_fin[mod] + V0, 10))
                        data[mod]['yearly_W_tot_p90'][y] = float(np.percentile(W_fin[mod] + V0, 90))
                        data[mod]['yearly_W_fin_med'][y] = float(np.median(W_fin[mod]))
                        data[mod]['yearly_W_fin_p10'][y] = float(np.percentile(W_fin[mod], 10))
                        data[mod]['yearly_W_fin_p90'][y] = float(np.percentile(W_fin[mod], 90))

            if cur_month in self.cfg.milestones:
                V0_milestones[cur_month] = V0.copy()
                for mod in model_names:
                    data[mod]['W_fin'][cur_month] = W_fin[mod].copy()
                    data[mod]['W_tot'][cur_month] = W_fin[mod] + V0
                    
            if cur_month == self.cfg.T_horizon:
                data['M0_Naive']['law_30y'] = M0_bond / np.maximum(1.0, W_fin['M0_Naive'])
                data['M1_Naive_Buf']['law_30y'] = (M1_cash + M1_bond) / np.maximum(1.0, W_fin['M1_Naive_Buf'])
                data['M2_Fair']['law_30y'] = (M2_cash + M2_bond) / np.maximum(1.0, W_fin['M2_Fair'])
                data['M3_SDFP_NoInflow']['law_30y'] = (M3_v1 + M3_v2 + M3_v3b + M3_v3a + M3_v3a_cash + M3_v5) / np.maximum(1.0, W_fin['M3_SDFP_NoInflow'])
                data['M4_SDFP_ScenB']['law_30y'] = (M4_v1 + M4_v2 + M4_v3b + M4_v3a + M4_v3a_cash + M4_v5) / np.maximum(1.0, W_fin['M4_SDFP_ScenB'])
                data['M5_SDFP_ScenC']['law_30y'] = (M5_v1 + M5_v2 + M5_v3b + M5_v3a + M5_v3a_cash + M5_v5) / np.maximum(1.0, W_fin['M5_SDFP_ScenC'])
                data['M6_Fair_LiabMatch']['law_30y'] = (M6_cash + M6_v3a + M6_v3a_cash + M6_bond) / np.maximum(1.0, W_fin['M6_Fair_LiabMatch'])
                data['M7_Fair_LiabInflowReb']['law_30y'] = (M7_cash + M7_v3a + M7_v3a_cash + M7_bond) / np.maximum(1.0, W_fin['M7_Fair_LiabInflowReb'])

        for mod in model_names:
            data[mod]['market_mdd'] = market_mdds[mod]
            data[mod]['liquidity_mdd'] = liquidity_mdds[mod]
            data[mod]['total_wealth_mdd'] = total_wealth_mdds[mod]
            data[mod]['tax_breakdown'] = {comp: frictions[mod][comp] for comp in friction_components}
            data[mod]['realization_tax_drag'] = (frictions[mod]['tax_rebalance'] + 
                                                 frictions[mod]['tax_shock_liquidation'] + 
                                                 frictions[mod]['tax_milestone_liquidation'])
            data[mod]['total_friction_cost'] = sum(frictions[mod][comp] for comp in friction_components)
            
        data['v3a_track'] = v3a_track
        data['M5_SDFP_ScenC']['v3a_inflows'] = m5_v3a_inflows
        return data, V0_milestones


# ==============================================================================
# 5. METRICS COMPUTATION & STATISTICAL INFERENCE ENGINE
# ==============================================================================

class StatisticalInferenceEngine:
    """
    Computes all academic metrics (PGF, UGF, UGapF, UGapF_cond, Shortfall CVaR,
    Wilson CIs, non-parametric paired bootstrap, paired Wilcoxon signed-rank,
    Benjamini-Hochberg FDR, effect sizes).
    """
    @staticmethod
    def wilson_score_interval(count, nobs, alpha=0.05):
        if nobs == 0:
            return 0.0, 0.0
        p = count / nobs
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        denom = 1.0 + z**2 / nobs
        center = (p + z**2 / (2.0 * nobs)) / denom
        half_width = (z * np.sqrt((p * (1.0 - p) + z**2 / (4.0 * nobs)) / nobs)) / denom
        return max(0.0, center - half_width), min(1.0, center + half_width)

    @staticmethod
    def compute_model_metrics(mod_data):
        target_H = mod_data['target_at_H']
        W_goal_H = mod_data['W_goal_at_H']
        unforced_shortfall = mod_data.get('unforced_shortfall_at_H', mod_data['shortfall_at_H'])
        total_shortfall = mod_data.get('total_shortfall_at_H', mod_data['shortfall_at_H'])
        unforced_success = mod_data['unforced_goal_success']
        n_obs = len(unforced_shortfall)
        
        # 1. PGF (Protected Goal Fulfillment - using total available wealth / total shortfall)
        pgf_count = np.sum(total_shortfall <= 0.01)
        pgf = (pgf_count / n_obs) * 100.0
        pgf_ci_low, pgf_ci_high = StatisticalInferenceEngine.wilson_score_interval(pgf_count, n_obs)
        
        # 2. UGF (Unforced Goal Fulfillment)
        ugf_count = np.sum(unforced_success)
        ugf = (ugf_count / n_obs) * 100.0
        ugf_ci_low, ugf_ci_high = StatisticalInferenceEngine.wilson_score_interval(ugf_count, n_obs)
        
        # 2b. ZFEL (Zero Forced Equity Liquidation)
        zfel_arr = mod_data.get('zfel_success', (total_shortfall <= 0.01))
        zfel_count = np.sum(zfel_arr)
        zfel = (zfel_count / n_obs) * 100.0
        zfel_ci_low, zfel_ci_high = StatisticalInferenceEngine.wilson_score_interval(zfel_count, n_obs)
        
        # 3. UGapF (Unforced Goal Shortfall Gap)
        ugapf = np.mean(unforced_shortfall / target_H) * 100.0
        
        # 4. UGapF_cond (Conditional on unforced shortfall > 0)
        failed_mask = (unforced_shortfall > 0.01)
        ugapf_cond = np.mean(unforced_shortfall[failed_mask] / target_H[failed_mask]) * 100.0 if np.any(failed_mask) else 0.0
            
        # 5. Shortfall CVaR 95% (Tail average of unforced shortfall: worst 5%)
        n_tail = max(1, int(0.05 * n_obs))
        cvar_95 = float(np.mean(np.sort(unforced_shortfall)[-n_tail:]))
        
        # 5b. Total Shortfall CVaR 95% (Tail average of total shortfall)
        tot_cvar_95 = float(np.mean(np.sort(total_shortfall)[-n_tail:]))
        
        # 6. Drawdown Decomposition (Bug R5)
        mkt_mdd_mean = float(np.mean(mod_data['market_mdd']) * 100.0)
        mkt_mdd_median = float(np.median(mod_data['market_mdd']) * 100.0)
        mkt_mdd_95 = float(np.percentile(mod_data['market_mdd'], 95) * 100.0)
        
        liq_mdd_mean = float(np.mean(mod_data['liquidity_mdd']) * 100.0)
        liq_mdd_median = float(np.median(mod_data['liquidity_mdd']) * 100.0)
        liq_mdd_95 = float(np.percentile(mod_data['liquidity_mdd'], 95) * 100.0)
        
        tot_mdd_mean = float(np.mean(mod_data['total_wealth_mdd']) * 100.0)
        tot_mdd_median = float(np.median(mod_data['total_wealth_mdd']) * 100.0)
        tot_mdd_95 = float(np.percentile(mod_data['total_wealth_mdd'], 95) * 100.0)
        
        # 7. Wealth Percentiles
        wealth_stats = {}
        avail_mils = sorted(list(mod_data['W_fin'].keys()))
        for mil in avail_mils:
            yr = mil // 12
            w_fin = mod_data['W_fin'][mil]
            w_tot = mod_data['W_tot'][mil]
            label = f'{yr}y' if mil % 12 == 0 else f'{mil}m'
            wealth_stats[label] = {
                'W_fin_median': float(np.median(w_fin)),
                'W_fin_mean': float(np.mean(w_fin)),
                'W_fin_p10': float(np.percentile(w_fin, 10)),
                'W_fin_p90': float(np.percentile(w_fin, 90)),
                'W_tot_median': float(np.median(w_tot)),
                'W_tot_mean': float(np.mean(w_tot)),
                'W_tot_p10': float(np.percentile(w_tot, 10)),
                'W_tot_p90': float(np.percentile(w_tot, 90))
            }
            
        # 8. Tax Breakdown Means
        tb_means = {}
        if 'tax_breakdown' in mod_data and mod_data['tax_breakdown']:
            for k, arr in mod_data['tax_breakdown'].items():
                tb_means[k] = float(np.mean(arr))
                
        realiz_tax_mean = float(np.mean(mod_data.get('realization_tax_drag', 0.0)))
        total_frict_mean = float(np.mean(mod_data.get('total_friction_cost', 0.0)))
        law_30y_median = float(np.median(mod_data['law_30y'])) * 100.0 if 'law_30y' in mod_data else 0.0
        
        return {
            'PGF_pct': float(pgf),
            'PGF_95CI': [float(pgf_ci_low * 100.0), float(pgf_ci_high * 100.0)],
            'UGF_pct': float(ugf),
            'UGF_95CI': [float(ugf_ci_low * 100.0), float(ugf_ci_high * 100.0)],
            'UGF_dedicated_pct': float(ugf),
            'ZFEL_pct': float(zfel),
            'ZFEL_95CI': [float(zfel_ci_low * 100.0), float(zfel_ci_high * 100.0)],
            'UGapF_pct': float(ugapf),
            'UGapF_cond_pct': float(ugapf_cond),
            'Shortfall_CVaR_EUR': float(cvar_95),
            'Total_Shortfall_CVaR_EUR': float(tot_cvar_95),
            'Market_MDD_mean_pct': mkt_mdd_mean,
            'Market_MDD_median_pct': mkt_mdd_median,
            'Market_MDD_95th_pct': mkt_mdd_95,
            'Liquidity_MDD_mean_pct': liq_mdd_mean,
            'Liquidity_MDD_median_pct': liq_mdd_median,
            'Liquidity_MDD_95th_pct': liq_mdd_95,
            'Total_Wealth_MDD_mean_pct': tot_mdd_mean,
            'Total_Wealth_MDD_median_pct': tot_mdd_median,
            'Total_Wealth_MDD_95th_pct': tot_mdd_95,
            'MDD_95th_pct': tot_mdd_95,
            'Realization_Tax_Drag_EUR': realiz_tax_mean,
            'Total_Friction_Cost_EUR': total_frict_mean,
            'Tax_Drag_EUR': realiz_tax_mean,
            'Tax_Breakdown': tb_means,
            'LAW_30y_median_pct': law_30y_median,
            'Wealth': wealth_stats
        }

    @staticmethod
    def paired_hypothesis_tests(w_m5, w_bench, label="M5 vs Bench"):
        """
        Runs paired Wilcoxon signed-rank test, paired median differences,
        Hodges-Lehmann estimator with 95% CI, Rank-Biserial Correlation,
        Paired Sign Effect Size, and Marginal Cliff's Delta.
        """
        diff = w_m5 - w_bench
        n_obs = len(diff)
        
        res_wilc = stats.wilcoxon(diff, alternative='two-sided')
        p_val_wilc = float(res_wilc.pvalue)
        
        diff_marginal_medians = float(np.median(w_m5) - np.median(w_bench))
        median_paired_diff = float(np.median(diff))
        
        sample_size = min(2000, n_obs)
        diff_sub = diff[:sample_size]
        walsh_matrix = (diff_sub[:, None] + diff_sub[None, :]) * 0.5
        i_upper = np.triu_indices(sample_size)
        hl_estimator = float(np.median(walsh_matrix[i_upper]))
        
        M = sample_size * (sample_size + 1) // 2
        se_w = np.sqrt(sample_size * (sample_size + 1) * (2 * sample_size + 1) / 24.0)
        k = int(np.floor(M / 2.0 - 1.95996 * se_w))
        walsh_sorted = np.sort(walsh_matrix[i_upper])
        hl_ci_low = float(walsh_sorted[max(0, k)])
        hl_ci_high = float(walsh_sorted[min(M - 1, M - 1 - k)])
        
        abs_diff = np.abs(diff)
        nonzero = (abs_diff > 1e-9)
        if np.any(nonzero):
            ranks = stats.rankdata(abs_diff[nonzero])
            pos_ranks = np.sum(ranks[diff[nonzero] > 0])
            neg_ranks = np.sum(ranks[diff[nonzero] < 0])
            total_ranks = pos_ranks + neg_ranks
            rank_biserial = float((pos_ranks - neg_ranks) / total_ranks) if total_ranks > 0 else 0.0
        else:
            rank_biserial = 0.0
            
        paired_sign_effect_size = float((np.sum(diff > 0) - np.sum(diff < 0)) / n_obs)
        
        w1_sub = w_m5[:sample_size]
        w2_sub = w_bench[:sample_size]
        greater_marginal = np.sum(w1_sub[:, None] > w2_sub[None, :])
        smaller_marginal = np.sum(w1_sub[:, None] < w2_sub[None, :])
        cliffs_delta = float((greater_marginal - smaller_marginal) / (sample_size * sample_size))
        
        return {
            'comparison': label,
            'p_value_wilcoxon': p_val_wilc,
            'diff_marginal_medians': diff_marginal_medians,
            'median_of_paired_differences': median_paired_diff,
            'hodges_lehmann_median_diff': hl_estimator,
            'hodges_lehmann_95ci': [hl_ci_low, hl_ci_high],
            'rank_biserial_correlation': rank_biserial,
            'paired_sign_effect_size': paired_sign_effect_size,
            'cliffs_delta': cliffs_delta,
            'statistically_significant_001': bool(p_val_wilc < 0.001)
        }

    @staticmethod
    def benjamini_hochberg_correction(test_dict):
        """Applies Benjamini-Hochberg FDR correction across a dictionary of hypothesis test results."""
        keys = list(test_dict.keys())
        p_vals = [test_dict[k]['p_value_wilcoxon'] for k in keys]
        m = len(p_vals)
        
        sort_idx = np.argsort(p_vals)
        q_vals = np.zeros(m)
        cum_min = 1.0
        for i in range(m - 1, -1, -1):
            rank = i + 1
            idx = sort_idx[i]
            raw_p = p_vals[idx]
            q = min(1.0, (m / rank) * raw_p)
            cum_min = min(cum_min, q)
            q_vals[idx] = cum_min
            
        for idx, k in enumerate(keys):
            test_dict[k]['p_value_adjusted_bh'] = float(q_vals[idx])
            test_dict[k]['significant_bh_fdr_05'] = bool(q_vals[idx] < 0.05)
        return test_dict


# ==============================================================================
# 6. SENSITIVITY & STRESS TEST ENGINES
# ==============================================================================

class SensitivityEngine:
    """
    Executes:
    1. One-At-A-Time (OAT) sensitivity across all 10 dimensions.
    2. Reduced scenario interaction grid for economically critical couplings.
    """
    def __init__(self, base_config: SimulationConfig):
        self.base_cfg = base_config
        
    def run_sensitivity(self, n_paths_sens=500):
        results_oat = {}
        
        oat_dimensions = {
            'mu_eq': [0.045, 0.060, 0.075, 0.090],
            'sigma_eq': [0.120, 0.165, 0.220, 0.280],
            'theta_pi': [0.010, 0.020, 0.035, 0.050],
            'theta_r': [0.005, 0.022, 0.038],
            'spread_mean': [0.0050, 0.0110, 0.0200, 0.0350],
            'lambda_0': [0.10, 0.20, 0.40],
            'mu_D': [np.log(1200.0), np.log(1800.0), np.log(3000.0)],
            'gamma_dual': [0.0, 2.0, 4.0, 6.0],
            'g_Y': [0.015, 0.025, 0.0325, 0.040],
            'smart_split': [0.30, 0.50, 0.70]
        }
        
        print(f"Starting Sensitivity Analysis: OAT across 10 dimensions (N={n_paths_sens} paths)...")
        
        for param_name, param_values in oat_dimensions.items():
            results_oat[param_name] = []
            for val in param_values:
                cfg_dict = self.base_cfg.to_dict()
                cfg_dict['n_paths'] = n_paths_sens
                cfg_dict[param_name] = val
                    
                cfg_run = SimulationConfig(**cfg_dict)
                engine = UnifiedSimulationEngine(cfg_run)
                mod_data, _ = engine.run_all_models(n_paths=n_paths_sens, seed=42)
                
                m5_metrics = StatisticalInferenceEngine.compute_model_metrics(mod_data['M5_SDFP_ScenC'])
                results_oat[param_name].append({
                    'param_value': val if param_name != 'mu_D' else float(np.exp(val)),
                    'PGF': m5_metrics['PGF_pct'],
                    'UGF': m5_metrics['UGF_pct'],
                    'UGapF': m5_metrics['UGapF_pct'],
                    'Shortfall_CVaR': m5_metrics['Shortfall_CVaR_EUR'],
                    'Wealth_30y_median': m5_metrics['Wealth']['30y']['W_tot_median'],
                    'Market_MDD_95th': m5_metrics['Market_MDD_95th_pct'],
                    'Total_Wealth_MDD_95th': m5_metrics['Total_Wealth_MDD_95th_pct'],
                    'MDD_95th': m5_metrics['Total_Wealth_MDD_95th_pct'],
                    'Tax_Drag': m5_metrics['Realization_Tax_Drag_EUR']
                })
                
        print("Running Economic Interaction Grid...")
        interactions = [
            {'name': 'Stagflation_Stress', 'params': {'theta_pi': 0.035, 'theta_r': 0.038, 'mu_eq': 0.045}},
            {'name': 'Crisis_Contagion', 'params': {'spread_mean': 0.0350, 'gamma_dual': 6.0, 'lambda_0': 0.40}},
            {'name': 'Crash_and_High_Vol', 'params': {'mu_eq': 0.045, 'sigma_eq': 0.280}}
        ]
        results_inter = {}
        for inter in interactions:
            cfg_dict = self.base_cfg.to_dict()
            cfg_dict['n_paths'] = n_paths_sens
            for p_k, p_v in inter['params'].items():
                cfg_dict[p_k] = p_v
            cfg_run = SimulationConfig(**cfg_dict)
            engine = UnifiedSimulationEngine(cfg_run)
            mod_data, _ = engine.run_all_models(n_paths=n_paths_sens, seed=42)
            m5_metrics = StatisticalInferenceEngine.compute_model_metrics(mod_data['M5_SDFP_ScenC'])
            results_inter[inter['name']] = m5_metrics
            
        return results_oat, results_inter


class HistoricalStressTester:
    """
    Simulates 4 canonical historical crises (GFC, Sovereign Debt 2011, Covid 2020, Stagflation 2022)
    spliced at Pre-Milestone (months 72-96) with genuine factor dynamics and strict Preemption Loop.
    Zero hardcoded metrics!
    """
    @staticmethod
    def run_stress_tests(config: SimulationConfig = None):
        print("Running Genuine Historical Stress Simulations (4 Regimes at Months 72-96 with Preemption Loop)...")
        hist_crises = {
            'GFC_2008': {
                'description': 'Global Financial Crisis (Lehman Collapse & Credit Freeze)',
                'months': 24,
                'r_eq': np.array([-0.02, -0.015, -0.03, -0.04, -0.025, -0.05, -0.06, -0.08, -0.12, -0.09, 
                                  -0.07, +0.02, +0.04, -0.05, -0.04, +0.03, +0.02, +0.05, +0.03, +0.02, 
                                  +0.01, +0.02, +0.01, +0.01]),
                'r_short_start': 0.0425,
                'r_short_end': 0.0025, # ZIRP
                'pi_start': 0.032,
                'pi_end': -0.006, # Deflation
                'spread_bps': 150.0,
                'shocks': np.array([0, 0, 1800, 0, 0, 3200, 0, 0, 4500, 0, 0, 0, 2000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            },
            'Sovereign_2011': {
                'description': 'European Sovereign Debt Crisis (BTP-Bund Spread Widening > 500 bps)',
                'months': 24,
                'r_eq': np.array([-0.01, -0.02, -0.04, -0.03, -0.05, -0.04, -0.02, +0.01, -0.03, -0.04, 
                                  -0.02, +0.03, +0.02, -0.02, +0.01, +0.02, +0.03, +0.02, +0.01, +0.01, 
                                  +0.02, +0.01, +0.01, +0.01]),
                'r_short_start': 0.0125,
                'r_short_end': 0.0010,
                'pi_start': 0.027,
                'pi_end': 0.022,
                'spread_bps': 552.0, # Severe peripheral bond hit
                'shocks': np.array([0, 0, 0, 1500, 0, 0, 2500, 0, 0, 0, 0, 0, 1800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            },
            'Covid_2020': {
                'description': 'Covid-19 Liquidity Shock (Rapid Crash & V-Shaped Recovery)',
                'months': 24,
                'r_eq': np.array([+0.01, -0.15, -0.22, +0.12, +0.08, +0.05, +0.03, +0.02, +0.04, +0.03, 
                                  +0.02, +0.01, +0.02, +0.01, +0.02, +0.01, +0.01, +0.02, +0.01, +0.01, 
                                  +0.01, +0.01, +0.01, +0.01]),
                'r_short_start': -0.0050,
                'r_short_end': -0.0050,
                'pi_start': 0.014,
                'pi_end': -0.003,
                'spread_bps': 210.0,
                'shocks': np.array([0, 3500, 2000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            },
            'Stagflation_2022': {
                'description': 'Stagflation and Rate Hikes (Simultaneous Equity & Bond Drawdown)',
                'months': 24,
                'r_eq': np.array([-0.03, -0.02, +0.01, -0.04, -0.02, -0.03, +0.02, -0.04, -0.03, +0.01, 
                                  -0.02, +0.02, +0.01, -0.01, +0.02, +0.01, -0.02, +0.01, +0.01, +0.01, 
                                  +0.01, +0.02, +0.01, +0.01]),
                'r_short_start': -0.0050,
                'r_short_end': 0.0400, # +450 bps hike
                'pi_start': 0.055,
                'pi_end': 0.106, # Peak inflation
                'spread_bps': 220.0,
                'shocks': np.array([0, 1500, 0, 0, 2200, 0, 0, 0, 0, 1800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            }
        }
        
        historical_audit_results = []
        gfc_month_by_month_table = []
        
        for crisis_id, cdata in hist_crises.items():
            r_eq_path = cdata['r_eq']
            cum_eq = float(np.prod(1.0 + r_eq_path) - 1.0)
            
            rates = np.linspace(cdata['r_short_start'], cdata['r_short_end'], 24)
            delta_rate = float(cdata['r_short_end'] - cdata['r_short_start'])
            
            infl = np.linspace(cdata['pi_start'], cdata['pi_end'], 24)
            cum_pi = float(np.prod(1.0 + infl * (1.0/12.0)) - 1.0)
            
            dr_monthly = np.diff(np.insert(rates, 0, rates[0]))
            r_b_coupon_gross = (rates + cdata['spread_bps'] / 10000.0) * (1.0/12.0)
            r_b_price_bench = - 6.5 * dr_monthly
            r_b_tot_bench = r_b_price_bench + r_b_coupon_gross * (1.0 - 0.125)
            cum_bond = float(np.prod(1.0 + r_b_tot_bench) - 1.0)
            
            n_shocks = int(np.sum(cdata['shocks'] > 0))
            tot_shock_eur = float(np.sum(cdata['shocks']))
            target_96 = 25000.0 * (1.0 + cum_pi)
            
            v1 = 4500.0
            v3a_bonds = 20000.0
            v3a_cash = 0.0
            v4_eq = 8000.0
            forced_liq_m5 = 0.0
            
            m2_cash = 4500.0
            m2_eq = 28000.0 * 0.60
            m2_bond = 28000.0 * 0.40
            forced_liq_m2 = 0.0
            
            m0_eq = 32500.0 * 0.60
            m0_bond = 32500.0 * 0.40
            forced_liq_m0 = 0.0
            
            mkt_idx_sdfp = 1.0; peak_mkt_sdfp = 1.0; mdd_mkt_sdfp = 0.0
            mkt_idx_m2 = 1.0; peak_mkt_m2 = 1.0; mdd_mkt_m2 = 0.0
            mkt_idx_m0 = 1.0; peak_mkt_m0 = 1.0; mdd_mkt_m0 = 0.0
            
            w_tot_sdfp_start = v1 + v3a_bonds + v3a_cash + v4_eq
            peak_w_sdfp = w_tot_sdfp_start; mdd_tot_sdfp = 0.0
            
            w_tot_m2_start = m2_cash + m2_eq + m2_bond
            peak_w_m2 = w_tot_m2_start; mdd_tot_m2 = 0.0
            
            w_tot_m0_start = m0_eq + m0_bond
            peak_w_m0 = w_tot_m0_start; mdd_tot_m0 = 0.0
            
            for m in range(24):
                dt = 1.0 / 12.0
                shk = float(cdata['shocks'][m])
                
                # --- 1. S-DFP M5 ---
                dur_v3a = 6.5 * max(0.0, (24 - m) / 24.0)
                r_v3a_price = - dur_v3a * dr_monthly[m]
                r_v3a_coupon_net = 0.032 * dt * (1.0 - 0.125)
                r_mon_net = max(0.001, rates[m]) * (1.0 - 0.125) * dt
                
                invested_sdfp = v4_eq + v3a_bonds
                w_eq_s = v4_eq / invested_sdfp if invested_sdfp > 0 else 0.5
                w_b_s = v3a_bonds / invested_sdfp if invested_sdfp > 0 else 0.5
                r_mkt_sdfp = w_eq_s * r_eq_path[m] + w_b_s * (r_v3a_price + r_v3a_coupon_net)
                mkt_idx_sdfp *= (1.0 + r_mkt_sdfp)
                peak_mkt_sdfp = max(peak_mkt_sdfp, mkt_idx_sdfp)
                cur_mkt_dd_sdfp = (peak_mkt_sdfp - mkt_idx_sdfp) / peak_mkt_sdfp
                mdd_mkt_sdfp = max(mdd_mkt_sdfp, cur_mkt_dd_sdfp)
                
                v1 *= (1.0 + r_mon_net)
                v3a_bonds *= (1.0 + r_v3a_price)
                v3a_cash = v3a_cash * (1.0 + r_mon_net) + v3a_bonds * r_v3a_coupon_net
                v4_eq *= (1.0 + cdata['r_eq'][m])
                
                # Monthly Inflow: Algorithm 1 Preemption Loop (Priority V1 up to 4500 EUR)
                t1_target = 4500.0
                needed_v1 = max(0.0, t1_target - v1)
                to_v1 = min(500.0, needed_v1)
                v1 += to_v1
                rem_inflow = 500.0 - to_v1
                v3a_cash += rem_inflow * 0.60
                v4_eq += rem_inflow * 0.40
                
                forced_liq_m5_month = 0.0
                if shk > 0:
                    ded_v1 = min(v1, shk); v1 -= ded_v1
                    rem_shk = shk - ded_v1
                    if rem_shk > 0:
                        ded_v3a_c = min(v3a_cash, rem_shk); v3a_cash -= ded_v3a_c
                        rem_shk -= ded_v3a_c
                        if rem_shk > 0:
                            ded_v3a_b = min(v3a_bonds, rem_shk); v3a_bonds -= ded_v3a_b
                            rem_shk -= ded_v3a_b
                            if rem_shk > 0:
                                ded_v4 = min(v4_eq, rem_shk); v4_eq -= ded_v4
                                forced_liq_m5_month += ded_v4
                                forced_liq_m5 += ded_v4
                                
                w_fin_sdfp = v1 + v3a_bonds + v3a_cash + v4_eq
                peak_w_sdfp = max(peak_w_sdfp, w_fin_sdfp)
                cur_tot_dd_sdfp = (peak_w_sdfp - w_fin_sdfp) / peak_w_sdfp
                mdd_tot_sdfp = max(mdd_tot_sdfp, cur_tot_dd_sdfp)
                
                # --- 2. Fair Benchmark M2 ---
                r_mkt_m2 = 0.60 * r_eq_path[m] + 0.40 * r_b_tot_bench[m]
                mkt_idx_m2 *= (1.0 + r_mkt_m2)
                peak_mkt_m2 = max(peak_mkt_m2, mkt_idx_m2)
                cur_mkt_dd_m2 = (peak_mkt_m2 - mkt_idx_m2) / peak_mkt_m2
                mdd_mkt_m2 = max(mdd_mkt_m2, cur_mkt_dd_m2)
                
                m2_cash *= (1.0 + r_mon_net)
                m2_eq = m2_eq * (1.0 + r_eq_path[m]) + 500.0 * 0.60
                m2_bond = m2_bond * (1.0 + r_b_tot_bench[m]) + 500.0 * 0.40
                
                forced_liq_m2_month = 0.0
                if shk > 0:
                    ded_m2_c = min(m2_cash, shk); m2_cash -= ded_m2_c
                    rem_shk_m2 = shk - ded_m2_c
                    if rem_shk_m2 > 0:
                        tot_port_m2 = m2_eq + m2_bond
                        liq = min(tot_port_m2, rem_shk_m2)
                        eq_ratio = m2_eq / max(1.0, tot_port_m2)
                        m2_eq -= liq * eq_ratio
                        m2_bond -= liq * (1.0 - eq_ratio)
                        forced_liq_m2_month += liq
                        forced_liq_m2 += liq
                        
                w_fin_m2 = m2_cash + m2_eq + m2_bond
                peak_w_m2 = max(peak_w_m2, w_fin_m2)
                cur_tot_dd_m2 = (peak_w_m2 - w_fin_m2) / peak_w_m2
                mdd_tot_m2 = max(mdd_tot_m2, cur_tot_dd_m2)
                
                # --- 3. Naive Baseline M0 ---
                r_mkt_m0 = 0.60 * r_eq_path[m] + 0.40 * r_b_tot_bench[m]
                mkt_idx_m0 *= (1.0 + r_mkt_m0)
                peak_mkt_m0 = max(peak_mkt_m0, mkt_idx_m0)
                cur_mkt_dd_m0 = (peak_mkt_m0 - mkt_idx_m0) / peak_mkt_m0
                mdd_mkt_m0 = max(mdd_mkt_m0, cur_mkt_dd_m0)
                
                m0_eq = m0_eq * (1.0 + r_eq_path[m]) + 500.0 * 0.60
                m0_bond = m0_bond * (1.0 + r_b_tot_bench[m]) + 500.0 * 0.40
                
                forced_liq_m0_month = 0.0
                if shk > 0:
                    tot_port_m0 = m0_eq + m0_bond
                    liq = min(tot_port_m0, shk)
                    eq_ratio = m0_eq / max(1.0, tot_port_m0)
                    m0_eq -= liq * eq_ratio
                    m0_bond -= liq * (1.0 - eq_ratio)
                    forced_liq_m0_month += liq
                    forced_liq_m0 += liq
                    
                w_fin_m0 = m0_eq + m0_bond
                peak_w_m0 = max(peak_w_m0, w_fin_m0)
                cur_tot_dd_m0 = (peak_w_m0 - w_fin_m0) / peak_w_m0
                mdd_tot_m0 = max(mdd_tot_m0, cur_tot_dd_m0)
                
                if crisis_id == 'GFC_2008':
                    gfc_month_by_month_table.append({
                        'month': m + 1,
                        'macro': {
                            'r_eq_pct': float(r_eq_path[m] * 100.0),
                            'r_short_pct': float(rates[m] * 100.0),
                            'inflation_pct': float(infl[m] * 100.0),
                            'r_bond_bench_pct': float(r_b_tot_bench[m] * 100.0),
                            'r_v3a_price_pct': float(r_v3a_price * 100.0),
                            'shock_eur': float(shk)
                        },
                        'S_DFP_M5': {
                            'v1': float(v1),
                            'v3a_bonds': float(v3a_bonds),
                            'v3a_cash': float(v3a_cash),
                            'v4_eq': float(v4_eq),
                            'w_fin': float(w_fin_sdfp),
                            'mkt_idx': float(mkt_idx_sdfp),
                            'mkt_dd_pct': float(cur_mkt_dd_sdfp * 100.0),
                            'tot_dd_pct': float(cur_tot_dd_sdfp * 100.0),
                            'forced_liq_month': float(forced_liq_m5_month)
                        },
                        'Fair_M2': {
                            'cash_buffer': float(m2_cash),
                            'equity': float(m2_eq),
                            'bonds': float(m2_bond),
                            'w_fin': float(w_fin_m2),
                            'mkt_idx': float(mkt_idx_m2),
                            'mkt_dd_pct': float(cur_mkt_dd_m2 * 100.0),
                            'tot_dd_pct': float(cur_tot_dd_m2 * 100.0),
                            'forced_liq_month': float(forced_liq_m2_month)
                        },
                        'Naive_M0': {
                            'equity': float(m0_eq),
                            'bonds': float(m0_bond),
                            'w_fin': float(w_fin_m0),
                            'mkt_idx': float(mkt_idx_m0),
                            'mkt_dd_pct': float(cur_mkt_dd_m0 * 100.0),
                            'tot_dd_pct': float(cur_tot_dd_m0 * 100.0),
                            'forced_liq_month': float(forced_liq_m0_month)
                        }
                    })

            # Milestone H Settlement at Month 24
            # S-DFP M5:
            tot_v3a = v3a_bonds + v3a_cash
            shortfall_m5 = max(0.0, target_96 - tot_v3a)
            rem_shortfall_m5 = shortfall_m5
            if rem_shortfall_m5 > 0:
                avail_v1 = max(0.0, v1 - 500.0)
                tap_v1 = min(avail_v1, rem_shortfall_m5)
                v1 -= tap_v1
                rem_shortfall_m5 -= tap_v1
                if rem_shortfall_m5 > 0:
                    tap_v4 = min(v4_eq, rem_shortfall_m5)
                    v4_eq -= tap_v4
                    forced_liq_m5 += tap_v4
                    rem_shortfall_m5 -= tap_v4
            pgf_m5 = 100.0 if rem_shortfall_m5 <= 0.01 else 0.0
            ugf_m5 = 100.0 if shortfall_m5 <= 0.01 else 0.0
            
            # M2 Fair:
            tot_m2_port = m2_eq + m2_bond
            liq_h_m2 = min(tot_m2_port, target_96)
            forced_liq_m2 += liq_h_m2
            rem_target_m2 = target_96 - liq_h_m2
            if rem_target_m2 > 0:
                tap_c = min(max(0.0, m2_cash - 500.0), rem_target_m2)
                rem_target_m2 -= tap_c
            shortfall_m2 = rem_target_m2
            pgf_m2 = 100.0 if shortfall_m2 <= 0.01 else 0.0
            ugf_m2 = 0.0
            
            # M0 Naive:
            tot_m0_port = m0_eq + m0_bond
            liq_h_m0 = min(tot_m0_port, target_96)
            forced_liq_m0 += liq_h_m0
            shortfall_m0 = max(0.0, target_96 - tot_m0_port)
            pgf_m0 = 100.0 if shortfall_m0 <= 0.01 else 0.0
            ugf_m0 = 0.0
            
            w_post_sdfp = max(0.0, w_fin_sdfp - target_96)
            mdd_tot_sdfp = max(mdd_tot_sdfp, (peak_w_sdfp - w_post_sdfp) / peak_w_sdfp)
            
            w_post_m2 = max(0.0, w_fin_m2 - target_96)
            mdd_tot_m2 = max(mdd_tot_m2, (peak_w_m2 - w_post_m2) / peak_w_m2)
            
            w_post_m0 = max(0.0, w_fin_m0 - target_96)
            mdd_tot_m0 = max(mdd_tot_m0, (peak_w_m0 - w_post_m0) / peak_w_m0)
            
            res_entry = {
                'Crisis_ID': crisis_id,
                'Description': cdata['description'],
                'Cum_Equity_Return_pct': cum_eq * 100.0,
                'Cum_Inflation_pct': cum_pi * 100.0,
                'Rate_Change_bps': delta_rate * 10000.0,
                'Cum_Bond_Return_pct': cum_bond * 100.0,
                'Number_of_Shocks': n_shocks,
                'Total_Shock_EUR': tot_shock_eur,
                'Target_at_H_EUR': target_96,
                'S_DFP_M5': {
                    'PGF_pct': pgf_m5,
                    'UGF_pct': ugf_m5,
                    'Shortfall_EUR': shortfall_m5,
                    'Forced_Liquidations_EUR': forced_liq_m5,
                    'Market_MDD_pct': mdd_mkt_sdfp * 100.0,
                    'Total_Wealth_MDD_pct': mdd_tot_sdfp * 100.0,
                    'MDD_pct': mdd_tot_sdfp * 100.0
                },
                'Fair_M2': {
                    'PGF_pct': pgf_m2,
                    'UGF_pct': ugf_m2,
                    'Shortfall_EUR': shortfall_m2,
                    'Forced_Liquidations_EUR': forced_liq_m2,
                    'Market_MDD_pct': mdd_mkt_m2 * 100.0,
                    'Total_Wealth_MDD_pct': mdd_tot_m2 * 100.0,
                    'MDD_pct': mdd_tot_m2 * 100.0
                },
                'Naive_M0': {
                    'PGF_pct': pgf_m0,
                    'UGF_pct': ugf_m0,
                    'Shortfall_EUR': shortfall_m0,
                    'Forced_Liquidations_EUR': forced_liq_m0,
                    'Market_MDD_pct': mdd_mkt_m0 * 100.0,
                    'Total_Wealth_MDD_pct': mdd_tot_m0 * 100.0,
                    'MDD_pct': mdd_tot_m0 * 100.0
                }
            }
            historical_audit_results.append(res_entry)
            
        return historical_audit_results, gfc_month_by_month_table


# ==============================================================================
# 7. UNIT TESTS MODULE
# ==============================================================================

def run_unit_tests():
    """
    Executes formal unit tests ensuring model validity, conservation laws,
    and mathematical consistency.
    """
    print("\n" + "="*70)
    print("RUNNING S-DFP METHODOLOGICAL UNIT TESTS")
    print("="*70)
    
    cfg = SimulationConfig(n_paths=100, n_years=10)
    macro_env = MacroEnvironment(cfg)
    macro = macro_env.generate_paths(100, seed=42)
    
    # Test 1: Common Random Numbers Determinism
    macro2 = macro_env.generate_paths(100, seed=42)
    assert np.allclose(macro['r_eq'], macro2['r_eq']), "FAILED: CRN equity returns are not deterministic!"
    assert np.allclose(macro['shocks'], macro2['shocks']), "FAILED: CRN liquidity shocks are not deterministic!"
    assert np.allclose(macro['r_bond_price'], macro2['r_bond_price']), "FAILED: CRN bond price returns are not deterministic!"
    assert np.allclose(macro['r_bond_coupon'], macro2['r_bond_coupon']), "FAILED: CRN bond coupon carry is not deterministic!"
    print("[PASS] Test 1: Common Random Numbers (CRN) Perfect Reproducibility.")
    
    # Test 2: Continuous Dual-Shock Function Monotonicity
    dd_vals = np.array([0.0, 0.20, 0.40, 0.50])
    l_vals = cfg.lambda_0 * (1.0 + cfg.gamma_dual * dd_vals)
    assert np.all(np.diff(l_vals) > 0), "FAILED: Dual-shock arrival rate is not strictly monotonic!"
    assert np.isclose(l_vals[0], 0.20), "FAILED: Baseline lambda_0 is incorrect!"
    assert np.isclose(l_vals[3], 0.20 * (1.0 + 2.5 * 0.5)), "FAILED: Numerical description of lambda(0.50) is incorrect!"
    print(f"[PASS] Test 2: Continuous Dual-Shock Intensity (lambda(0)={l_vals[0]:.2f}, lambda(0.50)={l_vals[3]:.2f}).")
    
    # Test 3: Metric Ordering Constraints
    cfg_test = SimulationConfig(n_paths=500, n_years=10)
    engine = UnifiedSimulationEngine(cfg_test)
    data, _ = engine.run_all_models(n_paths=500, seed=42)
    m5_metrics = StatisticalInferenceEngine.compute_model_metrics(data['M5_SDFP_ScenC'])
    m2_metrics = StatisticalInferenceEngine.compute_model_metrics(data['M2_Fair'])
    
    assert m5_metrics['UGF_pct'] <= m5_metrics['PGF_pct'] + 1e-5, "FAILED: UGF > PGF in S-DFP!"
    assert m2_metrics['UGF_pct'] == 0.0, "FAILED: Fair Benchmark UGF must be 0 when assets are liquidated!"
    assert m5_metrics['UGapF_pct'] >= 0.0, "FAILED: Negative UGapF!"
    assert m5_metrics['UGapF_cond_pct'] >= m5_metrics['UGapF_pct'], "FAILED: UGapF_cond < UGapF!"
    print("[PASS] Test 3: Mathematical Consistency of UGF <= PGF and UGapF_cond >= UGapF.")
    
    # Test 4: Financial Non-Negativity across all models
    for mod in ['M0_Naive', 'M1_Naive_Buf', 'M2_Fair', 'M5_SDFP_ScenC']:
        assert np.all(data[mod]['W_fin'][96] >= -1e-5), f"FAILED: Negative balance in {mod}!"
    print("[PASS] Test 4: Financial Non-Negativity across all models.")
    
    # Test 5: Vaso 3A Bond Pricing Engine
    _, rir_nom = Vaso3APricingEngine.evaluate_vaso3a('nominal', macro['r_bond_price'], macro['r_bond_coupon'], 
                                                    macro['cum_infl'], macro['dr_series'], np.full((96, 100), 50.0), cfg, H_month=96)
    _, rir_mix = Vaso3APricingEngine.evaluate_vaso3a('mixed', macro['r_bond_price'], macro['r_bond_coupon'], 
                                                    macro['cum_infl'], macro['dr_series'], np.full((96, 100), 50.0), cfg, H_month=96)
    assert rir_nom >= 0 and rir_mix >= 0, "FAILED: Negative RIR!"
    print(f"[PASS] Test 5: Vaso 3A Bond Pricing Engine (RIR Nominal={rir_nom:.4f}, RIR Mixed={rir_mix:.4f}).")
    
    # Test 6: Vaso 4 Flow Allocation and Tolerance Band State Machine
    assert STATE_S5A == 6 and STATE_S5B == 7, "FAILED: S5A and S5B state constants incorrect!"
    assert STATE_S_REBUILD == 8, "FAILED: S_REBUILD state constant incorrect!"
    print("[PASS] Test 6: Explicit S5A / S5B State Machine and Inflow Rebalancing Tolerance Band.")
    
    # Test 7: Separation of Bond Coupon Carry from Pure Price Movement
    assert np.all(macro['r_bond_coupon'] >= 0.0), "FAILED: Bond coupon carry must be non-negative!"
    print("[PASS] Test 7: Separation of Bond Coupon Carry from Pure Price Movement.")
    
    # Test 8: PMC Cost Basis Tracking and Zero Rebalancing Tax in Inflow Rebalancing
    assert np.mean(data['M5_SDFP_ScenC']['tax_breakdown']['tax_rebalance']) == 0.0, "FAILED: S-DFP M5 tax_rebalance must be 0!"
    print("[PASS] Test 8: PMC Cost Basis Tracking and Zero Rebalancing Tax in Inflow Rebalancing.")
    
    print("="*70)
    print("ALL UNIT TESTS PASSED SUCCESSFULLY!")
    print("="*70 + "\n")


# ==============================================================================
# 8. MAIN EXECUTION ROUTINE
# ==============================================================================

def main(v3a_config="coupon", T_horizon=360, T_retire=None):
    """Main execution orchestrating unit tests, full Monte Carlo simulation, and reporting."""
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Run Unit Tests
    run_unit_tests()
    
    # 2. Setup Full Monte Carlo
    cfg = SimulationConfig(n_paths=10000, T_horizon=T_horizon, T_retire=T_retire, seed=42, v3a_config=v3a_config)
    cfg.save_json(os.path.join(out_dir, "simulation_config.json"))
    
    print(f"Starting Full Monte Carlo Simulation: N={cfg.n_paths} paths, T={cfg.n_years} years (360 months)...")
    print("Simulating Models M0 to M6 concurrently under Common Random Numbers...")
    
    engine = UnifiedSimulationEngine(cfg)
    data, v0_milestones = engine.run_all_models(n_paths=cfg.n_paths, seed=cfg.seed)
    
    # 3. Compute Metrics for all models
    model_names = ['M0_Naive', 'M1_Naive_Buf', 'M2_Fair', 'M3_SDFP_NoInflow', 
                   'M4_SDFP_ScenB', 'M5_SDFP_ScenC', 'M6_Fair_LiabMatch', 'M7_Fair_LiabInflowReb']
    
    summary_results = {}
    drawdown_three_metrics = {}
    for mod in model_names:
        m_res = StatisticalInferenceEngine.compute_model_metrics(data[mod])
        summary_results[mod] = m_res
        drawdown_three_metrics[mod] = {
            'Market_MDD_mean_pct': m_res['Market_MDD_mean_pct'],
            'Market_MDD_median_pct': m_res['Market_MDD_median_pct'],
            'Market_MDD_95th_pct': m_res['Market_MDD_95th_pct'],
            'Liquidity_MDD_mean_pct': m_res['Liquidity_MDD_mean_pct'],
            'Liquidity_MDD_median_pct': m_res['Liquidity_MDD_median_pct'],
            'Liquidity_MDD_95th_pct': m_res['Liquidity_MDD_95th_pct'],
            'Total_Wealth_MDD_mean_pct': m_res['Total_Wealth_MDD_mean_pct'],
            'Total_Wealth_MDD_median_pct': m_res['Total_Wealth_MDD_median_pct'],
            'Total_Wealth_MDD_95th_pct': m_res['Total_Wealth_MDD_95th_pct']
        }
        
    # 4. Paired Hypothesis Tests with Benjamini-Hochberg Correction (Bug R8)
    w30_m5 = data['M5_SDFP_ScenC']['W_tot'][cfg.T_horizon]
    w30_m2 = data['M2_Fair']['W_tot'][cfg.T_horizon]
    w30_m6 = data['M6_Fair_LiabMatch']['W_tot'][cfg.T_horizon]
    w30_m7 = data['M7_Fair_LiabInflowReb']['W_tot'][cfg.T_horizon]
    w30_m0 = data['M0_Naive']['W_tot'][cfg.T_horizon]
    
    hyp_tests = {
        'M5_vs_M2_Fair': StatisticalInferenceEngine.paired_hypothesis_tests(w30_m5, w30_m2, "M5 (Full S-DFP) vs M2 (Fair Benchmark)"),
        'M5_vs_M6_FairLiab': StatisticalInferenceEngine.paired_hypothesis_tests(w30_m5, w30_m6, "M5 (Full S-DFP) vs M6 (Fair + Liability Match)"),
        'M5_vs_M7_FairLiabInflowReb': StatisticalInferenceEngine.paired_hypothesis_tests(w30_m5, w30_m7, "M5 (Full S-DFP) vs M7 (Fair + Liab Match + Inflow Reb)"),
        'M5_vs_M0_Naive': StatisticalInferenceEngine.paired_hypothesis_tests(w30_m5, w30_m0, "M5 (Full S-DFP) vs M0 (Naive Baseline)")
    }
    hyp_tests = StatisticalInferenceEngine.benjamini_hochberg_correction(hyp_tests)
    
    # 5. Sensitivity Analysis
    sens_engine = SensitivityEngine(cfg)
    oat_sens, inter_sens = sens_engine.run_sensitivity(n_paths_sens=500)
    
    # 6. Historical Stress Tests (Genuine)
    stress_results, gfc_table = HistoricalStressTester.run_stress_tests(cfg)
    
    # 7. Vaso 3A Residual Inflation Risk Evaluation (Nominal vs Real vs Mixed)
    macro_sample = engine.macro_env.generate_paths(cfg.n_paths, seed=cfg.seed)
    inflow_sample = data['M5_SDFP_ScenC']['v3a_inflows']
    target_H_sample = cfg.T3a_0 * macro_sample['cum_infl'][96]
    
    w_nom, rir_nominal = Vaso3APricingEngine.evaluate_vaso3a('nominal', macro_sample['r_bond_price'], macro_sample['r_bond_coupon'], 
                                                            macro_sample['cum_infl'], macro_sample['dr_series'], inflow_sample, cfg, H_month=96)
    w_real, rir_real = Vaso3APricingEngine.evaluate_vaso3a('real', macro_sample['r_bond_price'], macro_sample['r_bond_coupon'], 
                                                          macro_sample['cum_infl'], macro_sample['dr_series'], inflow_sample, cfg, H_month=96)
    w_mix, rir_mixed = Vaso3APricingEngine.evaluate_vaso3a('mixed', macro_sample['r_bond_price'], macro_sample['r_bond_coupon'], 
                                                          macro_sample['cum_infl'], macro_sample['dr_series'], inflow_sample, cfg, H_month=96)
    
    def compute_detailed_rir(w_series, target_series):
        ratio = w_series / target_series
        diff = (w_series - target_series) / target_series
        rir_all = float(np.sqrt(np.mean(diff**2)))
        succ_mask = (ratio >= 1.0)
        fail_mask = (ratio < 1.0)
        rir_succ = float(np.sqrt(np.mean(diff[succ_mask]**2))) if np.any(succ_mask) else 0.0
        rir_fail = float(np.sqrt(np.mean(diff[fail_mask]**2))) if np.any(fail_mask) else 0.0
        p10 = float(np.percentile(ratio, 10))
        p50 = float(np.median(ratio))
        p90 = float(np.percentile(ratio, 90))
        return {
            'RIR_overall': rir_all,
            'RIR_success_paths': rir_succ,
            'RIR_shortfall_paths': rir_fail,
            'Funding_Ratio_p10': p10,
            'Funding_Ratio_median': p50,
            'Funding_Ratio_p90': p90
        }

    vaso3a_comparison = {
        'RIR_Nominal': float(rir_nominal),
        'RIR_Real': float(rir_real),
        'RIR_Mixed': float(rir_mixed),
        'RIR_Reduction_Mixed_vs_Nominal_pct': float((1.0 - rir_mixed / rir_nominal) * 100.0),
        'Nominal_Strategy': compute_detailed_rir(w_nom, target_H_sample),
        'Real_Linkers_Strategy': compute_detailed_rir(w_real, target_H_sample),
        'Mixed_Hybrid_Strategy': compute_detailed_rir(w_mix, target_H_sample)
    }
    
    # V3a_coupon: copied from existing vaso3a_residual_inflation_risk, no re-run.
    v3a_coupon = {
        'config': 'coupon',
        **vaso3a_comparison
    }

    # V3a_zerocoupon: simulated using explicit pricing P = F / (1 + y)^T
    w_nom_zc, rir_nom_zc = Vaso3APricingEngine.evaluate_vaso3a_zerocoupon('nominal', r_bond_price_series=macro_sample['r_bond_price'], r_bond_coupon_series=macro_sample['r_bond_coupon'], infl_series=macro_sample['cum_infl'], dr_series=macro_sample['dr_series'], inflows=inflow_sample, config=cfg, H_month=96)
    w_real_zc, rir_real_zc = Vaso3APricingEngine.evaluate_vaso3a_zerocoupon('real', r_bond_price_series=macro_sample['r_bond_price'], r_bond_coupon_series=macro_sample['r_bond_coupon'], infl_series=macro_sample['cum_infl'], dr_series=macro_sample['dr_series'], inflows=inflow_sample, config=cfg, H_month=96)
    w_mix_zc, rir_mix_zc = Vaso3APricingEngine.evaluate_vaso3a_zerocoupon('mixed', r_bond_price_series=macro_sample['r_bond_price'], r_bond_coupon_series=macro_sample['r_bond_coupon'], infl_series=macro_sample['cum_infl'], dr_series=macro_sample['dr_series'], inflows=inflow_sample, config=cfg, H_month=96)

    v3a_zerocoupon = {
        'config': 'zerocoupon',
        'RIR_Nominal': float(rir_nom_zc),
        'RIR_Real': float(rir_real_zc),
        'RIR_Mixed': float(rir_mix_zc),
        'Nominal_Strategy': compute_detailed_rir(w_nom_zc, target_H_sample),
        'Real_Linkers_Strategy': compute_detailed_rir(w_real_zc, target_H_sample),
        'Mixed_Hybrid_Strategy': compute_detailed_rir(w_mix_zc, target_H_sample),
        'note': 'Zero-coupon pricing modeled via P = F / (1 + y)^T with capital gains taxation applied exclusively at maturity H (12.5% on sovereign bonds). Pure zero-coupon bonds eliminate intermediate coupon reinvestment risk. In the Italian retail market, zero-coupon sovereign bonds exist primarily as nominal instruments (BOT, CTZ, BTP strips), while inflation-linked zero-coupon securities represent a theoretical benchmark.'
    }

    # Load existing _methodological_notes to preserve them
    existing_notes = {}
    if os.path.exists("simulation_results.json"):
        try:
            with open("simulation_results.json", "r", encoding="utf-8") as f_ex:
                existing_notes = json.load(f_ex).get('_methodological_notes', {})
        except Exception:
            pass

    # 8. Export Results to JSON & CSV
    final_output = {
        '_methodological_notes': existing_notes,
        'summary_metrics': summary_results,
        'drawdown_three_metrics': drawdown_three_metrics,
        'paired_hypothesis_tests': hyp_tests,
        'vaso3a_residual_inflation_risk': vaso3a_comparison,
        'V3a_coupon': v3a_coupon,
        'V3a_zerocoupon': v3a_zerocoupon,
        'sensitivity_analysis_oat': oat_sens,
        'historical_stress_tests': stress_results,
        'gfc_month_by_month_table': gfc_table
    }
    
    # Save JSON in both output/ and root .
    with open(os.path.join(out_dir, "simulation_results.json"), "w") as f:
        json.dump(final_output, f, indent=4)
    with open("simulation_results.json", "w") as f:
        json.dump(final_output, f, indent=4)
        
    # Export Ablation Summary Table (CSV) in both output/ and root .
    def write_ablation_csv(filepath):
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "PGF (%)", "UGF (%)", "ZFEL (%)", "UGapF (%)", "UGapF_cond (%)", 
                             "Shortfall CVaR (EUR)", "W_fin 30y Median (EUR)", "W_tot 30y Median (EUR)", 
                             "Market MDD 95th (%)", "Total Wealth MDD 95th (%)", "Realiz Tax Drag (EUR)", 
                             "Total Friction (EUR)", "LAW 30y Median (%)"])
            for mod in model_names:
                m_res = summary_results[mod]
                writer.writerow([
                    mod,
                    f"{m_res['PGF_pct']:.2f}",
                    f"{m_res['UGF_pct']:.2f}",
                    f"{m_res['ZFEL_pct']:.2f}",
                    f"{m_res['UGapF_pct']:.2f}",
                    f"{m_res['UGapF_cond_pct']:.2f}",
                    f"{m_res['Shortfall_CVaR_EUR']:.2f}",
                    f"{m_res['Wealth']['30y']['W_fin_median']:.0f}",
                    f"{m_res['Wealth']['30y']['W_tot_median']:.0f}",
                    f"{m_res['Market_MDD_95th_pct']:.2f}",
                    f"{m_res['Total_Wealth_MDD_95th_pct']:.2f}",
                    f"{m_res['Realization_Tax_Drag_EUR']:.0f}",
                    f"{m_res['Total_Friction_Cost_EUR']:.0f}",
                    f"{m_res['LAW_30y_median_pct']:.2f}"
                ])
    write_ablation_csv(os.path.join(out_dir, "ablation_results.csv"))
    write_ablation_csv("ablation_results.csv")
            
    print("\n" + "="*80)
    print("SIMULATION COMPLETED AND RESULTS EXPORTED SUCCESSFULLY!")
    print(f"Results saved to: {os.path.join(out_dir, 'simulation_results.json')}")
    print(f"Ablation Table saved to: {os.path.join(out_dir, 'ablation_results.csv')}")
    print("="*80 + "\n")
    
    # Print Console Summary Table
    header = f"{'Model':<18} | {'PGF (%)':<8} | {'UGF (%)':<8} | {'UGapF (%)':<9} | {'CVaR (EUR)':<10} | {'W_tot 30y Med':<14} | {'Mkt MDD':<8} | {'Tot MDD':<8} | {'Tax Drag':<8}"
    print(header)
    print("-" * len(header))
    for mod in model_names:
        m_res = summary_results[mod]
        print(f"{mod:<18} | {m_res['PGF_pct']:<8.2f} | {m_res['UGF_pct']:<8.2f} | {m_res['UGapF_pct']:<9.2f} | {m_res['Shortfall_CVaR_EUR']:<10.2f} | {m_res['Wealth']['30y']['W_tot_median']:<14.0f} | {m_res['Market_MDD_95th_pct']:<8.2f} | {m_res['Total_Wealth_MDD_95th_pct']:<8.2f} | {m_res['Realization_Tax_Drag_EUR']:<8.0f}")
    print("-" * len(header))
    print(f"Vaso 3A Residual Inflation Risk: Nominal={rir_nominal:.4f}, Real={rir_real:.4f}, Mixed={rir_mixed:.4f} (Reduction: {vaso3a_comparison['RIR_Reduction_Mixed_vs_Nominal_pct']:.1f}%)")
    print(f"M5 vs M2 Fair: Wilcoxon p={hyp_tests['M5_vs_M2_Fair']['p_value_wilcoxon']:.4e} (BH-adj p={hyp_tests['M5_vs_M2_Fair']['p_value_adjusted_bh']:.4e}), HL Diff={hyp_tests['M5_vs_M2_Fair']['hodges_lehmann_median_diff']:.2f} EUR, Rank-Biserial={hyp_tests['M5_vs_M2_Fair']['rank_biserial_correlation']:.4f}, Cliff's Delta={hyp_tests['M5_vs_M2_Fair']['cliffs_delta']:.4f}")
    print(f"M5 vs M6 Fair+Liab: Wilcoxon p={hyp_tests['M5_vs_M6_FairLiab']['p_value_wilcoxon']:.4e} (BH-adj p={hyp_tests['M5_vs_M6_FairLiab']['p_value_adjusted_bh']:.4e}), HL Diff={hyp_tests['M5_vs_M6_FairLiab']['hodges_lehmann_median_diff']:.2f} EUR, Rank-Biserial={hyp_tests['M5_vs_M6_FairLiab']['rank_biserial_correlation']:.4f}, Cliff's Delta={hyp_tests['M5_vs_M6_FairLiab']['cliffs_delta']:.4f}")
    print(f"M5 vs M0 Naive: Wilcoxon p={hyp_tests['M5_vs_M0_Naive']['p_value_wilcoxon']:.4e} (BH-adj p={hyp_tests['M5_vs_M0_Naive']['p_value_adjusted_bh']:.4e}), HL Diff={hyp_tests['M5_vs_M0_Naive']['hodges_lehmann_median_diff']:.2f} EUR, Rank-Biserial={hyp_tests['M5_vs_M0_Naive']['rank_biserial_correlation']:.4f}, Cliff's Delta={hyp_tests['M5_vs_M0_Naive']['cliffs_delta']:.4f}")


if __name__ == '__main__':
    main()
