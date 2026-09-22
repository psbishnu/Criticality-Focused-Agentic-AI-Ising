#!/usr/bin/env python3
"""
Fair comparison of the proposed.

Default lattice sizes:
    L = 32, 64, 128

PRIMARY FAIRNESS PRINCIPLE
--------------------------
For each finite lattice size L, the primary target is the FULL-GRID
pseudocritical temperature Tc_full(L) extracted from the same dataset.
This avoids unfairly penalizing finite-size methods against the
thermodynamic-limit exact Tc.

Exact Tc = 2.269185... is reported only as a secondary physical reference.

Compared methods
----------------
1. Full-grid exhaustive reference
2. Uniform-grid sampling
3. Random sampling
4. Conventional adaptive peak refinement
5. Bayesian optimization: GP-UCB
6. Bayesian optimization: Expected Improvement (EI)
7. Active uncertainty sampling (GP variance)
8. Proposed Physics-Guided Agentic AI

Outputs
-------
Results/Comparative_PRE/
    comparison_all_results.csv
    comparison_final_budget.csv
    comparison_runtime.csv
    01_error_vs_budget.pdf
    02_query_reduction_vs_error.pdf
    03_final_method_comparison.pdf
    04_runtime_comparison.pdf
"""

from pathlib import Path

DATASETS = {
    32: Path("../JOB5_Noise/J5Data/MCD32.csv"),
    64: Path("../JOB5_Noise/J5Data/MCD64.csv"),
    128: Path("../JOB5_Noise/J5Data/MCD128.csv"),
}
LATTICE_SIZES = [32, 64, 128]

TEMPERATURE_COLUMN = "Temperature"
SPIN_PREFIX = "spin_"
J = 1.0
K_B = 1.0
EXTERNAL_FIELD = 0.0

QUERY_BUDGETS = [6, 8, 10, 12, 16, 20]
RANDOM_REPEATS = 30
RANDOM_SEED = 42

GP_LENGTH_SCALE = 0.20
GP_NOISE_LEVEL = 1e-6
GP_UCB_KAPPA = 2.0
EI_XI = 0.01
N_INITIAL_GP = 4

N_INITIAL_AGENT = 6
MIN_AGENT_QUERIES_BEFORE_STOP = 12
AGENT_STABILITY_WINDOW = 5
AGENT_TC_STABILITY_TOL = 0.015

EXPLORATION_WEIGHT = 0.40
CRITICALITY_WEIGHT = 0.45
BRACKETING_WEIGHT = 0.15
CRITICALITY_WIDTH = 0.20
BRACKET_WIDTH = 0.18

REFINE_INITIAL_POINTS = 6

RESULTS_ROOT = Path("Results")
OUTPUT_DIR = RESULTS_ROOT / "Comparative_PRE"

FIG_WIDTH = 11
FIG_HEIGHT = 8
FONT_SIZE = 17
AXIS_FONT_SIZE = 19
TITLE_FONT_SIZE = 21
LEGEND_FONT_SIZE = 12
LINE_WIDTH = 3.0
MARKER_SIZE = 7

EXACT_TC = 2.0 / __import__("math").log(1.0 + __import__("math").sqrt(2.0))

import argparse
import gc
import time
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "font.weight": "bold",
    "axes.labelsize": AXIS_FONT_SIZE,
    "axes.labelweight": "bold",
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.titleweight": "bold",
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
})

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", nargs="*", type=int, default=QUERY_BUDGETS)
    return p.parse_args()

def save_pdf(fig, outbase):
    fig.tight_layout()
    fig.savefig(str(outbase)+".pdf", bbox_inches="tight")
    plt.close(fig)

def spin_sort_key(name):
    return int(name.split("_")[-1])

def normalize(v):
    v=np.asarray(v,float)
    finite=np.isfinite(v)
    if not np.any(finite):
        return np.zeros_like(v)
    vv=v.copy()
    vv[~finite]=np.nanmin(vv[finite])
    lo,hi=np.min(vv),np.max(vv)
    if hi-lo<1e-14:
        return np.zeros_like(vv)
    return (vv-lo)/(hi-lo)

def parabolic_peak_tc(T,y):
    T=np.asarray(T,float); y=np.asarray(y,float)
    ok=np.isfinite(T)&np.isfinite(y)
    T,y=T[ok],y[ok]
    order=np.argsort(T); T,y=T[order],y[order]
    i=int(np.argmax(y))
    if 0<i<len(T)-1:
        try:
            a,b,_=np.polyfit(T[i-1:i+2],y[i-1:i+2],2)
            if a<0:
                xv=-b/(2*a)
                if T[i-1]<=xv<=T[i+1]:
                    return float(xv)
        except Exception:
            pass
    return float(T[i])

def load_aggregate(path,L):
    if not path.exists():
        raise FileNotFoundError(path)
    header=pd.read_csv(path,nrows=0).columns.tolist()
    spin_cols=sorted([c for c in header if c.startswith(SPIN_PREFIX)], key=spin_sort_key)
    if len(spin_cols)!=L*L:
        raise ValueError(f"L={L}: expected {L*L} spin columns, found {len(spin_cols)}")
    dtype_map={c:np.int8 for c in spin_cols}
    dtype_map[TEMPERATURE_COLUMN]=np.float64
    df=pd.read_csv(path,usecols=[TEMPERATURE_COLUMN]+spin_cols,dtype=dtype_map)
    T=df[TEMPERATURE_COLUMN].to_numpy(float)
    spins=df[spin_cols].to_numpy(np.int8,copy=False)
    n=L*L
    m=spins.mean(axis=1)
    am=np.abs(m)
    raw=pd.DataFrame({"T":T,"m":m,"am":am})
    rows=[]
    for temp,g in raw.groupby("T",sort=True):
        mv=g["m"].to_numpy(float)
        amv=g["am"].to_numpy(float)
        m2=np.mean(mv**2)
        chi=n/(K_B*temp)*max(m2-np.mean(amv)**2,0.0)
        rows.append({"Temperature":float(temp),"N_configs":int(len(g)),"Susceptibility":float(chi)})
    del df,spins,raw
    gc.collect()
    return pd.DataFrame(rows).sort_values("Temperature").reset_index(drop=True)

def gp_fit(Tsel,ysel):
    scaler=StandardScaler()
    ys=scaler.fit_transform(np.asarray(ysel).reshape(-1,1)).ravel()
    kernel=(ConstantKernel(1.0,(1e-3,1e3))*RBF(GP_LENGTH_SCALE,(1e-3,10.0))
            +WhiteKernel(GP_NOISE_LEVEL,(1e-10,1e-1)))
    gp=GaussianProcessRegressor(kernel=kernel,normalize_y=False,n_restarts_optimizer=2,random_state=RANDOM_SEED)
    gp.fit(np.asarray(Tsel).reshape(-1,1),ys)
    return gp

def initial_even_indices(n,n0):
    return list(dict.fromkeys(np.linspace(0,n-1,min(n0,n),dtype=int).tolist()))

def estimate_from_indices(agg,idx):
    s=agg.iloc[sorted(set(idx))]
    return parabolic_peak_tc(s["Temperature"],s["Susceptibility"])

def method_uniform(agg,budget):
    n=len(agg)
    idx=np.unique(np.linspace(0,n-1,min(budget,n),dtype=int))
    return estimate_from_indices(agg,idx),len(idx)

def method_random(agg,budget,seed):
    rng=np.random.default_rng(seed)
    n=len(agg)
    idx=np.sort(rng.choice(n,size=min(budget,n),replace=False))
    return estimate_from_indices(agg,idx),len(idx)

def method_peak_refinement(agg,budget):
    n=len(agg); budget=min(budget,n)
    selected=initial_even_indices(n,min(REFINE_INITIAL_POINTS,budget))
    while len(selected)<budget:
        peak_tc=estimate_from_indices(agg,selected)
        unselected=[i for i in range(n) if i not in selected]
        idx=min(unselected,key=lambda i:abs(float(agg.iloc[i]["Temperature"])-peak_tc))
        selected.append(idx)
    return estimate_from_indices(agg,selected),len(set(selected))

def method_gp(agg,budget,mode):
    n=len(agg); budget=min(budget,n)
    selected=initial_even_indices(n,min(N_INITIAL_GP,budget))
    while len(selected)<budget:
        sel=agg.iloc[selected]
        gp=gp_fit(sel["Temperature"].to_numpy(float),sel["Susceptibility"].to_numpy(float))
        Xall=agg["Temperature"].to_numpy(float).reshape(-1,1)
        mu,std=gp.predict(Xall,return_std=True)
        if mode=="ucb":
            acq=mu+GP_UCB_KAPPA*std
        elif mode=="ei":
            best=np.max(mu[np.array(selected,dtype=int)])
            improvement=mu-best-EI_XI
            z=np.divide(improvement,std,out=np.zeros_like(improvement),where=std>1e-12)
            acq=improvement*norm.cdf(z)+std*norm.pdf(z)
        elif mode=="uncertainty":
            acq=std.copy()
        else:
            raise ValueError(mode)
        acq[np.array(selected,dtype=int)]=-np.inf
        selected.append(int(np.argmax(acq)))
    return estimate_from_indices(agg,selected),len(set(selected))

def method_proposed_agent(agg,budget):
    n=len(agg); budget=min(budget,n)
    selected=initial_even_indices(n,min(N_INITIAL_AGENT,budget))
    tc_history=[estimate_from_indices(agg,selected)]

    def current_tc():
        return estimate_from_indices(agg,selected)

    while len(set(selected))<budget:
        if len(set(selected))>=min(MIN_AGENT_QUERIES_BEFORE_STOP,budget) and len(tc_history)>=AGENT_STABILITY_WINDOW:
            recent=np.asarray(tc_history[-AGENT_STABILITY_WINDOW:],float)
            if np.all(np.isfinite(recent)) and np.max(recent)-np.min(recent)<=AGENT_TC_STABILITY_TOL:
                tc_now=recent[-1]
                selected_t=agg.iloc[selected]["Temperature"].to_numpy(float)
                has_left=np.any((selected_t<tc_now)&(selected_t>tc_now-BRACKET_WIDTH))
                has_right=np.any((selected_t>tc_now)&(selected_t<tc_now+BRACKET_WIDTH))
                if has_left and has_right:
                    break

        sel=agg.iloc[selected]
        gp=gp_fit(sel["Temperature"].to_numpy(float),sel["Susceptibility"].to_numpy(float))
        temps=agg["Temperature"].to_numpy(float)
        mu,std=gp.predict(temps.reshape(-1,1),return_std=True)
        predicted_tc=float(temps[int(np.argmax(mu))])
        criticality=np.exp(-0.5*((temps-predicted_tc)/max(CRITICALITY_WIDTH,1e-8))**2)
        selected_t=temps[np.array(selected,dtype=int)]
        left_near=np.any((selected_t<predicted_tc)&(selected_t>predicted_tc-BRACKET_WIDTH))
        right_near=np.any((selected_t>predicted_tc)&(selected_t<predicted_tc+BRACKET_WIDTH))
        bracket=np.zeros_like(temps)
        if not left_near:
            bracket+=np.exp(-0.5*((temps-(predicted_tc-BRACKET_WIDTH/2))/max(BRACKET_WIDTH/2,1e-8))**2)
        if not right_near:
            bracket+=np.exp(-0.5*((temps-(predicted_tc+BRACKET_WIDTH/2))/max(BRACKET_WIDTH/2,1e-8))**2)
        acq=(EXPLORATION_WEIGHT*normalize(std)+CRITICALITY_WEIGHT*normalize(criticality)+BRACKETING_WEIGHT*normalize(bracket))
        acq[np.array(selected,dtype=int)]=-np.inf
        selected.append(int(np.argmax(acq)))
        tc_history.append(current_tc())
    return current_tc(),len(set(selected))

def summarize_stochastic(values,ref_tc):
    values=np.asarray(values,float)
    errors=np.abs(values-ref_tc)
    return {
        "Tc_mean":float(np.mean(values)),
        "Tc_std":float(np.std(values,ddof=1)) if len(values)>1 else 0.0,
        "Abs_Error_to_FullGrid_mean":float(np.mean(errors)),
        "Abs_Error_to_FullGrid_std":float(np.std(errors,ddof=1)) if len(errors)>1 else 0.0,
    }

def main():
    args=parse_args()
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    all_rows=[]; runtime_rows=[]

    for L in LATTICE_SIZES:
        agg=load_aggregate(DATASETS[L],L)
        nT=len(agg)
        tc_full=parabolic_peak_tc(agg["Temperature"],agg["Susceptibility"])
        valid_budgets=sorted(set(min(int(b),nT) for b in args.budgets if int(b)>=4))

        for budget in valid_budgets:
            for label,runner in [
                ("Uniform Grid",lambda:method_uniform(agg,budget)),
                ("Adaptive Peak Refinement",lambda:method_peak_refinement(agg,budget)),
                ("BO-GP-UCB",lambda:method_gp(agg,budget,"ucb")),
                ("BO-GP-EI",lambda:method_gp(agg,budget,"ei")),
                ("Active GP-Uncertainty",lambda:method_gp(agg,budget,"uncertainty")),
                ("Proposed Physics-Guided Agent",lambda:method_proposed_agent(agg,budget)),
            ]:
                t0=time.perf_counter(); tc,q=runner(); elapsed=time.perf_counter()-t0
                all_rows.append({
                    "L":L,"Method":label,"Budget":budget,"Queries_Used":q,
                    "Tc_mean":tc,"Tc_std":0.0,"FullGrid_Tc":tc_full,
                    "Abs_Error_to_FullGrid_mean":abs(tc-tc_full),
                    "Abs_Error_to_FullGrid_std":0.0,
                    "Abs_Error_to_Exact_secondary":abs(tc-EXACT_TC),
                    "Query_Reduction_Percent":(1-q/nT)*100.0,
                })
                runtime_rows.append({"L":L,"Method":label,"Budget":budget,"Seconds":elapsed})

            t0=time.perf_counter()
            vals=[method_random(agg,budget,RANDOM_SEED+r)[0] for r in range(RANDOM_REPEATS)]
            elapsed=time.perf_counter()-t0
            s=summarize_stochastic(vals,tc_full)
            all_rows.append({
                "L":L,"Method":"Random Sampling","Budget":budget,"Queries_Used":budget,
                **s,"FullGrid_Tc":tc_full,
                "Abs_Error_to_Exact_secondary":abs(s["Tc_mean"]-EXACT_TC),
                "Query_Reduction_Percent":(1-budget/nT)*100.0,
            })
            runtime_rows.append({"L":L,"Method":"Random Sampling","Budget":budget,"Seconds":elapsed})

    results=pd.DataFrame(all_rows)
    runtime_df=pd.DataFrame(runtime_rows)
    results.to_csv(OUTPUT_DIR/"comparison_all_results.csv",index=False)
    runtime_df.to_csv(OUTPUT_DIR/"comparison_runtime.csv",index=False)

    max_budget=results["Budget"].max()
    final_df=results[results["Budget"]==max_budget].copy()
    final_df.to_csv(OUTPUT_DIR/"comparison_final_budget.csv",index=False)

    grouped=(results.groupby(["Method","Budget"],as_index=False)
             .agg(Mean_Error=("Abs_Error_to_FullGrid_mean","mean")))
    fig,ax=plt.subplots(figsize=(FIG_WIDTH,FIG_HEIGHT))
    for method,g in grouped.groupby("Method"):
        g=g.sort_values("Budget")
        ax.plot(g["Budget"],g["Mean_Error"],marker="o",markersize=MARKER_SIZE,linewidth=LINE_WIDTH,label=method)
    ax.set_xlabel("Temperature-query budget")
    ax.set_ylabel(r"Mean $|\hat{T}_c(L)-T_c^{\rm full}(L)|$")
    ax.set_title("Critical-Temperature Error vs Query Budget")
    ax.grid(alpha=0.25)
    ax.legend(prop={"weight":"bold","size":LEGEND_FONT_SIZE})
    save_pdf(fig,OUTPUT_DIR/"01_error_vs_budget")

    final_avg=(final_df.groupby("Method",as_index=False)
               .agg(Error=("Abs_Error_to_FullGrid_mean","mean"),
                    Reduction=("Query_Reduction_Percent","mean")))
    fig,ax=plt.subplots(figsize=(FIG_WIDTH,FIG_HEIGHT))
    for _,r in final_avg.iterrows():
        ax.scatter(r["Reduction"],r["Error"],s=120)
        ax.annotate(r["Method"],(r["Reduction"],r["Error"]),xytext=(5,5),
                    textcoords="offset points",fontweight="bold",fontsize=11)
    ax.set_xlabel("Mean temperature-query reduction (%)")
    ax.set_ylabel(r"Mean $T_c(L)$ error to full-grid reference")
    ax.set_title("Accuracy--Efficiency Tradeoff")
    ax.grid(alpha=0.25)
    save_pdf(fig,OUTPUT_DIR/"02_query_reduction_vs_error")

    final_plot=(final_df.groupby("Method",as_index=False)
                .agg(Error=("Abs_Error_to_FullGrid_mean","mean"))
                .sort_values("Error"))
    fig,ax=plt.subplots(figsize=(13,8))
    x=np.arange(len(final_plot))
    ax.bar(x,final_plot["Error"])
    ax.set_xticks(x)
    ax.set_xticklabels(final_plot["Method"],rotation=22,ha="right",fontweight="bold")
    ax.set_ylabel(r"Mean $T_c(L)$ error")
    ax.set_title(f"Method Comparison at Query Budget = {max_budget}")
    ax.grid(axis="y",alpha=0.25)
    save_pdf(fig,OUTPUT_DIR/"03_final_method_comparison")

    runtime_avg=(runtime_df.groupby("Method",as_index=False)
                 .agg(Seconds=("Seconds","mean")).sort_values("Seconds"))
    fig,ax=plt.subplots(figsize=(13,8))
    x=np.arange(len(runtime_avg))
    ax.bar(x,runtime_avg["Seconds"])
    ax.set_xticks(x)
    ax.set_xticklabels(runtime_avg["Method"],rotation=22,ha="right",fontweight="bold")
    ax.set_ylabel("Mean analysis runtime (s)")
    ax.set_title("Temperature-Selection Runtime")
    ax.grid(axis="y",alpha=0.25)
    save_pdf(fig,OUTPUT_DIR/"04_runtime_comparison")

    print("Saved comparative results to:",OUTPUT_DIR.resolve())
    print("Primary metric: error to finite-L full-grid pseudocritical Tc.")

if __name__=="__main__":
    main()
