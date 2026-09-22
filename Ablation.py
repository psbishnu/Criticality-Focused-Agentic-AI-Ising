#!/usr/bin/env python3
"""
Ablation analysis for the proposed.

Default lattice sizes:
    L = 32, 64, 128

Ablations
---------
1. Full Agent
2. No GP uncertainty
3. No critical-region focus
4. No physics bracketing critic
5. No autonomous stopping
6. Frozen adaptive memory
7. Random acquisition (negative control)

Primary target:
    finite-L full-grid pseudocritical Tc(L)

Outputs:
Results/Ablation_PRE/
    ablation_all_results.csv
    ablation_summary.csv
    ablation_runtime.csv
    01_ablation_error.pdf
    02_ablation_query_usage.pdf
    03_ablation_efficiency_tradeoff.pdf
    04_ablation_runtime.pdf
"""

from pathlib import Path

DATASETS = {
    32: Path("../JOB5_Noise/J5Data/MCD32.csv"),
    64: Path("../JOB5_Noise/J5Data/MCD64.csv"),
    128: Path("../JOB5_Noise/J5Data/MCD128.csv"),
}
LATTICE_SIZES=[32,64,128]

TEMPERATURE_COLUMN="Temperature"
SPIN_PREFIX="spin_"
K_B=1.0

MAX_TEMPERATURE_QUERIES=20
N_INITIAL_TEMPERATURES=6

MIN_QUERIES_BEFORE_STOP=12
STABILITY_WINDOW=5
TC_STABILITY_TOL=0.015

GP_LENGTH_SCALE=0.20
GP_NOISE_LEVEL=1e-6

EXPLORATION_WEIGHT=0.40
CRITICALITY_WEIGHT=0.45
BRACKETING_WEIGHT=0.15
CRITICALITY_WIDTH=0.20
BRACKET_WIDTH=0.18

RANDOM_SEED=42
RANDOM_NEGATIVE_CONTROL_REPEATS=20

RESULTS_ROOT=Path("Results")
OUTPUT_DIR=RESULTS_ROOT/"Ablation_PRE"

FIG_WIDTH=11
FIG_HEIGHT=8
FONT_SIZE=17
AXIS_FONT_SIZE=19
TITLE_FONT_SIZE=21
LEGEND_FONT_SIZE=12

EXACT_TC=2.0/__import__("math").log(1.0+__import__("math").sqrt(2.0))

import gc
import time
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,RBF,WhiteKernel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore",category=RuntimeWarning)

plt.rcParams.update({
    "font.size":FONT_SIZE,
    "font.weight":"bold",
    "axes.labelsize":AXIS_FONT_SIZE,
    "axes.labelweight":"bold",
    "axes.titlesize":TITLE_FONT_SIZE,
    "axes.titleweight":"bold",
    "xtick.labelsize":FONT_SIZE,
    "ytick.labelsize":FONT_SIZE,
    "legend.fontsize":LEGEND_FONT_SIZE,
})

VARIANTS={
    "Full Agent":{
        "uncertainty":True,"criticality":True,"bracketing":True,
        "early_stop":True,"frozen_memory":False,"random_acquisition":False,
    },
    "No Uncertainty":{
        "uncertainty":False,"criticality":True,"bracketing":True,
        "early_stop":True,"frozen_memory":False,"random_acquisition":False,
    },
    "No Criticality Focus":{
        "uncertainty":True,"criticality":False,"bracketing":True,
        "early_stop":True,"frozen_memory":False,"random_acquisition":False,
    },
    "No Physics Critic":{
        "uncertainty":True,"criticality":True,"bracketing":False,
        "early_stop":True,"frozen_memory":False,"random_acquisition":False,
    },
    "No Autonomous Stop":{
        "uncertainty":True,"criticality":True,"bracketing":True,
        "early_stop":False,"frozen_memory":False,"random_acquisition":False,
    },
    "Frozen Memory":{
        "uncertainty":True,"criticality":True,"bracketing":True,
        "early_stop":False,"frozen_memory":True,"random_acquisition":False,
    },
    "Random Acquisition":{
        "uncertainty":False,"criticality":False,"bracketing":False,
        "early_stop":False,"frozen_memory":False,"random_acquisition":True,
    },
}

def save_pdf(fig,outbase):
    fig.tight_layout()
    fig.savefig(str(outbase)+".pdf",bbox_inches="tight")
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
    header=pd.read_csv(path,nrows=0).columns.tolist()
    spin_cols=sorted([c for c in header if c.startswith(SPIN_PREFIX)],key=spin_sort_key)
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
        rows.append({"Temperature":float(temp),"Susceptibility":float(chi),"N_configs":int(len(g))})
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

def current_tc(agg,selected):
    s=agg.iloc[sorted(set(selected))]
    return parabolic_peak_tc(s["Temperature"],s["Susceptibility"])

def run_variant(agg,cfg,seed=RANDOM_SEED):
    rng=np.random.default_rng(seed)
    n=len(agg); budget=min(MAX_TEMPERATURE_QUERIES,n)
    selected=list(dict.fromkeys(np.linspace(0,n-1,min(N_INITIAL_TEMPERATURES,budget),dtype=int).tolist()))
    tc_history=[current_tc(agg,selected)]

    frozen_gp=None
    if cfg["frozen_memory"]:
        s=agg.iloc[selected]
        frozen_gp=gp_fit(s["Temperature"],s["Susceptibility"])

    while len(set(selected))<budget:
        if cfg["early_stop"] and len(set(selected))>=MIN_QUERIES_BEFORE_STOP and len(tc_history)>=STABILITY_WINDOW:
            recent=np.asarray(tc_history[-STABILITY_WINDOW:],float)
            if np.all(np.isfinite(recent)) and np.max(recent)-np.min(recent)<=TC_STABILITY_TOL:
                tc=recent[-1]
                selected_t=agg.iloc[selected]["Temperature"].to_numpy(float)
                left=np.any((selected_t<tc)&(selected_t>tc-BRACKET_WIDTH))
                right=np.any((selected_t>tc)&(selected_t<tc+BRACKET_WIDTH))
                if left and right:
                    break

        unselected=[i for i in range(n) if i not in selected]

        if cfg["random_acquisition"]:
            selected.append(int(rng.choice(unselected)))
            tc_history.append(current_tc(agg,selected))
            continue

        s=agg.iloc[selected]
        gp=frozen_gp if cfg["frozen_memory"] else gp_fit(s["Temperature"],s["Susceptibility"])
        temps=agg["Temperature"].to_numpy(float)
        mu,std=gp.predict(temps.reshape(-1,1),return_std=True)
        predicted_tc=float(temps[int(np.argmax(mu))])
        criticality=np.exp(-0.5*((temps-predicted_tc)/max(CRITICALITY_WIDTH,1e-8))**2)

        selected_t=temps[np.asarray(selected,dtype=int)]
        bracket=np.zeros_like(temps)

        if cfg["bracketing"]:
            left=np.any((selected_t<predicted_tc)&(selected_t>predicted_tc-BRACKET_WIDTH))
            right=np.any((selected_t>predicted_tc)&(selected_t<predicted_tc+BRACKET_WIDTH))
            if not left:
                bracket+=np.exp(-0.5*((temps-(predicted_tc-BRACKET_WIDTH/2))/max(BRACKET_WIDTH/2,1e-8))**2)
            if not right:
                bracket+=np.exp(-0.5*((temps-(predicted_tc+BRACKET_WIDTH/2))/max(BRACKET_WIDTH/2,1e-8))**2)

        acq=np.zeros_like(temps,float)
        if cfg["uncertainty"]:
            acq+=EXPLORATION_WEIGHT*normalize(std)
        if cfg["criticality"]:
            acq+=CRITICALITY_WEIGHT*normalize(criticality)
        if cfg["bracketing"]:
            acq+=BRACKETING_WEIGHT*normalize(bracket)
        if np.max(acq)-np.min(acq)<1e-14:
            acq=normalize(std)

        acq[np.asarray(selected,dtype=int)]=-np.inf
        selected.append(int(np.argmax(acq)))
        tc_history.append(current_tc(agg,selected))

    return {"Tc":current_tc(agg,selected),"Queries":len(set(selected))}

def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    rows=[]; runtime_rows=[]

    for L in LATTICE_SIZES:
        agg=load_aggregate(DATASETS[L],L)
        tc_full=parabolic_peak_tc(agg["Temperature"],agg["Susceptibility"])
        nT=len(agg)

        for name,cfg in VARIANTS.items():
            if name=="Random Acquisition":
                estimates=[]; queries=[]
                t0=time.perf_counter()
                for r in range(RANDOM_NEGATIVE_CONTROL_REPEATS):
                    res=run_variant(agg,cfg,seed=RANDOM_SEED+r)
                    estimates.append(res["Tc"]); queries.append(res["Queries"])
                elapsed=time.perf_counter()-t0
                estimates=np.asarray(estimates,float)
                qmean=float(np.mean(queries))
                rows.append({
                    "L":L,"Variant":name,
                    "Tc_mean":float(np.mean(estimates)),
                    "Tc_std":float(np.std(estimates,ddof=1)),
                    "FullGrid_Tc":tc_full,
                    "Abs_Error_to_FullGrid":float(np.mean(np.abs(estimates-tc_full))),
                    "Queries_Used":qmean,
                    "Query_Reduction_Percent":(1-qmean/nT)*100.0,
                    "Abs_Error_to_Exact_secondary":abs(float(np.mean(estimates))-EXACT_TC),
                })
                runtime_rows.append({"L":L,"Variant":name,"Seconds":elapsed})
            else:
                t0=time.perf_counter()
                res=run_variant(agg,cfg)
                elapsed=time.perf_counter()-t0
                rows.append({
                    "L":L,"Variant":name,
                    "Tc_mean":res["Tc"],"Tc_std":0.0,
                    "FullGrid_Tc":tc_full,
                    "Abs_Error_to_FullGrid":abs(res["Tc"]-tc_full),
                    "Queries_Used":res["Queries"],
                    "Query_Reduction_Percent":(1-res["Queries"]/nT)*100.0,
                    "Abs_Error_to_Exact_secondary":abs(res["Tc"]-EXACT_TC),
                })
                runtime_rows.append({"L":L,"Variant":name,"Seconds":elapsed})

    df=pd.DataFrame(rows)
    rt=pd.DataFrame(runtime_rows)
    df.to_csv(OUTPUT_DIR/"ablation_all_results.csv",index=False)
    rt.to_csv(OUTPUT_DIR/"ablation_runtime.csv",index=False)

    summary=(df.groupby("Variant",as_index=False)
             .agg(Mean_Tc_Error=("Abs_Error_to_FullGrid","mean"),
                  Mean_Queries=("Queries_Used","mean"),
                  Mean_Query_Reduction_Percent=("Query_Reduction_Percent","mean"),
                  Mean_Exact_Error_Secondary=("Abs_Error_to_Exact_secondary","mean"))
             .sort_values("Mean_Tc_Error"))
    summary.to_csv(OUTPUT_DIR/"ablation_summary.csv",index=False)

    fig,ax=plt.subplots(figsize=(13,8))
    x=np.arange(len(summary))
    ax.bar(x,summary["Mean_Tc_Error"])
    ax.set_xticks(x)
    ax.set_xticklabels(summary["Variant"],rotation=22,ha="right",fontweight="bold")
    ax.set_ylabel(r"Mean $|\hat{T}_c(L)-T_c^{\rm full}(L)|$")
    ax.set_title("Ablation Study: Critical-Temperature Error")
    ax.grid(axis="y",alpha=0.25)
    save_pdf(fig,OUTPUT_DIR/"01_ablation_error")

    fig,ax=plt.subplots(figsize=(13,8))
    x=np.arange(len(summary))
    ax.bar(x,summary["Mean_Queries"])
    ax.set_xticks(x)
    ax.set_xticklabels(summary["Variant"],rotation=22,ha="right",fontweight="bold")
    ax.set_ylabel("Mean temperature queries used")
    ax.set_title("Ablation Study: Query Usage")
    ax.grid(axis="y",alpha=0.25)
    save_pdf(fig,OUTPUT_DIR/"02_ablation_query_usage")

    fig,ax=plt.subplots(figsize=(11,8))
    for _,r in summary.iterrows():
        ax.scatter(r["Mean_Query_Reduction_Percent"],r["Mean_Tc_Error"],s=120)
        ax.annotate(r["Variant"],(r["Mean_Query_Reduction_Percent"],r["Mean_Tc_Error"]),
                    xytext=(5,5),textcoords="offset points",fontweight="bold",fontsize=11)
    ax.set_xlabel("Mean temperature-query reduction (%)")
    ax.set_ylabel("Mean Tc(L) error")
    ax.set_title("Ablation Accuracy--Efficiency Tradeoff")
    ax.grid(alpha=0.25)
    save_pdf(fig,OUTPUT_DIR/"03_ablation_efficiency_tradeoff")

    rt_summary=(rt.groupby("Variant",as_index=False)
                .agg(Seconds=("Seconds","mean")).sort_values("Seconds"))
    fig,ax=plt.subplots(figsize=(13,8))
    x=np.arange(len(rt_summary))
    ax.bar(x,rt_summary["Seconds"])
    ax.set_xticks(x)
    ax.set_xticklabels(rt_summary["Variant"],rotation=22,ha="right",fontweight="bold")
    ax.set_ylabel("Mean analysis runtime (s)")
    ax.set_title("Ablation Runtime")
    ax.grid(axis="y",alpha=0.25)
    save_pdf(fig,OUTPUT_DIR/"04_ablation_runtime")

    print("Saved ablation results to:",OUTPUT_DIR.resolve())
    print(summary.to_string(index=False))

if __name__=="__main__":
    main()
