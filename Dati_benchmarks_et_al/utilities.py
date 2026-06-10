import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

suffixes = {
    "schur"        : "_sp",     # Algorithms
    "realschur"    : "_[r]sp", 
    "complexschur" : "_[c]sp",
    "transfree"    : "",   
    "taylor"               : "_t",  # Approximants
    "scaling_and_squaring" : "_sas",
    "diagonalcheap"        : "_d"
}


def produce_plot(
        df:pd.DataFrame,
        ax,
        hue="exp",
        window=None
    ) -> None:
    # sort rows by value of u * k
    df_sorted = df.loc[(df["epsilon"] * df["cond_expA_F"]).sort_values(ascending=False).index]

    # the "hue" column is used to distinguish the configurations
    if hue=="Ytrue_method":
        df_sorted["hue"] = df_sorted["Ytrue_method"]
    else:
        df_sorted["hue"]  = "exp" 
        df_sorted["hue"] += df_sorted["algorithm"].map(
                #lambda a: "_sp" if a in ["realschur", "complexschur"] else ""
                lambda a: suffixes.get(a, "")
            )
        df_sorted["hue"] += df_sorted["approximant"].map(
                #lambda a: "_"+"".join([x.lower()[0] for x in a.split("_")])
                lambda a: suffixes.get(a, "")
            )
    
    df_sorted["err_ubound"] = df_sorted["epsilon"]*df_sorted["cond_expA_F"]

    if window:
        low, up = window
        if up < low:
            raise ValueError("Invalid window: upper bound is greater than lower bound")
        df_sorted["rel_err_F"] = df_sorted["rel_err_F"].map(
            lambda x: x if low<=x<=up else (up if x>up else low)
        )
        df_sorted["err_ubound"] = df_sorted["err_ubound"].map(
            lambda x: x if low<=x<=up else (up if x>up else low)
        )
        ax.axhline(y=low, color="grey", alpha=0.5, linestyle="--")
        ax.axhline(y=up, color="grey", alpha=0.5, linestyle="--")

    sns.scatterplot(x=range(len(df_sorted)), y=df_sorted["rel_err_F"],
                    ax=ax, hue=df_sorted["hue"], style=df_sorted["hue"])
    ax.plot(range(len(df_sorted)), df_sorted["err_ubound"],
            color="teal", alpha = 0.8, label="k * u")
    
    ax.set_xlabel("Sample", fontsize=12)
    ax.set_ylabel("rel_err_F", fontsize=12)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)
    

def produce_perfprof(
        ax,
        taus,
        solver_vals:dict,
        solvers,
        unify_schur:None|str=None
    ) -> None:

    if unify_schur:
        if unify_schur in ["mean", "average", "avg", "media"]:
            combfun = np.average
        elif unify_schur in ["max", "maximum", "massimo"]:
            combfun = np.max
        elif unify_schur in ["min", "minimum", "minimo"]:
            combfun = np.min
        elif unify_schur in ["complex", "complexschur"]:
            combfun = lambda x: x[0]
        elif unify_schur in ["real", "realschur"]:
            combfun = lambda x: x[1]
        else: ValueError(f"unify_schur option {unify_schur} not valid! (use None for no unification)")
        
        # combiniamo i "realschur" e "complexschur" in un unico "schur"
        solver_vals["schur_diagonalcheap"] = np.array(
            [combfun([c,r]) for c,r in zip(
                    solver_vals["complexschur_diagonalcheap"],
                    solver_vals["realschur_diagonalcheap"])])
        solver_vals["schur_taylor"] = np.array(
            [combfun([c,r]) for c,r in zip(
                    solver_vals["complexschur_taylor"],
                    solver_vals["realschur_taylor"])])  
        # escludiamo i "realschur" e "complexschur" dai solvers
        solvers = [s for s in solver_vals.keys() if not ("realschur" in s or "complexschur" in s)]

    for solver in solvers:
        label = "exp"
        alg, *appx_parts = solver.split("_")
        appx = "_".join(appx_parts)
        #label += "_sp" if "schur" in alg.lower() else ""
        label += suffixes.get(alg, "")
        #label += "_"+"".join([x.lower()[0] for x in appx.split("_")])
        label += suffixes.get(appx, "")
        ax.step(taus, solver_vals[solver], where="post", label=label)

    ax.set_xlabel("Tau", fontsize=12)
    ax.set_ylabel("Fraction of problems", fontsize=12)
    ax.set_xlim(1, max(taus))
    ax.legend()
    ax.grid(True, alpha=0.7)