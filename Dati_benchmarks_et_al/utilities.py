import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def produce_plot(
        df:pd.DataFrame,
        ax
    ) -> None:
    # sort rows by value of u * k
    df_sorted = df.loc[(df["epsilon"] * df["cond_expA_F"]).sort_values(ascending=False).index]

    # the "hue" column is used to distinguish the configurations
    df_sorted["hue"]  = "exp" 
    df_sorted["hue"] += df_sorted["algorithm"].map(
            lambda a: "_sp" if a in ["realschur", "complexschur"] else ""
        )
    df_sorted["hue"] += df_sorted["approximant"].map(
            lambda a: "_"+"".join([x[0] for x in a.split("_")])
        )

    sns.scatterplot(x=range(len(df_sorted)), y=df_sorted["rel_err_F"],
                    ax=ax, hue=df_sorted["hue"], style=df_sorted["hue"])
    ax.plot(range(len(df_sorted)), df_sorted["epsilon"]*df_sorted["cond_expA_F"],
            color="teal", alpha = 0.8, label="k * u")
    
    ax.set_xlabel("Matrix", fontsize=12)
    ax.set_ylabel("rel_err_F", fontsize=12)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)
    