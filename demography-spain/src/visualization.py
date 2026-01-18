import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

from data_ingestion import load_births, load_women_15_49,load_fertility_rates
from preprocessing import group_foreigners, compute_mean_annual_population
from analysis import compute_tfr_from_rates, compare_asfr_by_age
# =====================
# DATA
# =====================
births = group_foreigners(load_births())
women = group_foreigners(load_women_15_49())
women_mean = compute_mean_annual_population(women)

# Restricción temporal
births = births[(births["anio"] >= 2002) & (births["anio"] <= 2024)]
women_mean = women_mean[(women_mean["anio"] >= 2002) & (women_mean["anio"] <= 2024)]

# Tasa de nacimientos por 1.000 mujeres
df_rate = (
    births.merge(
        women_mean.groupby(["anio", "nacionalidad"], as_index=False)["poblacion"].sum(),
        on=["anio", "nacionalidad"]
    )
)

df_rate["rate_per_1000"] = df_rate["nacimientos"] / df_rate["poblacion"] * 1000
# =====================
# GRÁFICA A1
# =====================
plt.figure()
for nat in ["espanola", "extranjera"]:
    sub = df_rate[df_rate["nacionalidad"] == nat]
    plt.plot(sub["anio"], sub["rate_per_1000"], label=nat)

plt.xlabel("Año")
plt.ylabel("Nacimientos por 1.000 mujeres (15–49)")
plt.title("Evolución de la tasa de nacimientos")
plt.legend()
plt.show()
pivot = df_rate.pivot(
    index="anio",
    columns="nacionalidad",
    values="rate_per_1000"
)

pivot["ratio"] = pivot["extranjera"] / pivot["espanola"]

plt.figure()
plt.plot(pivot.index, pivot["ratio"])
plt.axhline(1)
plt.xlabel("Año")
plt.ylabel("Ratio extranjeras / españolas")
plt.title("Ratio de intensidad reproductiva")
plt.show()


fertility = load_fertility_rates().rename(columns={"Nacionalidad": "nacionalidad"})
fertility = group_foreigners(fertility)
fertility = fertility[(fertility["anio"] >= 2002) & (fertility["anio"] <= 2024)]

latest_year = fertility["anio"].max()
asfr = compare_asfr_by_age(fertility)
asfr_latest = asfr[asfr["anio"] == latest_year]
plt.figure()
plt.plot(asfr_latest["grupo_edad"], asfr_latest["tasa_es"], label="espanola")
plt.plot(asfr_latest["grupo_edad"], asfr_latest["tasa_ex"], label="extranjera")
plt.xlabel("Grupo de edad")
plt.ylabel("Tasa específica de fecundidad")
plt.title(f"Perfil de fecundidad por edad ({latest_year})")
plt.legend()
plt.xticks(rotation=45)
plt.show()

tfr = compute_tfr_from_rates(fertility)

plt.figure()
for nat in ["espanola", "extranjera"]:
    sub = tfr[tfr["nacionalidad"] == nat]
    plt.plot(sub["anio"], sub["tfr_calculado"], label=nat)

plt.xlabel("Año")
plt.ylabel("Hijos por mujer")
plt.title("TFR reconstruido por nacionalidad")
plt.legend()
plt.show()
years = [2002, 2010, 2015, 2020, 2024]

summary = (
    df_rate.merge(
        tfr,
        on=["anio", "nacionalidad"]
    )
)

summary = summary[summary["anio"].isin(years)]

table = summary.pivot_table(
    index="anio",
    columns="nacionalidad",
    values=["rate_per_1000", "tfr_calculado"]
).round(2)

print("\nTABLA D1 — Indicadores clave")
print(table)
