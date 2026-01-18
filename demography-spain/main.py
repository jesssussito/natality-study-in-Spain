import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# =====================
# PATHS
# =====================
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

# =====================
# IMPORTS
# =====================
from data_ingestion import (
    load_births,
    load_women_15_49,
    load_fertility_rates,
)
from preprocessing import (
    group_foreigners,
    compute_mean_annual_population,
)
from analysis import (
    compare_asfr_by_age,
    compute_tfr_from_rates,
    mean_age_at_childbearing,
    build_population_mean_15_49,
    kitagawa_decomposition,
    merge_population_and_fertility_rates
)

# =====================
# MAIN
# =====================
def main():

    # -----------------
    # DATA INGESTION
    # -----------------
    births = group_foreigners(load_births())
    women = group_foreigners(load_women_15_49())

    fertility = load_fertility_rates().rename(
        columns={"Nacionalidad": "nacionalidad"}
    )
    fertility = group_foreigners(fertility)

    women_mean = build_population_mean_15_49(women)

    # -----------------
    # RESTRICCIÓN TEMPORAL
    # -----------------
    births = births[(births["anio"] >= 2002) & (births["anio"] <= 2024)]
    women_mean = women_mean[(women_mean["anio"] >= 2002) & (women_mean["anio"] <= 2024)]
    fertility = fertility[(fertility["anio"] >= 2002) & (fertility["anio"] <= 2024)]
    population_mean_15_49 = build_population_mean_15_49(women)

    population_and_rates_df = merge_population_and_fertility_rates(
        population_mean_15_49,      # población media por edad y nacionalidad
        fertility        # tasas ASFR por edad y nacionalidad
    )
    print("\n--- DESCOMPOSICIÓN KITAGAWA ---")
    print(kitagawa_decomposition(population_and_rates_df, 2010))
    print(kitagawa_decomposition(population_and_rates_df, 2020))


    # =====================
    # BLOQUE A1 — YA EXISTENTE
    # =====================
    df_rate = births.merge(
    women_mean.groupby(["anio", "nacionalidad"], as_index=False)
    ["poblacion"]
    .sum(),
    on=["anio", "nacionalidad"],
)



    df_rate["rate_per_1000"] = (
    df_rate["nacimientos"] / df_rate["poblacion"] * 1000

)
    df_rate["rate_smoothed"] = (
    df_rate.groupby("nacionalidad")["rate_per_1000"]
    .transform(lambda x: x.rolling(3, center=True, min_periods=1).mean())
)



    plt.figure()
    for nat in ["espanola", "extranjera"]:
        sub = df_rate[df_rate["nacionalidad"] == nat]
        plt.plot(sub["anio"], sub["rate_smoothed"], label=nat)


    plt.xlabel("Año")
    plt.ylabel("Nacimientos por 1.000 mujeres (15–49)")
    plt.title("Gráfica A1 — Tasa de nacimientos por 1.000 mujeres")
    plt.legend()
    plt.show()

    # =====================
    # BLOQUE A2 — YA EXISTENTE
    # =====================
    pivot = df_rate.pivot(
        index="anio",
        columns="nacionalidad",
        values="rate_per_1000",
    )

    pivot["ratio"] = pivot["extranjera"] / pivot["espanola"]
    pivot["ratio_smoothed"] = pivot["ratio"].rolling(3, center=True, min_periods=1).mean()

    plt.figure()
    plt.plot(pivot.index, pivot["ratio_smoothed"])

    plt.axhline(1)
    plt.xlabel("Año")
    plt.ylabel("Ratio extranjeras / españolas")
    plt.title("Gráfica A2 — Ratio de intensidad reproductiva")
    plt.show()

    # =====================
    # BLOQUE B1 — YA EXISTENTE
    # =====================
    asfr = compare_asfr_by_age(fertility)
    latest_year = asfr["anio"].max()
    asfr_latest = asfr[asfr["anio"] == latest_year]

    plt.figure()
    plt.plot(asfr_latest["grupo_edad"], asfr_latest["tasa_es"], label="espanola")
    plt.plot(asfr_latest["grupo_edad"], asfr_latest["tasa_ex"], label="extranjera")
    plt.xlabel("Grupo de edad")
    plt.ylabel("Tasa específica de fecundidad")
    plt.title(f"Gráfica B1 — Perfil de fecundidad por edad ({latest_year})")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

    # =========================================================
    # 🔹 OPCIONAL 1 — DIFERENCIAL DE ASFR POR EDAD
    # =========================================================
    plt.figure()
    plt.bar(
        asfr_latest["grupo_edad"],
        asfr_latest["diferencial_absoluto"]
    )
    plt.axhline(0)
    plt.xlabel("Grupo de edad")
    plt.ylabel("Diferencial ASFR (extranjera − española)")
    plt.title(f"Diferencial de fecundidad por edad ({latest_year})")
    plt.xticks(rotation=45)
    plt.show()

    # =========================================================
    # 🔹 OPCIONAL 2 — EDAD MEDIA A LA MATERNIDAD (TEMPO)
    # =========================================================
    mac = mean_age_at_childbearing(fertility)

    plt.figure()
    for nat in ["espanola", "extranjera"]:
        sub = mac[mac["nacionalidad"] == nat]
        plt.plot(sub["anio"], sub["edad_media_maternidad"], label=nat)

    plt.xlabel("Año")
    plt.ylabel("Edad media a la maternidad")
    plt.title("Evolución de la edad media a la maternidad")
    plt.legend()
    plt.show()

    # =========================================================
    # 🔹 OPCIONAL 3 — HEATMAP ASFR (edad × año)
    # =========================================================
    for nat in ["espanola", "extranjera"]:
        df_nat = fertility[fertility["nacionalidad"] == nat]

        heatmap = df_nat.pivot_table(
            index="grupo_edad",
            columns="anio",
            values="tasa"
        )

        plt.figure()
        plt.imshow(heatmap, aspect="auto")
        plt.colorbar(label="Tasa específica de fecundidad")
        plt.yticks(range(len(heatmap.index)), heatmap.index)
        plt.xticks(range(0, len(heatmap.columns), 2),
                   heatmap.columns[::2],
                   rotation=45)
        plt.xlabel("Año")
        plt.ylabel("Grupo de edad")
        plt.title(f"Heatmap ASFR — {nat}")
        plt.show()
    # =====================
    # TABLA PROFESIONAL ANUAL
    # =====================

    # 1️⃣ Tasas agregadas
    rates = df_rate[["anio", "nacionalidad", "rate_per_1000"]]

    # 2️⃣ TFR anual
    tfr_df = compute_tfr_from_rates(fertility)

    # 3️⃣ Edad media anual
    mac_df = mean_age_at_childbearing(fertility)

    # 4️⃣ Merge completo
    summary_full = rates.merge(tfr_df, on=["anio", "nacionalidad"])
    summary_full = summary_full.merge(mac_df, on=["anio", "nacionalidad"])

    # 5️⃣ Pivot para vista final
    table_full = summary_full.pivot_table(
        index="anio",
        columns="nacionalidad",
        values=["rate_per_1000", "tfr_calculado", "edad_media_maternidad"]
    ).round(2)

    print("\n=== TABLA SINTÉTICA ANUAL ===")
    print(table_full)
    # =====================
    # TABLA FINAL
    # =====================
    tfr = compute_tfr_from_rates(fertility)
    years = [2002, 2010, 2015, 2020, 2024]

    summary = df_rate.merge(
        tfr,
        on=["anio", "nacionalidad"],
    )

    summary = summary[summary["anio"].isin(years)]

    table = summary.pivot_table(
        index="anio",
        columns="nacionalidad",
        values=["rate_per_1000", "tfr_calculado"],
    ).round(2)

    print("\n=== TABLA D1 — INDICADORES CLAVE ===")
    print(table)
    result_2010 = kitagawa_decomposition(population_and_rates_df, 2010)
    result_2020 = kitagawa_decomposition(population_and_rates_df, 2020)

    print(result_2010)
    print(result_2020)

# =====================
# RUN
# =====================
if __name__ == "__main__":
    main()
