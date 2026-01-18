import pandas as pd
MONTHS_ES_TO_EN = {
    "enero": "January",
    "febrero": "February",
    "marzo": "March",
    "abril": "April",
    "mayo": "May",
    "junio": "June",
    "julio": "July",
    "agosto": "August",
    "septiembre": "September",
    "octubre": "October",
    "noviembre": "November",
    "diciembre": "December",
}


def group_foreigners(df):
    """
    Normaliza la variable nacionalidad en dos categorías:
    - espanola
    - extranjera

    Regla:
    - cualquier variante de 'Espaniola' -> espanola
    - cualquier otro valor -> extranjera
    """
    df = df.copy()

    df["nacionalidad"] = (
        df["nacionalidad"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["nacionalidad"] = df["nacionalidad"].apply(
        lambda x: "espanola" if "espan" in x else "extranjera"
    )

    return df

def compute_mean_annual_population(df):
    """
    Calcula la población femenina media anual (15–49) a partir de cortes
    semestrales (enero / julio).

    Supuestos:
    - La media simple enero-julio aproxima la población media anual.
    - La población está desagregada por grupo de edad y nacionalidad.
    - La columna 'anio' contiene fechas en formato textual en español."""
    df = df.copy()

    # 1️⃣ Convertir población a numérico
    df["poblacion"] = (
        df["poblacion"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .astype(int)
    )

    # 2️⃣ Normalizar meses al inglés (robusto al locale)
    for es, en in MONTHS_ES_TO_EN.items():
        df["anio"] = df["anio"].str.replace(es, en, regex=False)

    # 3️⃣ Parsear fecha
    df["fecha"] = pd.to_datetime(
        df["anio"],
        format="%d de %B de %Y",
        errors="coerce"
    )

    # 4️⃣ Extraer año calendario
    df["anio"] = df["fecha"].dt.year

    # 5️⃣ Eliminar observaciones sin fecha válida
    df = df.dropna(subset=["anio"])

    df["anio"] = df["anio"].astype(int)

    # 6️⃣ Calcular población media anual
    df_mean = (
        df.groupby(
            ["anio", "grupo_edad", "nacionalidad"],
            as_index=False
        )["poblacion"]
        .mean()
    )

    return df_mean
 
def normalize_official_tfr(df):
    """
    Normaliza el TFR oficial para expresarlo en hijos por mujer.

    Supuesto:
    - La columna 'tfr' está expresada como suma de tasas específicas
      por 1.000 mujeres (convención estadística del origen).

    Devuelve el mismo DataFrame con:
    - columna 'tfr_normalizado' en hijos por mujer
    """
    df = df.copy()
    df["tfr_normalizado"] = df["tfr"] / 1000
    return df
def rescale_official_tfr(tfr_csv, tfr_calculated):
    """
    Ajusta la escala del TFR oficial para hacerlo comparable con el
    TFR reconstruido a partir de tasas específicas.

    El factor de escala se estima empíricamente como la media del cociente:
        tfr_calculado / tfr_csv

    Supuesto:
    - El TFR oficial y el reconstruido miden el mismo fenómeno
      pero en escalas distintas.
    """
    df = tfr_csv.merge(
        tfr_calculated,
        on=["anio", "nacionalidad"],
        how="inner"
    )

    # Factor empírico medio por nacionalidad
    scale_factors = (
        df.groupby("nacionalidad")
        .apply(lambda x: (x["tfr_calculado"] / x["tfr"]).mean())
        .to_dict()
    )

    df["tfr_csv_ajustado"] = df.apply(
        lambda r: r["tfr"] * scale_factors[r["nacionalidad"]],
        axis=1
    )

    return df, scale_factors