def build_population_mean_15_49(population):
    """
    Construye la población femenina media anual 15–49
    a partir de recuentos a 1 enero / 1 julio.

    Resultado:
    - anio
    - nacionalidad
    - poblacion_media_15_49
    """

    # Nos quedamos SOLO con edades fértiles estándar
    valid_ages = [
        "De 15 a 19 anios",
        "De 20 a 24 anios",
        "De 25 a 29 anios",
        "De 30 a 34 anios",
        "De 35 a 39 anios",
        "De 40 a 44 anios",
        "De 45 a 49 anios",
    ]

    pop = population[population["grupo_edad"].isin(valid_ages)].copy()
    pop["anio"] = pop["anio"].str.extract(r"(\d{4})").astype(int)

    # Primero agregamos la población por edad y nacionalidad (sumar categorías)
    pop_agg = (
        pop.groupby(["anio", "grupo_edad", "nacionalidad"], as_index=False)
        ["poblacion"]
        .sum()
    )

    # Luego calculamos la media entre enero y julio
    pop_media = (
        pop_agg.groupby(["anio", "grupo_edad", "nacionalidad"], as_index=False)
        ["poblacion"]
        .mean()
    )
    pop_total = (
    pop_media.groupby(["anio", "nacionalidad"], as_index=False)["poblacion"]
    .sum()
    .rename(columns={"poblacion": "poblacion_media_15_49"})
    )

    return pop_media
def birth_rate_per_1000_women(births, population_mean_15_49):
    """
    Calcula la tasa anual de nacimientos por 1.000 mujeres 15–49.
    """
    
    births_agg = (
        births.groupby(["anio", "nacionalidad"], as_index=False)
        ["nacimientos"]
        .sum()
    )

    df = births_agg.merge(
        population_mean_15_49,
        on=["anio", "nacionalidad"],
        how="inner"
    )

    df["birth_rate_per_1000"] = (
        df["nacimientos"] / df["poblacion_media_15_49"] * 1000
    )

    return df

def fertility_intensity_ratio(df):
    """
    df debe contener:
    - anio
    - nacionalidad
    - birth_rate_per_1000
    """

    pivot = df.pivot_table(
        index="anio",
        columns="nacionalidad",
        values="birth_rate_per_1000"
    )

    pivot["fertility_intensity_ratio"] = (
        pivot["extranjera"] / pivot["espanola"]
    )

    return pivot.reset_index()

def merge_population_and_fertility_rates(population, fertility_rates):
    """
    Une población femenina media anual y tasas específicas de fecundidad
    por grupo de edad, año y nacionalidad, y calcula nacimientos esperados.

    IMPORTANTE:
    - Las tasas están expresadas como nacimientos por 1.000 mujeres.
    - Por tanto, se divide la tasa entre 1.000 antes de multiplicar.

    Devuelve un DataFrame con:
    - anio
    - grupo_edad
    - nacionalidad
    - poblacion
    - tasa (por 1.000 mujeres)
    - nacimientos_esperados
    """
    df = population.merge(
        fertility_rates,
        on=["anio", "grupo_edad", "nacionalidad"],
        how="inner"
    )

    # Corrección de unidad: tasa por 1.000 mujeres
    df["nacimientos_esperados"] = df["poblacion"] * (df["tasa"] / 1000)

    return df
def compute_tfr_from_rates(fertility_rates):
    """
    Calcula el TFR (Total Fertility Rate) a partir de tasas específicas por edad.

    Supuestos:
    - Las tasas están expresadas como nacimientos por 1.000 mujeres.
    - Los grupos de edad son quinquenales (amplitud = 5 años).
    - Se cubre el intervalo 15–49.

    Devuelve un DataFrame con:
    - anio
    - nacionalidad
    - tfr_calculado
    """
    df = fertility_rates.copy()

    # Convertir tasa por 1.000 a tasa por mujer
    df["tasa_por_mujer"] = df["tasa"] / 1000

    # Contribución de cada grupo quinquenal
    df["contribucion_tfr"] = df["tasa_por_mujer"] * 5

    tfr = (
        df.groupby(["anio", "nacionalidad"], as_index=False)
        ["contribucion_tfr"]
        .sum()
        .rename(columns={"contribucion_tfr": "tfr_calculado"})
    )

    return tfr
def kitagawa_decomposition(df, year):
    """
    Aplica una descomposición tipo Kitagawa del diferencial de fecundidad
    entre población española y extranjera para un año dado.

    Parámetros
    ----------
    df : DataFrame
        Debe contener:
        - anio
        - grupo_edad
        - nacionalidad ('espanola', 'extranjera')
        - poblacion
        - tasa (por 1.000 mujeres)

    year : int
        Año para el que se realiza la descomposición.
    """
    # Filtrar año
    d = df[df["anio"] == year].copy()

    # Separar grupos
    d_es = d[d["nacionalidad"] == "espanola"]
    d_ex = d[d["nacionalidad"] == "extranjera"]

    # Unir por grupo de edad
    m = d_es.merge(
        d_ex,
        on="grupo_edad",
        suffixes=("_es", "_ex")
    )

    # Convertir tasas a nacimientos por mujer
    m["f_es"] = m["tasa_es"] / 1000
    m["f_ex"] = m["tasa_ex"] / 1000

    # Pesos poblacionales
    m["w_es"] = m["poblacion_es"] / m["poblacion_es"].sum()
    m["w_ex"] = m["poblacion_ex"] / m["poblacion_ex"].sum()

    # Medias
    m["f_bar"] = (m["f_es"] + m["f_ex"]) / 2
    m["w_bar"] = (m["w_es"] + m["w_ex"]) / 2

    # Componentes
    m["efecto_estructura"] = (m["w_ex"] - m["w_es"]) * m["f_bar"]
    m["efecto_tasas"] = (m["f_ex"] - m["f_es"]) * m["w_bar"]

    efecto_estructura = m["efecto_estructura"].sum()
    efecto_tasas = m["efecto_tasas"].sum()
    diferencial_total = efecto_estructura + efecto_tasas

    return {
        "anio": year,
        "diferencial_total": diferencial_total,
        "efecto_estructura": efecto_estructura,
        "efecto_tasas": efecto_tasas,
        "contribuciones_por_edad": m[
            ["grupo_edad", "efecto_estructura", "efecto_tasas"]
        ],
    }

def compare_asfr_by_age(df):
    """
    Compara las tasas específicas de fecundidad por grupo de edad
    entre población española y extranjera para todos los años.

    Parámetros
    ----------
    df : DataFrame
        Debe contener:
        - anio
        - grupo_edad
        - nacionalidad ('espanola', 'extranjera')
        - tasa (por 1.000 mujeres)"""
    # Separar por nacionalidad
    df_es = df[df["nacionalidad"] == "espanola"]
    df_ex = df[df["nacionalidad"] == "extranjera"]

    # Unir por año y edad
    m = df_es.merge(
        df_ex,
        on=["anio", "grupo_edad"],
        suffixes=("_es", "_ex")
    )

    # Diferencias
    m["diferencial_absoluto"] = m["tasa_ex"] - m["tasa_es"]
    m["ratio_extranjera_espanola"] = m["tasa_ex"] / m["tasa_es"]

    return m[
        [
            "anio",
            "grupo_edad",
            "tasa_es",
            "tasa_ex",
            "diferencial_absoluto",
            "ratio_extranjera_espanola",
        ]
    ]
   
def mean_age_at_childbearing(df):
    """
    Calcula la edad media a la maternidad (MAC) por año y nacionalidad
    a partir de tasas específicas de fecundidad por edad.

    Parámetros
    ----------
    df : DataFrame
        Debe contener:
        - anio
        - grupo_edad (quinquenal)
        - nacionalidad
        - tasa (por 1.000 mujeres)

    """
    d = df.copy()

    # Mapear edad central de cada grupo quinquenal
    age_map = {
        "De 15 a 19 anios": 17.5,
        "De 20 a 24 anios": 22.5,
        "De 25 a 29 anios": 27.5,
        "De 30 a 34 anios": 32.5,
        "De 35 a 39 anios": 37.5,
        "De 40 a 44 anios": 42.5,
        "De 45 a 49 anios": 47.5,
    }

    # Filtrar solo edades fértiles estándar
    d = d[d["grupo_edad"].isin(age_map.keys())]

    # Asignar edad central
    d["edad_central"] = d["grupo_edad"].map(age_map)

    # MAC = sum(edad * tasa) / sum(tasa)
    mac = (
        d.groupby(["anio", "nacionalidad"])
        .apply(lambda x: (x["edad_central"] * x["tasa"]).sum() / x["tasa"].sum())
        .reset_index(name="edad_media_maternidad")
    )

    return mac
def build_pseudo_cohorts(df):
    """
    Construye pseudo-cohortes a partir de tasas específicas de fecundidad
    usando la edad central de los grupos quinquenales.

    Parámetros
    ----------
    df : DataFrame
        Debe contener:
        - anio
        - grupo_edad
        - nacionalidad
        - tasa (por 1.000 mujeres)

    """
    d = df.copy()

    age_map = {
        "De 15 a 19 anios": 17.5,
        "De 20 a 24 anios": 22.5,
        "De 25 a 29 anios": 27.5,
        "De 30 a 34 anios": 32.5,
        "De 35 a 39 anios": 37.5,
        "De 40 a 44 anios": 42.5,
        "De 45 a 49 anios": 47.5,
    }

    # Filtrar edades fértiles estándar
    d = d[d["grupo_edad"].isin(age_map.keys())]

    # Edad central
    d["edad"] = d["grupo_edad"].map(age_map)

    # Cohorte aproximada
    d["cohorte"] = (d["anio"] - d["edad"]).round().astype(int)

    return d[
        ["cohorte", "edad", "anio", "nacionalidad", "tasa"]
    ]
def compare_cohorts_by_age(df, cohort_min=None, cohort_max=None):
    """
    Compara tasas específicas por edad dentro de pseudo-cohortes.

    Parámetros
    ----------
    df : DataFrame
        Salida de build_pseudo_cohorts.
    cohort_min, cohort_max : int, opcional
        Filtro de cohortes.

    """
    d = df.copy()

    if cohort_min is not None:
        d = d[d["cohorte"] >= cohort_min]
    if cohort_max is not None:
        d = d[d["cohorte"] <= cohort_max]

    return (
        d.groupby(["cohorte", "edad", "nacionalidad"], as_index=False)
        ["tasa"]
        .mean()
    )