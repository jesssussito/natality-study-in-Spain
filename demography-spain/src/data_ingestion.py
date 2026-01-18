from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "processed"


def _validate_columns(df, expected_cols, name):
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"{name}: faltan columnas obligatorias: {missing}"
        )


def load_births():
    df = pd.read_csv(RAW_DATA_DIR / "births_by_nationality.csv", sep=",")
    _validate_columns(
        df,
        ["anio", "nacionalidad", "nacimientos"],
        "births_by_nationality",
    )
    return df


def load_women_15_49():
    df = pd.read_csv(
    RAW_DATA_DIR / "women_15_49_by_nationality.csv",
    sep=",",
    thousands="."
    )
    _validate_columns(
        df,
        ["grupo_edad", "nacionalidad", "anio", "poblacion"],
        "women_15_49_by_nationality",
    )
    print(df.dtypes)

    return df


def load_fertility_rates():
    df = pd.read_csv(
        RAW_DATA_DIR / "fertility_rates_by_age_and_nationality.csv", sep=","
    )
    _validate_columns(
        df,
        ["grupo_edad", "Nacionalidad", "anio", "tasa"],
        "fertility_rates_by_age_and_nationality",
    )
    return df


def load_tfr():
    df = pd.read_csv(RAW_DATA_DIR / "tfr_by_nationality.csv", sep=",")
    _validate_columns(
        df,
        ["anio", "nacionalidad", "tfr"],
        "tfr_by_nationality",
    )
    return df
