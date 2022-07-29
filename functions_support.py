from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
import json
from fastapi import FastAPI
from pydantic import BaseModel, Field
import datetime
from enum import Enum
from fastapi.logger import logger as fastapi_logger
from logging.handlers import RotatingFileHandler
from fastapi.encoders import jsonable_encoder
from typing import Union
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from pgmpy.inference import VariableElimination
from sklearn.pipeline import Pipeline, make_pipeline


reader = XMLBIFReader("first_model.xml")
model = reader.get_model()
inference = VariableElimination(model)
pl = joblib.load("pipeline.sav")

# Opening JSON file
with open("null_dict.json") as json_file:
    null_dict = json.load(json_file)

ord_cols = ["A_PARA", "A_GESTA", "EUTOCITO_ANTERIOR"]
int_cols = [
    "IDADE_MATERNA",
    "PESO_INICIAL",
    "IMC",
    "NUMERO_CONSULTAS_PRE_NATAL",
    "IDADE_GESTACIONAL_ADMISSAO",
    "SEMANAS_GESTACAO_PARTO",
    "ESTIMATIVA_PESO_ECO_30",
]
cat_cols = [
    "GS",
    "TRAB_PARTO_ENTRADA_ESPONTANEO",
    "VIGIADA_CENTRO_SAUDE",
    "TRAB_PARTO_NO_PARTO",
    "GRUPO_ROBSON",
    "VIGIADA_NESTE_HOSPITAL",
    "VIGIADA",
    "APRESENTACAO_ADMISSAO",
    "APRESENTACAO_NO_PARTO",
    "APRESENTACAO_30",
    "TIPO_PARTO",
    "TIPO_GRAVIDEZ",
]

cols = [
    "IDADE_MATERNA",
    "GS",
    "PESO_INICIAL",
    "IMC",
    "A_PARA",
    "A_GESTA",
    "EUTOCITO_ANTERIOR",
    "TIPO_GRAVIDEZ",
    "VIGIADA",
    "NUMERO_CONSULTAS_PRE_NATAL",
    "VIGIADA_CENTRO_SAUDE",
    "VIGIADA_NESTE_HOSPITAL",
    "ESTIMATIVA_PESO_ECO_30",
    "APRESENTACAO_30",
    "APRESENTACAO_ADMISSAO",
    "IDADE_GESTACIONAL_ADMISSAO",
    "TRAB_PARTO_ENTRADA_ESPONTANEO",
    "TIPO_PARTO",
    "APRESENTACAO_NO_PARTO",
    "TRAB_PARTO_NO_PARTO",
    "SEMANAS_GESTACAO_PARTO",
    "GRUPO_ROBSON",
]

network_cols = [
    "IDADE_MATERNA",
    "GS",
    "PESO_INICIAL",
    "IMC",
    "A_PARA",
    "A_GESTA",
    "EUTOCITO_ANTERIOR",
    "TIPO_GRAVIDEZ",
    "VIGIADA",
    "NUMERO_CONSULTAS_PRE_NATAL",
    "VIGIADA_CENTRO_SAUDE",
    "VIGIADA_NESTE_HOSPITAL",
    "ESTIMATIVA_PESO_ECO_30",
    "APRESENTACAO_30",
    "APRESENTACAO_ADMISSAO",
    "IDADE_GESTACIONAL_ADMISSAO",
    "TRAB_PARTO_ENTRADA_ESPONTANEO",
    "TIPO_PARTO",
    "APRESENTACAO_NO_PARTO",
    "TRAB_PARTO_NO_PARTO",
    "SEMANAS_GESTACAO_PARTO",
    "GRUPO_ROBSON",
]


def get_missing_score(row):
    score = 0
    opt = jsonable_encoder(row)
    for c in cols:
        print(
            opt[c], opt[c] == "nan", opt[c] == np.nan, type(opt[c]), pd.isnull(opt[c])
        )

        if pd.isnull(opt[c]):
            print("SEEEs")
            score += null_dict[c]
    return score / len(cols)


def create_missing(x, ord_cols, int_cols, cat_cols):
    for c in ord_cols:
        if c not in x.keys():
            x[c] = np.nan
    for c in int_cols:
        if c not in x.keys():
            x[c] = np.nan
    for c in cat_cols:
        if c not in x.keys():
            x[c] = np.nan
    return x


def transfrom_array_to_df_onehot(pl, nparray):
    col_list = []
    col_list_int = pl["preprocessor"].transformers_[0][2]  # changes col location
    # print(col_list_int)
    ordinal_col = pl["preprocessor"].transformers[1][2]
    original_col = pl["preprocessor"].transformers[2][2]
    col_list = col_list_int + ordinal_col

    col_list = col_list + original_col
    df1 = pd.DataFrame(nparray, columns=col_list)
    return df1


def get_correctness_score(row, model):
    score = 0
    opt = jsonable_encoder(row)

    df = pd.DataFrame(opt, index=[0])
    # print(df)
    df[cat_cols] = df[cat_cols].astype(str)
    # print(df.to_dict())
    x_treated = pl.transform(df)
    # print(opt)
    for c in network_cols:
        # print(c)

        df_evidence = transfrom_array_to_df_onehot(pl, x_treated)
        df_evidence = df_evidence.astype(str)
        truth = df_evidence[c].values[0]

        df_evidence.drop(
            columns=[c, "GS"], inplace=True
        )  # remove on pipeline for prod?
        evidence = df_evidence.to_dict("records").copy()[0]
        for k, v in evidence.items():
            evidence[k] = v.replace(".", "_")

        # print(evidence)
        query = inference.query(variables=[c], evidence=evidence, show_progress=True)
        pred = query.state_names[c][query.values.argmax()]
        perc = query.values
        print(perc.max())
        #  print(truth, pred)
        if truth == pred.replace("_", "."):
            score += perc.max() * 100
    return score / len(cols)


def calculate_score(missing_score, correctness_score):
    print("m", missing_score)
    print("c", correctness_score)

    return round((missing_score + correctness_score) / 2, 2)
