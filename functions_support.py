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
from pgmpy.inference import VariableElimination, BeliefPropagation
from sklearn.pipeline import Pipeline, make_pipeline
import scipy.stats as ss

reader = XMLBIFReader("first_model.xml")
model = reader.get_model()
pl = joblib.load("pipeline.sav")

# Opening JSON file
with open("null_dict.json") as json_file:
    null_dict = json.load(json_file)

with open("iqr_dict.json") as json_file:
    iqr_dict = json.load(json_file)

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
    "IDENTIFICADOR",
    "DATA_PARTO",
    "IDADE_MATERNA",
    "GS",
    "PESO_INICIAL",
    "PESO_ADMISSAO_INTERNAMENTO",
    "IMC",
    "CIGARROS",
    "ALCOOL",
    "ESTUPEFACIENTES",
    "A_PARA",
    "A_GESTA",
    "EUTOCITO_ANTERIOR",
    "VENTOSAS_ANTERIOR",
    "FORCEPS_ANTERIOR",
    "CESARIANAS_ANTERIOR",
    "CESARIANAS_MOTIVO_ANTERIOR",
    "TIPO_GRAVIDEZ",
    "VIGIADA",
    "NUMERO_CONSULTAS_PRE_NATAL",
    "VIGIADA_HOSPITAL",
    "VIGIADA_PARICULAR",
    "VIGIADA_CENTRO_SAUDE",
    "VIGIADA_NESTE_HOSPITAL",
    "ESTIMATIVA_PESO_ECO_24",
    "ESTIMATIVA_PESO_ECO_25",
    "ESTIMATIVA_PESO_ECO_26",
    "ESTIMATIVA_PESO_ECO_27",
    "ESTIMATIVA_PESO_ECO_28",
    "ESTIMATIVA_PESO_ECO_29",
    "ESTIMATIVA_PESO_ECO_30",
    "ESTIMATIVA_PESO_ECO_31",
    "ESTIMATIVA_PESO_ECO_32",
    "ESTIMATIVA_PESO_ECO_33",
    "ESTIMATIVA_PESO_ECO_34",
    "ESTIMATIVA_PESO_ECO_35",
    "ESTIMATIVA_PESO_ECO_36",
    "ESTIMATIVA_PESO_ECO_37",
    "ESTIMATIVA_PESO_ECO_38",
    "ESTIMATIVA_PESO_ECO_39",
    "ESTIMATIVA_PESO_ECO_40",
    "ESTIMATIVA_PESO_ECO_41",
    "ESTIMATIVA_PESO_ECO_42",
    "APRESENTACAO_42",
    "APRESENTACAO_41",
    "APRESENTACAO_40",
    "APRESENTACAO_39",
    "APRESENTACAO_38",
    "APRESENTACAO_37",
    "APRESENTACAO_36",
    "APRESENTACAO_35",
    "APRESENTACAO_34",
    "APRESENTACAO_33",
    "APRESENTACAO_32",
    "APRESENTACAO_31",
    "APRESENTACAO_30",
    "APRESENTACAO_29",
    "APRESENTACAO_28",
    "APRESENTACAO_27",
    "APRESENTACAO_26",
    "APRESENTACAO_25",
    "APRESENTACAO_24",
    "G_TERAPEUTICA",
    "NUM_RN",
    "E_ALT_UT",
    "BACIA",
    "APRESENTACAO_ADMISSAO",
    "BISHOP_SCORE",
    "BISHOP_CONSISTENCIA",
    "BISHOP_DESCIDA",
    "BISHOP_DILATACAO",
    "BISHOP_EXTINCAO",
    "BISHOP_POSICAO",
    "IDADE_GESTACIONAL_ADMISSAO",
    "TRAB_PARTO_ENTRADA_ESPONTANEO",
    "TRAB_PARTO_ENTRADA_INDUZIDO",
    "RPM",
    "HIPERTENSAO_CRONICA",
    "HIPERTENSAO_GESTACIONAL",
    "HIPERTENSAO_PRE_ECLAMPSIA",
    "DIABETES_GESTACIONAL",
    "DIABETES_GESTACIONAL_DIETA",
    "DIABETES_GESTACIONAL_INSULINA",
    "DIABETES_GESTACIONAL_ANTIBIO",
    "DIABETES_MATERNA",
    "DIABETES_TIPO1",
    "DIABETES_TIPO2",
    "HEMATOLOGICA",
    "RESPIRATORIA",
    "CEREBRAL",
    "CARDIACA",
    "TIPO_PARTO",
    "APRESENTACAO_NO_PARTO",
    "TRAB_PARTO_NO_PARTO",
    "SEMANAS_GESTACAO_PARTO",
    "GRUPO_ROBSON",
]

network_cols = [
    "IDADE_MATERNA",
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


def get_iqr_score(row):
    score = 0

    opt = jsonable_encoder(row)
    for c in int_cols:
        # print(c, iqr_dict[c], opt[c])
        x = opt[c]
        iqr = iqr_dict[c]["iqr"]
        q3 = iqr_dict[c]["q3"]
        q1 = iqr_dict[c]["q1"]

        if not x == np.nan:
            ll_threshold = q1 - iqr * 3
            uu_threshold = q3 + iqr * 3
            l_threshold = q1 - iqr * 1.5
            u_threshold = q3 + iqr * 1.5
            if x < ll_threshold or x > uu_threshold:
                print("out of range")
                score += 2
            elif x < l_threshold or x > u_threshold:
                print("near range")
                score += 1
            else:
                score += 0
    return score / len(int_cols)


def get_missing_score(row):
    score = 0
    null_count = 0
    opt = jsonable_encoder(row)
    for c in cols:

        if pd.isnull(opt[c]):
            print("MISSSING: ", c, null_dict[c])
            score += null_dict[c] / 100
            null_count += 1
    print(score, len(cols), null_count)
    return score / (len(cols) - null_count)


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


def get_score_for_not_match(query, varia, truth):
    probas = query.values
    rr = ss.rankdata(
        [-el for el in probas], method="max"
    )  # Negative so we make the reverse

    pred_idx = query.values.argmax()
    pred = query.state_names[varia][pred_idx]
    pred_proba = probas[pred_idx]
    pred_ranking = rr[pred_idx]

    true_idx = query.state_names[varia].index(truth)
    true_proba = probas[true_idx]
    true_ranking = rr[true_idx]
    states_nr = len(query.values)
    if pred_ranking != true_ranking:
        if true_ranking == states_nr:
            return 1 - true_proba
    return 0


def get_correctness_score(row, model):
    inference = VariableElimination(model)

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

        df_evidence = transfrom_array_to_df_onehot(pl, x_treated)[network_cols]
        df_evidence = df_evidence.astype(str)
        df_evidence.replace("\.", "_", regex=True, inplace=True)
        truth = df_evidence[c].values[0]

        df_evidence.drop(columns=[c], inplace=True)  # remove on pipeline for prod?
        evidence = df_evidence.to_dict("records").copy()[0]

        # print(evidence)
        query = inference.query(variables=[c], evidence=evidence, show_progress=False)
        pred = query.state_names[c][query.values.argmax()]
        score += get_score_for_not_match(query, c, truth)

    return score / len(network_cols)


def calculate_score(missing_score, correctness_score, iqr_score):
    print("m", missing_score)
    print("c", correctness_score)
    print("iqr", iqr_score)

    return round((missing_score + correctness_score + iqr_score) / 3, 2)
