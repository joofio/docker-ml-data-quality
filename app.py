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
from fastapi.responses import JSONResponse

from sklearn.pipeline import Pipeline, make_pipeline
from functions_support import *


class Row(BaseModel):
    IDADE_MATERNA: Union[int, None] = np.nan
    GS: Union[str, None] = np.nan
    PESO_INICIAL: Union[float, None] = np.nan
    IMC: Union[float, None] = np.nan
    A_PARA: Union[float, None] = np.nan
    A_GESTA: Union[int, None] = np.nan
    EUTOCITO_ANTERIOR: Union[int, None] = np.nan
    TIPO_GRAVIDEZ: Union[str, None] = np.nan
    VIGIADA: Union[str, None] = np.nan
    NUMERO_CONSULTAS_PRE_NATAL: Union[int, None] = np.nan
    VIGIADA_CENTRO_SAUDE: Union[str, None] = np.nan
    VIGIADA_NESTE_HOSPITAL: Union[str, None] = np.nan
    ESTIMATIVA_PESO_ECO_30: Union[float, None] = np.nan
    APRESENTACAO_30: Union[str, None] = np.nan
    APRESENTACAO_ADMISSAO: Union[str, None] = np.nan
    IDADE_GESTACIONAL_ADMISSAO: Union[int, None] = np.nan
    TRAB_PARTO_ENTRADA_ESPONTANEO: Union[str, None] = np.nan
    TIPO_PARTO: Union[str, None] = np.nan
    APRESENTACAO_NO_PARTO: Union[str, None] = np.nan
    TRAB_PARTO_NO_PARTO: Union[str, None] = np.nan
    SEMANAS_GESTACAO_PARTO: Union[int, None] = np.nan
    GRUPO_ROBSON: Union[str, None] = np.nan

    IDENTIFICADOR: Union[str, None] = np.nan
    DATA_PARTO: Union[str, None] = np.nan
    PESO_ADMISSAO_INTERNAMENTO: Union[str, None] = np.nan
    CIGARROS: Union[str, None] = np.nan
    ALCOOL: Union[str, None] = np.nan
    ESTUPEFACIENTES: Union[str, None] = np.nan
    VENTOSAS_ANTERIOR: Union[float, None] = np.nan
    FORCEPS_ANTERIOR: Union[float, None] = np.nan
    CESARIANAS_ANTERIOR: Union[float, None] = np.nan
    CESARIANAS_MOTIVO_ANTERIOR: Union[str, None] = np.nan
    VIGIADA_HOSPITAL: Union[str, None] = np.nan
    VIGIADA_PARICULAR: Union[str, None] = np.nan
    ESTIMATIVA_PESO_ECO_24: Union[float, None] = np.nan
    ESTIMATIVA_PESO_ECO_25: Union[float, None] = np.nan
    ESTIMATIVA_PESO_ECO_26: Union[float, None] = np.nan
    ESTIMATIVA_PESO_ECO_27: Union[float, None] = np.nan
    ESTIMATIVA_PESO_ECO_28: Union[float, None] = np.nan
    ESTIMATIVA_PESO_ECO_29: Union[float, None] = np.nan
    ESTIMATIVA_PESO_ECO_31: Union[float, None] = np.nan
    ESTIMATIVA_PESO_ECO_32: Union[float, None] = np.nan
    ESTIMATIVA_PESO_ECO_33: Union[float, None] = np.nan
    ESTIMATIVA_PESO_ECO_34: Union[float, None] = np.nan
    ESTIMATIVA_PESO_ECO_35: Union[float, None] = np.nan
    ESTIMATIVA_PESO_ECO_36: Union[float, None] = np.nan
    ESTIMATIVA_PESO_ECO_37: Union[float, None] = np.nan
    ESTIMATIVA_PESO_ECO_38: Union[float, None] = np.nan
    ESTIMATIVA_PESO_ECO_39: Union[float, None] = np.nan
    ESTIMATIVA_PESO_ECO_40: Union[float, None] = np.nan
    ESTIMATIVA_PESO_ECO_41: Union[float, None] = np.nan
    ESTIMATIVA_PESO_ECO_42: Union[float, None] = np.nan
    APRESENTACAO_42: Union[str, None] = np.nan
    APRESENTACAO_41: Union[str, None] = np.nan
    APRESENTACAO_40: Union[str, None] = np.nan
    APRESENTACAO_39: Union[str, None] = np.nan
    APRESENTACAO_38: Union[str, None] = np.nan
    APRESENTACAO_37: Union[str, None] = np.nan
    APRESENTACAO_36: Union[str, None] = np.nan
    APRESENTACAO_35: Union[str, None] = np.nan
    APRESENTACAO_34: Union[str, None] = np.nan
    APRESENTACAO_33: Union[str, None] = np.nan
    APRESENTACAO_32: Union[str, None] = np.nan
    APRESENTACAO_31: Union[str, None] = np.nan
    APRESENTACAO_29: Union[str, None] = np.nan
    APRESENTACAO_28: Union[str, None] = np.nan
    APRESENTACAO_27: Union[str, None] = np.nan
    APRESENTACAO_26: Union[str, None] = np.nan
    APRESENTACAO_25: Union[str, None] = np.nan
    APRESENTACAO_24: Union[str, None] = np.nan
    G_TERAPEUTICA: Union[str, None] = np.nan
    NUM_RN: Union[str, None] = np.nan
    E_ALT_UT: Union[str, None] = np.nan
    BACIA: Union[str, None] = np.nan
    BISHOP_SCORE: Union[int, None] = np.nan
    BISHOP_CONSISTENCIA: Union[int, None] = np.nan
    BISHOP_DESCIDA: Union[int, None] = np.nan
    BISHOP_DILATACAO: Union[int, None] = np.nan
    BISHOP_EXTINCAO: Union[int, None] = np.nan
    BISHOP_POSICAO: Union[int, None] = np.nan
    TRAB_PARTO_ENTRADA_INDUZIDO: Union[str, None] = np.nan
    RPM: Union[str, None] = np.nan
    HIPERTENSAO_CRONICA: Union[str, None] = np.nan
    HIPERTENSAO_GESTACIONAL: Union[str, None] = np.nan
    HIPERTENSAO_PRE_ECLAMPSIA: Union[str, None] = np.nan
    DIABETES_GESTACIONAL: Union[str, None] = np.nan
    DIABETES_GESTACIONAL_DIETA: Union[str, None] = np.nan
    DIABETES_GESTACIONAL_INSULINA: Union[str, None] = np.nan
    DIABETES_GESTACIONAL_ANTIBIO: Union[str, None] = np.nan
    DIABETES_MATERNA: Union[str, None] = np.nan
    DIABETES_TIPO1: Union[str, None] = np.nan
    DIABETES_TIPO2: Union[str, None] = np.nan
    HEMATOLOGICA: Union[str, None] = np.nan
    RESPIRATORIA: Union[str, None] = np.nan
    CEREBRAL: Union[str, None] = np.nan
    CARDIACA: Union[str, None] = np.nan


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/quality_check")
async def get_predict_easy(row: Row):
    # print(row)
    fastapi_logger.info("called quality_check")
    fastapi_logger.info(row)

    missing_score, missing_dict = get_missing_score(row)
    correctness_score, correctness_dict = get_correctness_score(row, model)
    iqr_score, iqr_dict = get_iqr_score(row)  # ??
    expectations_score, expectations_dict = get_expecations_score(row)
    # print(missing_score)
    # print(correctness_score)
    # print(iqr_dict)
    # print(missing_dict)
    # print(correctness_dict)
    # print(expectations_dict)
    missing_df = pd.DataFrame(missing_dict, index=[0])
    correctness_df = pd.DataFrame(correctness_dict, index=[0])
    iqr_df = pd.DataFrame(iqr_dict, index=[0])
    expectations_df = pd.DataFrame(expectations_dict, index=[0])
    result_df = pd.concat([missing_df, correctness_df, iqr_df, expectations_df])
    result_df.index = ["missing", "correctness", "iqr", "expectations"]
    result_df.replace(np.nan, None, inplace=True)
    final_score = calculate_score(
        missing_score, correctness_score, iqr_score, expectations_score
    )
    final_dict = result_df.to_dict()
    final_dict["score"] = final_score
    # print(final_dict)
    return JSONResponse(content=final_dict)
