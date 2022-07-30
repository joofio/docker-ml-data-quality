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


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/quality_check")
async def get_predict_easy(row: Row):
    print(row)
    fastapi_logger.info("called quality_check")
    fastapi_logger.info(row)

    missing_score = get_missing_score(row)
    correctness_score = get_correctness_score(row, model)
    iqr_score = get_iqr_score(row)  # ??
    # print(missing_score)
    # print(correctness_score)
    final_score = calculate_score(missing_score, correctness_score, iqr_score)

    return final_score
