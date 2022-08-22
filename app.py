from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
import json
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
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
from typing import List


class Row(BaseModel):
    IDADE_MATERNA: Union[int, None] = None
    GS: Union[str, None] = None
    PESO_INICIAL: Union[float, None] = None
    IMC: Union[float, None] = None
    A_PARA: Union[float, None] = None
    A_GESTA: Union[int, None] = None
    EUTOCITO_ANTERIOR: Union[int, None] = None
    TIPO_GRAVIDEZ: Union[str, None] = None
    VIGIADA: Union[str, None] = None
    NUMERO_CONSULTAS_PRE_NATAL: Union[int, None] = None
    VIGIADA_CENTRO_SAUDE: Union[str, None] = None
    VIGIADA_NESTE_HOSPITAL: Union[str, None] = None
    ESTIMATIVA_PESO_ECO_30: Union[float, None] = None
    APRESENTACAO_30: Union[str, None] = None
    APRESENTACAO_ADMISSAO: Union[str, None] = None
    IDADE_GESTACIONAL_ADMISSAO: Union[int, None] = None
    TRAB_PARTO_ENTRADA_ESPONTANEO: Union[str, None] = None
    TIPO_PARTO: Union[str, None] = None
    APRESENTACAO_NO_PARTO: Union[str, None] = None
    TRAB_PARTO_NO_PARTO: Union[str, None] = None
    SEMANAS_GESTACAO_PARTO: Union[int, None] = None
    GRUPO_ROBSON: Union[str, None] = None
    IDENTIFICADOR: Union[str, None] = None
    DATA_PARTO: Union[str, None] = None
    PESO_ADMISSAO_INTERNAMENTO: Union[str, None] = None
    CIGARROS: Union[str, None] = None
    ALCOOL: Union[str, None] = None
    ESTUPEFACIENTES: Union[str, None] = None
    VENTOSAS_ANTERIOR: Union[float, None] = None
    FORCEPS_ANTERIOR: Union[float, None] = None
    CESARIANAS_ANTERIOR: Union[float, None] = None
    CESARIANAS_MOTIVO_ANTERIOR: Union[str, None] = None
    VIGIADA_HOSPITAL: Union[str, None] = None
    VIGIADA_PARICULAR: Union[str, None] = None
    ESTIMATIVA_PESO_ECO_24: Union[float, None] = None
    ESTIMATIVA_PESO_ECO_25: Union[float, None] = None
    ESTIMATIVA_PESO_ECO_26: Union[float, None] = None
    ESTIMATIVA_PESO_ECO_27: Union[float, None] = None
    ESTIMATIVA_PESO_ECO_28: Union[float, None] = None
    ESTIMATIVA_PESO_ECO_29: Union[float, None] = None
    ESTIMATIVA_PESO_ECO_31: Union[float, None] = None
    ESTIMATIVA_PESO_ECO_32: Union[float, None] = None
    ESTIMATIVA_PESO_ECO_33: Union[float, None] = None
    ESTIMATIVA_PESO_ECO_34: Union[float, None] = None
    ESTIMATIVA_PESO_ECO_35: Union[float, None] = None
    ESTIMATIVA_PESO_ECO_36: Union[float, None] = None
    ESTIMATIVA_PESO_ECO_37: Union[float, None] = None
    ESTIMATIVA_PESO_ECO_38: Union[float, None] = None
    ESTIMATIVA_PESO_ECO_39: Union[float, None] = None
    ESTIMATIVA_PESO_ECO_40: Union[float, None] = None
    ESTIMATIVA_PESO_ECO_41: Union[float, None] = None
    ESTIMATIVA_PESO_ECO_42: Union[float, None] = None
    APRESENTACAO_42: Union[str, None] = None
    APRESENTACAO_41: Union[str, None] = None
    APRESENTACAO_40: Union[str, None] = None
    APRESENTACAO_39: Union[str, None] = None
    APRESENTACAO_38: Union[str, None] = None
    APRESENTACAO_37: Union[str, None] = None
    APRESENTACAO_36: Union[str, None] = None
    APRESENTACAO_35: Union[str, None] = None
    APRESENTACAO_34: Union[str, None] = None
    APRESENTACAO_33: Union[str, None] = None
    APRESENTACAO_32: Union[str, None] = None
    APRESENTACAO_31: Union[str, None] = None
    APRESENTACAO_29: Union[str, None] = None
    APRESENTACAO_28: Union[str, None] = None
    APRESENTACAO_27: Union[str, None] = None
    APRESENTACAO_26: Union[str, None] = None
    APRESENTACAO_25: Union[str, None] = None
    APRESENTACAO_24: Union[str, None] = None
    G_TERAPEUTICA: Union[str, None] = None
    NUM_RN: Union[str, None] = None
    E_ALT_UT: Union[str, None] = None
    BACIA: Union[str, None] = None
    BISHOP_SCORE: Union[int, None] = None
    BISHOP_CONSISTENCIA: Union[int, None] = None
    BISHOP_DESCIDA: Union[int, None] = None
    BISHOP_DILATACAO: Union[int, None] = None
    BISHOP_EXTINCAO: Union[int, None] = None
    BISHOP_POSICAO: Union[int, None] = None
    TRAB_PARTO_ENTRADA_INDUZIDO: Union[str, None] = None
    RPM: Union[str, None] = None
    HIPERTENSAO_CRONICA: Union[str, None] = None
    HIPERTENSAO_GESTACIONAL: Union[str, None] = None
    HIPERTENSAO_PRE_ECLAMPSIA: Union[str, None] = None
    DIABETES_GESTACIONAL: Union[str, None] = None
    DIABETES_GESTACIONAL_DIETA: Union[str, None] = None
    DIABETES_GESTACIONAL_INSULINA: Union[str, None] = None
    DIABETES_GESTACIONAL_ANTIBIO: Union[str, None] = None
    DIABETES_MATERNA: Union[str, None] = None
    DIABETES_TIPO1: Union[str, None] = None
    DIABETES_TIPO2: Union[str, None] = None
    HEMATOLOGICA: Union[str, None] = None
    RESPIRATORIA: Union[str, None] = None
    CEREBRAL: Union[str, None] = None
    CARDIACA: Union[str, None] = None

    class Config:
        schema_extra = {
            "GS": "O,RH_POSITIVO",
            "IDADE_MATERNA": 25,
            "PESO_INICIAL": 90.0,
            "IMC": 34.3,
            "A_PARA": 3.0,
            "A_GESTA": 4,
            "TIPO_GRAVIDEZ": "ESPONTANEA",
            "VIGIADA": "S",
            "NUMERO_CONSULTAS_PRE_NATAL": 6.0,
            "VIGIADA_NESTE_HOSPITAL": "S",
            "IDADE_GESTACIONAL_ADMISSAO": 37.0,
            "TRAB_PARTO_ENTRADA_ESPONTANEO": "S",
            "TIPO_PARTO": "Cesariana",
            "APRESENTACAO_NO_PARTO": "Cefálica de vértice",
            "TRAB_PARTO_NO_PARTO": "Espontâneo",
            "SEMANAS_GESTACAO_PARTO": 37.0,
            "GRUPO_ROBSON": "10",
        }


class AlgorithmInfo(BaseModel):
    model: str
    version: float


class Result(BaseModel):
    correctness: Union[float, None] = Field(
        default=None,
        description="The percentage of how wrong the field is",
        title="Correctness of the field",
    )
    iqr: Union[float, None] = Field(
        default=None,
        description="The score of how inside the IQR the field is",
        title="Correctness of the field via IQR",
    )
    missing: Union[float, None] = Field(
        default=None,
        description="The percentage of how strange the missing of the field is",
        title="Missing of the field",
    )
    expectations: Union[float, None] = Field(
        default=None,
        description="If the field is out of the expectations",
        title="Expectations of the field",
    )
    column_score: Union[float, None] = Field(
        default=None,
        description="The overall percentage of how wrong the field is",
        title="Score of the column",
    )


class Meta(BaseModel):
    """
    Meta information for the model result
    """

    Correctness: AlgorithmInfo
    IQR: AlgorithmInfo
    Missing: AlgorithmInfo
    expecations: AlgorithmInfo
    timestamp: datetime.datetime


class Response(BaseModel):
    meta: Meta
    results: List[Result]
    row_score: float


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/quality_check", response_model=Response)
async def get_quality_score(row: Row):
    # print(row)
    fastapi_logger.info("called quality_check")
    fastapi_logger.info(row)

    opt = jsonable_encoder(row)
    df = pd.DataFrame(opt, index=[0])
    df = df.fillna(value=np.nan)

    missing_score, missing_dict = get_missing_score(opt)
    correctness_score, correctness_dict = get_correctness_score(df, model)
    iqr_score, iqr_dict = get_iqr_score(df)  # ??
    expectations_score, expectations_dict = get_expecations_score(df)
    outlier_elliptic_score = get_outlier_elliptic_score(df)
    outlier_local_outlier_factor_score = get_outlier_local_outlier_factor_score(df)
    print(outlier_elliptic_score, outlier_local_outlier_factor_score)
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
    # print(datetime.datetime.now())
    meta = {
        "IQR": {"model": "z-score", "version": 0.1},
        "Missing": {"model": "traindata", "version": 0.1},
        "Correctness": {"model": "bayes", "version": 0.1},
        "expecations": {"model": "human", "version": 0.1},
        "lof": {"model": "lof", "version": 0.1},
        "elliptic": {"model": "elliptic", "version": 0.1},
        "timestamp": datetime.datetime.now().strftime("%Y%m%dT%H%M%S"),
    }
    final_dict = {
        "meta": meta,
        "columns": result_df.to_dict(),
        "row": [
            {
                "row_score": final_score,
                "lof_outlier": int(outlier_local_outlier_factor_score[0]),
                "elliptic_outlier": int(outlier_elliptic_score[0]),
            }
        ],
    }
    # print(final_dict)
    return JSONResponse(content=final_dict)
