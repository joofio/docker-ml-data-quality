from fastapi import FastAPI
from typing import Dict, Any
import datetime
from fastapi.logger import logger as fastapi_logger
from logging.handlers import RotatingFileHandler
import logging
import pandas as pd
import numpy as np
from fhir.resources.bundle import Bundle
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from functions_support import (
    get_correctness_score,
    get_missing_score,
    get_iqr_score,
    get_expecations_score,
    get_outlier_elliptic_score,
    get_outlier_local_outlier_factor_score,
    calculate_score,
    gritbot_decision,
    extract_from_message,
    transform_to_fhir,
    COLS_TO_ADD,
    GIT_COMMIT,
    model,
    make_decisions,
    quality_score
)
from os.path import exists
from os import makedirs

formatter = logging.Formatter(
    "[%(asctime)s.%(msecs)03d] %(levelname)s [%(thread)d] - %(message)s",
    "%Y-%m-%d %H:%M:%S",
)
handler = RotatingFileHandler("logfile.log", backupCount=0)
logging.getLogger("fastapi")
fastapi_logger.addHandler(handler)
handler.setFormatter(formatter)
fastapi_logger.setLevel(logging.INFO)

fastapi_logger.info("****************** Starting Server *****************")
fastapi_logger.info("****************** Creating Folders *****************")

log_path = "log_folder"

# Check whether the specified path exists or not
isExist = exists(log_path)

if not isExist:

    # Create a new directory because it does not exist
    makedirs(log_path)
    fastapi_logger.info("The new directory is created!")

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post(
    "/quality_check",
)
async def get_quality_score(row: Dict[Any, Any]):
    # print(row)
    fastapi_logger.info("called quality_check")
    fastapi_logger.info(row)

    ndf = pd.DataFrame(row, index=[0])
    final_score,decisions,lof_score,ee_score,missing_score,correctness_score,iqr_score,expectations_score=quality_score(ndf)
    print(final_score,decisions,lof_score,ee_score,missing_score,correctness_score,iqr_score,expectations_score)
    meta = {
        "commit": GIT_COMMIT,
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
        "columns": decisions,
        "scores": {
            "final_score": final_score,
            "missing_score": missing_score,
            "correctness_score": correctness_score,
            "iqr_score": iqr_score,
            "expectations_score": expectations_score,
            "lof_outlier": lof_score,
            "elliptic_outlier": ee_score,
        },
    }

    print(final_dict)
    return JSONResponse(content=final_dict)
    #    missing_score, correctness_score, iqr_score, expectations_score


@app.post(
    "/fhir/r5/Bundle/$quality_check",
)
async def get_quality_score_fhir(row: Dict[Any, Any]):
    # print(row)
    try:
        bundle = Bundle.parse_obj(row)
    except Exception as err:
        # print(err)
        # print(err.args)
        print(err.__dict__)
        return {"results": "error", "explanation": repr(err)}

    fastapi_logger.info("called quality_check_fhir")
    fastapi_logger.info(row)
    ndf = extract_from_message(bundle)


    final_score,decisions,lof_score,ee_score,missing_score,correctness_score,iqr_score,expectations_score=quality_score(ndf)
    meta = {
        "commit": GIT_COMMIT,
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
        "columns": decisions,
        "scores": {
            "final_score": final_score,
            "missing_score": missing_score,
            "correctness_score": correctness_score,
            "iqr_score": iqr_score,
            "expectations_score": expectations_score,
            "lof_outlier": lof_score,
            "elliptic_outlier": ee_score,
        },
    }

    print(final_dict)
    fhir_response = transform_to_fhir(final_dict)
    print(fhir_response)
    return JSONResponse(content=jsonable_encoder(fhir_response))
    #    missing_score, correctness_score, iqr_score, expectations_score
