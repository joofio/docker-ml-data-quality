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
    # print(ndf)
    df = ndf.reindex(columns=COLS_TO_ADD)

    # print(df)
    # df, _ = df.align(pd.DataFrame(columns=COLS_TO_ADD))

    # df = df.fillna(value=np.nan)
    # print(df.to_dict("records"))
    missing_score, missing_dict = get_missing_score(df.to_dict("records")[0])
    correctness_score, correctness_dict = get_correctness_score(df, model)
    iqr_score, iqr_dict = get_iqr_score(df)  # ??
    expectations_score, expectations_dict, statistics = get_expecations_score(df)
    outlier_elliptic_score = get_outlier_elliptic_score(df)
    outlier_local_outlier_factor_score = get_outlier_local_outlier_factor_score(df)
    # print(outlier_elliptic_score, outlier_local_outlier_factor_score)
    # print(df)
    gritbot_score = gritbot_decision(df)

    missing_df = pd.DataFrame(missing_dict, index=[0])
    # print(missing_df)
    correctness_df = pd.DataFrame(correctness_dict, index=[0])
    iqr_df = pd.DataFrame(iqr_dict, index=[0])
    expectations_df = pd.DataFrame(expectations_dict)
    #  print(expectations_df)
    if len(expectations_df) > 0:
        expectations_df = expectations_df.loc[["count", "text"], :]
    else:
        expectations_df = pd.DataFrame({c: [np.nan, np.nan] for c in df.columns})
    # print(expectations_df)
    # print(expectations_dict)
    result_df = pd.concat([missing_df, correctness_df, iqr_df, expectations_df])
    result_df.index = ["missing", "correctness", "iqr", "expectations", "rule"]
    result_df.replace(np.nan, None, inplace=True)
    # print(result_df)
    # result_df.to_csv("sss.csv")

    lof_score = (
        0
        if int(outlier_local_outlier_factor_score[0]) < 0
        else int(outlier_local_outlier_factor_score[0])
    )
    ee_score = (
        0 if int(outlier_elliptic_score[0]) > 0 else 1
    )  # original Predict labels (1 inlier, -1 outlier) of X according to fitted model.
    #   print(lof_score, outlier_elliptic_score[0])
    final_score = calculate_score(
        missing_score,
        correctness_score,
        iqr_score,
        expectations_score,
        lof_score,
        ee_score,
    )
    # print(datetime.datetime.now())
    decisions = make_decisions(result_df)
    decisions["correctness_cols"]["gritbot"] = gritbot_score
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

    # ndf = pd.DataFrame(row, index=[0])
    # print(ndf)
    df = ndf.reindex(columns=COLS_TO_ADD)

    # print(df)
    # df, _ = df.align(pd.DataFrame(columns=COLS_TO_ADD))

    # df = df.fillna(value=np.nan)
    # print(df.to_dict("records"))
    missing_score, missing_dict = get_missing_score(df.to_dict("records")[0])
    correctness_score, correctness_dict = get_correctness_score(df, model)
    iqr_score, iqr_dict = get_iqr_score(df)  # ??
    expectations_score, expectations_dict, statistics = get_expecations_score(df)
    outlier_elliptic_score = get_outlier_elliptic_score(df)
    outlier_local_outlier_factor_score = get_outlier_local_outlier_factor_score(df)
    # print(outlier_elliptic_score, outlier_local_outlier_factor_score)
    # print(df)
    gritbot_score = gritbot_decision(df)

    missing_df = pd.DataFrame(missing_dict, index=[0])
    # print(missing_df)
    correctness_df = pd.DataFrame(correctness_dict, index=[0])
    iqr_df = pd.DataFrame(iqr_dict, index=[0])
    expectations_df = pd.DataFrame(expectations_dict)
    #  print(expectations_df)
    if len(expectations_df) > 0:
        expectations_df = expectations_df.loc[["count", "text"], :]
    else:
        expectations_df = pd.DataFrame({c: [np.nan, np.nan] for c in df.columns})
    # print(expectations_df)
    # print(expectations_dict)
    result_df = pd.concat([missing_df, correctness_df, iqr_df, expectations_df])
    result_df.index = ["missing", "correctness", "iqr", "expectations", "rule"]
    result_df.replace(np.nan, None, inplace=True)
    # print(result_df)
    # result_df.to_csv("sss.csv")

    lof_score = (
        0
        if int(outlier_local_outlier_factor_score[0]) < 0
        else int(outlier_local_outlier_factor_score[0])
    )
    ee_score = (
        0 if int(outlier_elliptic_score[0]) > 0 else 1
    )  # original Predict labels (1 inlier, -1 outlier) of X according to fitted model.
    #   print(lof_score, outlier_elliptic_score[0])
    final_score = calculate_score(
        missing_score,
        correctness_score,
        iqr_score,
        expectations_score,
        lof_score,
        ee_score,
    )
    # print(datetime.datetime.now())
    decisions = make_decisions(result_df)
    decisions["correctness_cols"]["gritbot"] = gritbot_score
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
