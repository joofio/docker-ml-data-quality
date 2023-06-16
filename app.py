from fastapi import FastAPI
from typing import Dict, Any
import datetime
from fastapi.logger import logger as fastapi_logger
import pandas as pd
import numpy as np
from fastapi.responses import JSONResponse
from functions_support import (
    get_correctness_score,
    get_missing_score,
    get_iqr_score,
    get_expecations_score,
    get_outlier_elliptic_score,
    get_outlier_local_outlier_factor_score,
    calculate_score,
    gritbot_decision,
    COLS_TO_ADD,
    GIT_COMMIT,
    model,
    make_decisions,
)


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

    df = pd.DataFrame(row, index=[0])
    # df = df.reindex(columns=COLS_TO_ADD)
    df, _ = df.align(pd.DataFrame(columns=COLS_TO_ADD))

    # df = df.fillna(value=np.nan)
    # print(df)
    missing_score, missing_dict = get_missing_score(df.to_dict())
    correctness_score, correctness_dict = get_correctness_score(df, model)
    iqr_score, iqr_dict = get_iqr_score(df)  # ??
    expectations_score, expectations_dict, statistics = get_expecations_score(df)
    outlier_elliptic_score = get_outlier_elliptic_score(df)
    outlier_local_outlier_factor_score = get_outlier_local_outlier_factor_score(df)
    # print(outlier_elliptic_score, outlier_local_outlier_factor_score)
    gritbot_score = gritbot_decision(df)

    missing_df = pd.DataFrame(missing_dict, index=[0])
    # print(missing_df)
    correctness_df = pd.DataFrame(correctness_dict, index=[0])
    iqr_df = pd.DataFrame(iqr_dict, index=[0])
    expectations_df = pd.DataFrame(expectations_dict)
    print(expectations_df)
    if len(expectations_df) > 0:
        expectations_df = expectations_df.loc[["count", "text"], :]
    else:
        expectations_df = pd.DataFrame({c: [np.nan, np.nan] for c in df.columns})
    print(expectations_df)
    # print(expectations_dict)
    result_df = pd.concat([missing_df, correctness_df, iqr_df, expectations_df])
    result_df.index = ["missing", "correctness", "iqr", "expectations", "rule"]
    result_df.replace(np.nan, None, inplace=True)
    # print(result_df)
    result_df.to_csv("sss.csv")
    final_score = calculate_score(
        missing_score, correctness_score, iqr_score, expectations_score
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
        "row": [
            {
                "row_score": final_score,
                "lof_outlier": "NOK"
                if int(outlier_local_outlier_factor_score[0]) < 0
                else "OK",
                "elliptic_outlier": int(outlier_elliptic_score[0]),
            }
        ],
    }
    # print(final_dict)
    return JSONResponse(content=final_dict)
