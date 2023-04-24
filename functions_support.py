from pgmpy.readwrite import XMLBIFReader
import json
import joblib
import pandas as pd
import numpy as np
from pgmpy.inference import VariableElimination
import scipy.stats as ss
import great_expectations as ge
from os import getenv
import pickle
from preprocessing import preprocess_df
from io import StringIO
import sys


GIT_COMMIT = getenv("GIT_COMMIT", None)


reader = XMLBIFReader("model_2.xml")
model = reader.get_model()
pl = joblib.load("pipeline.sav")

ee = joblib.load("EllipticEnvelope.sav")
lof = joblib.load("LocalOutlierFactor.sav")
my_expectation_suite = json.load(open("my_expectation_file.json"))
outliers_model = joblib.load("gritbot.sav")


# Opening JSON file
with open("null_dict.json") as json_file:
    null_dict = json.load(json_file)

with open("iqr_dict.json") as json_file:
    iqr_dict = json.load(json_file)

with open("standardizer.pickle", "rb") as handle:
    standardizer = pickle.load(handle)


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

outlier_cols = [
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


def get_iqr_score(opt):
    score = 0
    result_dict = {}
    # opt = jsonable_encoder(row)
    # print(opt)
    for c in int_cols:
        # print(c, iqr_dict[c], opt[c])
        x = opt[c].values[0]
        iqr = iqr_dict[c]["iqr"]
        q3 = iqr_dict[c]["q3"]
        q1 = iqr_dict[c]["q1"]

        if not x == np.nan:
            ll_threshold = q1 - iqr * 3
            uu_threshold = q3 + iqr * 3
            l_threshold = q1 - iqr * 1.5
            u_threshold = q3 + iqr * 1.5
            # print(x)
            if x < ll_threshold or x > uu_threshold:
                print("out of range")
                score += 2
                result_dict[c] = 2
            elif x < l_threshold or x > u_threshold:
                print("near range")
                score += 1
                result_dict[c] = 1

            else:
                score += 0
                result_dict[c] = 0

    return score / len(int_cols), result_dict


def get_missing_score(opt):
    score = 0
    null_count = 0
    result_dict = {}
    # opt = jsonable_encoder(row)
    for c in cols:

        if pd.isnull(opt[c]):
            print("MISSSING: ", c, null_dict[c])
            score += null_dict[c] / 100
            null_count += 1
            result_dict[c] = null_dict[c] / 100
        else:
            result_dict[c] = 0
    return score / (len(cols) - null_count), result_dict


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


def parse_ge_result(re):
    res_dict = {
        "expectation_type": [],
        "cols": [],
        "unexpected_count": [],
        "missing_percent": [],
        "missing_count": [],
        "unexpected_percent": [],
        "unexpected_percent_total": [],
        "unexpected_percent_nonmissing": [],
        "success": [],
    }
    rr = re.to_json_dict()
    for c in rr["results"]:
        exp_type = c["expectation_config"]["expectation_type"]

        res_dict["expectation_type"].append(exp_type)
        if "pair" in exp_type:
            cols = [
                c["expectation_config"]["kwargs"].get("column_A"),
                c["expectation_config"]["kwargs"].get("column_B"),
            ]
        else:
            cols = [c["expectation_config"]["kwargs"].get("column")]
        res_dict["cols"].append(",".join(cols))
        #  print(exp_type,",".join(cols))
        res = c["result"]
        #  print(res)
        res_dict["unexpected_count"].append(res.get("unexpected_count"))
        res_dict["missing_percent"].append(res.get("missing_percent"))
        res_dict["missing_count"].append(res.get("missing_count"))
        res_dict["unexpected_percent"].append(res.get("unexpected_percent"))
        res_dict["unexpected_percent_total"].append(res.get("unexpected_percent_total"))
        res_dict["unexpected_percent_nonmissing"].append(
            res.get("unexpected_percent_nonmissing")
        )
        res_dict["success"].append(c["success"])
    results_df = pd.DataFrame.from_dict(res_dict)
    return results_df


def get_expecations_score(df):
    # opt = jsonable_encoder(row)
    result_dict = {}
    # df = pd.DataFrame(opt, index=[0])
    my_df = ge.from_pandas(df, expectation_suite=my_expectation_suite)
    result = my_df.validate()
    result_df = parse_ge_result(result)
    print(result_df)
    issues = result_df[result_df["success"] is False]
    for idx, row in issues.iterrows():
        result_dict[row["cols"]] = {
            "count": row["unexpected_count"],
            "rule": row["expectation_type"],
            "percentage": round(row["unexpected_percent"], 2),
        }
    # print(idx,row)
    score = len(issues) / len(result_df)
    print(result_dict)
    return score, result_dict


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


def standardize_null(x, mapping):
    if x in mapping.keys():
        return mapping[x]
    if pd.isna(x):
        return np.nan
    return x


def get_correctness_score(df, model):
    inference = VariableElimination(model)

    score = 0
    # opt = jsonable_encoder(row)
    result_dict = {}
    # df = pd.DataFrame(opt, index=[0])

    # print(df)
    df[cat_cols] = df[cat_cols].astype(str)
    # df.to_csv("debug.csv")
    for col in df.columns:
        df[col] = df[col].apply(standardize_null, mapping=standardizer)
    for i in cat_cols:
        df[i].replace({"None": np.nan}, inplace=True)
        df[i] = df[i].astype(str)
    for c in int_cols:
        df[c].replace({"None": np.nan}, inplace=True)
        df[c] = np.where(pd.isnull(df[c]), df[c], df[c].astype(float))
    for c in ord_cols:
        df[c].replace({"None": np.nan}, inplace=True)

        df[c] = np.where(pd.isnull(df[c]), df[c], df[c].astype("Int64"))
    x_treated = pl.transform(df)
    # print(opt)
    # print(x_treated)
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
        matchs = get_score_for_not_match(query, c, truth)
        score += matchs
        result_dict[c] = matchs
    return score / len(network_cols), result_dict


def calculate_score(missing_score, correctness_score, iqr_score, expectation_score):
    print("m", missing_score)
    print("c", correctness_score)
    print("iqr", iqr_score)
    print("expectations", expectation_score)

    return round((missing_score + correctness_score + iqr_score) / 3, 2)


def get_outlier_elliptic_score(df):
    # print(df)
    df[cat_cols] = df[cat_cols].astype(str)
    # print(df.to_dict())
    x_treated = pl.transform(df[outlier_cols])
    # print(opt)
    return ee.predict(x_treated)


def get_outlier_local_outlier_factor_score(df):
    # print(df)
    df[cat_cols] = df[cat_cols].astype(str)
    # print(df.to_dict())
    x_treated = pl.transform(df[outlier_cols])
    # print(opt)
    return lof.predict(x_treated)


def create_response_outlier(out):
    results = []
    for _, x in out[out["suspicious_value"] != {}].iterrows():
        susp = x["suspicious_value"]
        # print(susp)
        # print(susp["column"])
        cond = x["group_statistics"]
        # print(cond)
        results.append(
            f"""Suspicious Column: {susp["column"]}. Suspicious Value: {susp["value"]}"""
        )
    return results


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def gritbot_decision(df):
    df_clean = preprocess_df(df, "CHSJ")
    df_clean["silo"] = "CHSJ"  # must be corrected
    new_outliers = outliers_model.predict(df_clean)
    if len(new_outliers) >= 1:
        with Capturing() as output:
            outliers_model.print_outliers(new_outliers)

        return "".join(output[3:])
    else:
        return "0"
