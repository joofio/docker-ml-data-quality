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
from fhir.resources.bundle import Bundle
from fhir.resources.device import Device
from fhir.resources.observation import Observation
from fhir.resources.messageheader import MessageHeader

import uuid

FHIR_PUBLIC_LINK = "https://joofio.github.io/obs-cdss-fhir"

COLS_TO_ADD = ["silo",
    "IDENTIFICADOR",
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
    "DATA_PARTO",
    "PESO_ADMISSAO_INTERNAMENTO",
    "CIGARROS",
    "ALCOOL",
    "ESTUPEFACIENTES",
    "VENTOSAS_ANTERIOR",
    "FORCEPS_ANTERIOR",
    "CESARIANAS_ANTERIOR",
    "CESARIANAS_MOTIVO_ANTERIOR",
    "VIGIADA_HOSPITAL",
    "VIGIADA_PARICULAR",
    "ESTIMATIVA_PESO_ECO_24",
    "ESTIMATIVA_PESO_ECO_25",
    "ESTIMATIVA_PESO_ECO_26",
    "ESTIMATIVA_PESO_ECO_27",
    "ESTIMATIVA_PESO_ECO_28",
    "ESTIMATIVA_PESO_ECO_29",
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
    "BISHOP_SCORE",
    "BISHOP_CONSISTENCIA",
    "BISHOP_DESCIDA",
    "BISHOP_DILATACAO",
    "BISHOP_EXTINCAO",
    "BISHOP_POSICAO",
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
]
GIT_COMMIT = getenv("GIT_COMMIT", None)


reader = XMLBIFReader("model_total.xml")
model = reader.get_model()
pl = joblib.load("pipeline.sav")
outpl=joblib.load("outlier_pipeline.sav")
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
    "silo"
]

network_cols = ['IDADE_MATERNA',
 'GS',
 'PESO_INICIAL',
 'IMC',
 'A_PARA',
 'A_GESTA',
 'EUTOCITO_ANTERIOR',
 'VENTOSAS_ANTERIOR',
 'CESARIANAS_ANTERIOR',
 'TIPO_GRAVIDEZ',
 'VIGIADA',
 'NUMERO_CONSULTAS_PRE_NATAL',
 'VIGIADA_PARICULAR',
 'VIGIADA_CENTRO_SAUDE',
 'VIGIADA_NESTE_HOSPITAL',
 'APRESENTACAO_ADMISSAO',
 'IDADE_GESTACIONAL_ADMISSAO',
 'TRAB_PARTO_ENTRADA_ESPONTANEO',
 'TIPO_PARTO',
 'APRESENTACAO_NO_PARTO',
 'TRAB_PARTO_NO_PARTO',
 'SEMANAS_GESTACAO_PARTO',
 'GRUPO_ROBSON',
 'silo']

outlier_cols = ['IDADE_MATERNA', 'GS', 'PESO_INICIAL', 'IMC', 'A_PARA', 'A_GESTA',
       'EUTOCITO_ANTERIOR', 'VENTOSAS_ANTERIOR', 'CESARIANAS_ANTERIOR',
       'TIPO_GRAVIDEZ', 'VIGIADA', 'NUMERO_CONSULTAS_PRE_NATAL',
       'VIGIADA_PARICULAR', 'VIGIADA_CENTRO_SAUDE', 'VIGIADA_NESTE_HOSPITAL',
       'APRESENTACAO_ADMISSAO', 'IDADE_GESTACIONAL_ADMISSAO',
       'TRAB_PARTO_ENTRADA_ESPONTANEO', 'TIPO_PARTO', 'APRESENTACAO_NO_PARTO',
       'TRAB_PARTO_NO_PARTO', 'SEMANAS_GESTACAO_PARTO', 'GRUPO_ROBSON',
       'silo']


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
                #    print("out of range")
                score += 2
                result_dict[c] = 2
            elif x < l_threshold or x > u_threshold:
                #     print("near range")
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
    # print(opt)
    # opt = jsonable_encoder(row)
    for c in cols:
        # print(opt[c])
        #  print(pd.isnull(opt[c]))
        if pd.isnull(opt[c]):
            # print("MISSSING: ", c, null_dict[c])
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
        "text": [],
        "unexpected_count": [],
        "success": [],
    }
    rr = re.to_json_dict()
    # res_dict["statistics"] = rr["statistics"]
    for c in rr["results"]:
        exp_type = c["expectation_config"]["expectation_type"]
        # print(exp_type)
        res_dict["expectation_type"].append(exp_type)
        if "pair" in exp_type:
            cols = [
                c["expectation_config"]["kwargs"].get("column_A"),
                c["expectation_config"]["kwargs"].get("column_B"),
            ]

        else:
            cols = [c["expectation_config"]["kwargs"].get("column")]
        if "expect_column_values_to_be_between" in exp_type:
            text = (
                str(exp_type)
                + "->"
                + "["
                + str(c["expectation_config"]["kwargs"]["min_value"])
                + ","
                + str(c["expectation_config"]["kwargs"]["max_value"])
                + "]"
            )
        else:
            text = str(exp_type)
            # print(exp_type)
        res_dict["text"].append(text)

        res_dict["cols"].append(",".join(cols))
        #  print(exp_type, ",".join(cols))
        res = c["result"]
        #  print(res)
        res_dict["unexpected_count"].append(res.get("unexpected_count"))

        res_dict["success"].append(c["success"])
    # print(res_dict)
    results_df = pd.DataFrame.from_dict(res_dict)
    return results_df


def get_expecations_score(df):
    # opt = jsonable_encoder(row)
    result_dict = {}
    # df = pd.DataFrame(opt, index=[0])
    my_df = ge.from_pandas(df, expectation_suite=my_expectation_suite)
    result = my_df.validate()
    #  print(result)
    statistics = result["statistics"]
    result_df = parse_ge_result(result)

    #   print(result_df)
    issues = result_df[result_df["success"] == False]
    for idx, row in issues.iterrows():
        #  print("expectation", row)
        result_dict[row["cols"]] = {
            "count": row["unexpected_count"],
            "rule": row["expectation_type"],
            "text": row["text"]
            # "percentage": round(row["unexpected_percent"], 2),
        }
    # print(idx,row)
    score = len(issues) / len(result_df)
    #  print(result_dict)
    return score, result_dict, statistics


def get_score_for_not_match(query, varia, truth):
    probas = query.values
    rr = ss.rankdata(
        [-el for el in probas], method="max"
    )  # Negative so we make the reverse, ranks the data
    # print(probas, rr)
    # print("truth", truth)

    pred_idx = query.values.argmax()  # index do valor maximo
    pred = query.state_names[varia][pred_idx]  # name of the vaariable selected
    pred_proba = probas[pred_idx]  # probability of the selected value
    pred_ranking = rr[pred_idx]  # ranking of the selected (not always 1?)
    if str(pred_proba)=="nan":
        print(probas)
        print("eror on a column.....")
        print(
            "pred_idx",
            pred_idx,
            "pred",
            pred,
            "pred_proba",
            pred_proba,
            "pred_ranking",
            pred_ranking,
        )
        return 0
    true_idx = query.state_names[varia].index(truth)  # truth index
    true_proba = probas[true_idx]  # probability of truth
    true_ranking = rr[true_idx]
    states_nr = len(query.values)
    # print("true_idx", true_idx, "true_proba", true_proba, "true_ranking", true_ranking)
    if pred_ranking != true_ranking:
        if true_ranking == states_nr:
            # print(1 - true_proba)
            #  print(1 - pred_proba)
            return 1 - true_proba
    return 0


def standardize_null(x, mapping):
    # print("x", x)
    # print(mapping)
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
    df=df[network_cols]

    net_cat_cols=[col for col in cat_cols if col  in df.columns ]
    net_int_cols=[col for col in int_cols if col  in df.columns ]
    net_ord_cols=[col for col in ord_cols if col  in df.columns ]

    # print(df)
    df[net_cat_cols] = df[net_cat_cols].astype(str)
    # df.to_csv("debug.csv")
    for col in df.columns:
        #   print("col", df[col])
        df[col] = df[col].apply(standardize_null, mapping=standardizer)
    for i in net_cat_cols:
        df[i].replace({"None": np.nan}, inplace=True)
        df[i] = df[i].astype(str)
    for c in net_int_cols:
        df[c].replace({"None": np.nan}, inplace=True)
        df[c] = np.where(pd.isnull(df[c]), df[c], df[c].astype(float))
    for c in net_ord_cols:
        df[c].replace({"None": np.nan}, inplace=True)
        df[c] = np.where(pd.isnull(df[c]), df[c], df[c].astype("Int64"))
    #print(df)
    df.to_csv("tt.csv")

    x_treated = pl.transform(df)
    # print(opt)
    # print(x_treated)
    for c in network_cols:
        #print(c)

        df_evidence = transfrom_array_to_df_onehot(pl, x_treated)[network_cols]
        df_evidence = df_evidence.astype(str)
        df_evidence.replace("\.", "_", regex=True, inplace=True)
        truth = df_evidence[c].values[0]
        #   print("true", truth)
        df_evidence.drop(columns=[c], inplace=True)  # remove on pipeline for prod?
        evidence = df_evidence.to_dict("records").copy()[0]

        # print(evidence)
        query = inference.query(variables=[c], evidence=evidence, show_progress=False)
      #  pred = query.state_names[c][query.values.argmax()]
      #  print("pred", pred, "query", query)
        matchs = get_score_for_not_match(query, c, truth)
      #  print(matchs)
        score += matchs
        result_dict[c] = matchs
    return score / len(network_cols), result_dict


def calculate_score(
    missing_score, correctness_score, iqr_score, expectation_score, lof, outli
):
    #   print("m", missing_score)
    #   print("c", correctness_score)
    #   print("iqr", iqr_score)
    #   print("expectations", expectation_score)

    return round(
        (
            missing_score
            + correctness_score
            + iqr_score
            + expectation_score
            + lof
            + outli
        )
        / 6,
        2,
    )


def get_outlier_elliptic_score(df):
    #try:# print(df)
    for col in df.columns:
        #   print("col", df[col])
        df[col] = df[col].apply(standardize_null, mapping=standardizer)
    df[cat_cols] = df[cat_cols].astype(str)
    
    # print(df.to_dict())
    x_treated = outpl.transform(df[outlier_cols])
    # print(opt)
    return ee.predict(x_treated)
    #except:
       # print("error on get_outlier_elliptic_score")
     #   return [[0][0]]

def get_outlier_local_outlier_factor_score(df):
    for col in df.columns:
        #   print("col", df[col])
        df[col] = df[col].apply(standardize_null, mapping=standardizer)
    df[cat_cols] = df[cat_cols].astype(str)
    # print(df.to_dict())
    x_treated = outpl.transform(df[outlier_cols])
    # print(opt)
    return lof.predict(x_treated)
   # except:
   #     print("error on get_outlier_local_outlier_factor_score")

      #  return [[0][0]]

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


def gritbot_decision(df):
    df_clean = preprocess_df(df, "CHSJ")
    df_clean["silo"] = "CHSJ"  # must be corrected
    # df_clean.drop(columns=["IDENTIFICADOR"], inplace=True)
    grit_bot_score = []

    new_outliers = outliers_model.predict(df_clean)

    for idx, row in new_outliers.iterrows():
        if row.get("suspicious_value").get("column") is None:
            return grit_bot_score
        d = {}
        d["column"] = row.get("suspicious_value").get("column")
        d["group_statistics"] = row["group_statistics"]
        d["conditions"] = row["conditions"]
        d["outlier_score"] = row["outlier_score"]
        grit_bot_score.append(d)
    return grit_bot_score


def make_decisions(df):

    # set the threshold value
    missing_threshold = 0.5
    correctness_threshold = 0.5
    iqr_threshold = 1.5
    expectations_threshold = 0.5

    missing_assess = df.loc["missing", :].apply(
        lambda x: "OK" if x <= missing_threshold else "NOK" if pd.notnull(x) else np.nan
    )
    missing_cols = {
        "assessment": missing_assess.to_dict(),
        "values": df.loc["missing", :].to_dict(),
    }

    correct_assess = df.loc["correctness", :].apply(
        lambda x: "OK"
        if pd.notnull(x) and x <= correctness_threshold
        else "NOK"
        if pd.notnull(x)
        else np.nan
    )
    correctness_cols = {
        "assessment": correct_assess.dropna().to_dict(),
        "values": df.loc["correctness", :].dropna().to_dict(),
    }
    # print(correctness_cols)

    iqr_assess = df.loc["iqr", :].apply(
        lambda x: "OK"
        if pd.notnull(x) and x <= iqr_threshold
        else "NOK"
        if pd.notnull(x)
        else np.nan
    )
    iqr_cols = {
        "assessment": iqr_assess.dropna().to_dict(),
        "values": df.loc["iqr", :].dropna().to_dict(),
    }

    mask = df.loc["expectations", :] > expectations_threshold
    filtered_row = df.loc["rule", mask]
    # print(filtered_row)
    expectations_cols = filtered_row.to_dict()

    return {
        "missing_cols": missing_cols,
        "correctness_cols": {
            "bayes": correctness_cols,
            "gritbot": {},
            "iqr": iqr_cols,
            "expectations": expectations_cols,
        },
    }


def extract_from_message(bundle):
    df_dict = {}
    for ent in bundle.entry:
        # print(ent.resource)
        # print(ent.resource.resource_type)
        if ent.resource.resource_type == "Observation":
            col = ent.resource.code.coding[0].code
            for k, v in ent.resource.__dict__.items():
                # print(k, v)
                if k.startswith("value") and v is not None:
                    val = v
                    if k == "valueQuantity":
                        val = float(v.value)
            # val = ent.resource.code.coding[0].value
            print(col)
            print(val)
            df_dict[col] = val

    print(df_dict)
    return pd.DataFrame(df_dict, index=[0])


def transform_to_fhir(mydict):
    # print(mydict)
    ids = [str(uuid.uuid1()) for i in range(1, 10)]

    # current date and time
    # now = datetime.datetime.now()
    header_id = str(uuid.uuid1())

    header_dict = {
        "resourceType": "MessageHeader",
        "id": header_id,
        "eventCoding": {"code": "obs-dq"},
        "destination": [{"name": "Requester"}],
        "source": {"name": "DQ-system"},
    }
    msh = MessageHeader(**header_dict)

    mess_dict = {
        "resourceType": "Bundle",
        "type": "message",
        "meta": {
            "profile": [FHIR_PUBLIC_LINK + "/StructureDefinition/MessageForRequest"]
        },
        "id": str(uuid.uuid1()),
        "entry": [
            {
                "resource": msh.dict(),
                "fullUrl": "http://localhost:8080/fhir/MessageHeader/" + header_id,
            }
        ],
    }

    mlmodel = {
        "resourceType": "Device",
        "id": "MLModelExample",
        "meta": {
            "profile": [
                "https://joofio.github.io/obs-cdss-fhir/StructureDefinition/MLModel"
            ]
        },
        "identifier": [{"value": "1"}],
        "status": "active",
        # "version": [{"value": "1", "type": {"coding": [{"display": "x"}]}}],
    }
    mlmodel["version"] = []

    for k, v in mydict["meta"].items():
        if k not in ["commit", "timestamp"]:
            print(k, v)
            mlmodel["version"].append(
                {"value": str(v["version"]), "type": {"coding": [{"display": k}]}}
            )

    # {'meta': {'commit': None, 'IQR': {'model': 'z-score', 'version': 0.1},
    # 'Missing': {'model': 'traindata', 'version': 0.1}, 'Correctness': {'model': 'bayes', 'version': 0.1}, 'expecations':
    #  {'model': 'human', 'version': 0.1}, 'lof': {'model': 'lof', 'version': 0.1},
    # 'elliptic': {'model': 'elliptic', 'version': 0.1}, 'timestamp': '20230622T120556'},

    dev = Device(**mlmodel)
    mess_dict["entry"].append(
        {
            "resource": dev.dict(),
            "fullUrl": "http://localhost:8080/fhir/Device/MLModelExample",
        }
    )
    for idx, x in enumerate(mydict["scores"].items()):
        print(x)
        obs_dict = {
            "resourceType": "Observation",
            "id": ids[idx],
            "status": "final",
            "device": {"reference": "MLModelExample"},
            "code": {
                "coding": [
                    {
                        "code": x[0],
                        "system": "http://example.org/CodeSystem/my-cs",
                        "display": x[0],
                    }
                ]
            },
            "valueQuantity": {"value": float(x[1])},
        }
        obs = Observation(**obs_dict)

        mess_dict["entry"].append(
            {
                "resource": obs.dict(),
                "fullUrl": "http://localhost:8080/fhir/Observation/" + ids[idx],
            }
        )
    msg = Bundle(**mess_dict)

    return msg.dict()



def quality_score(ndf):
    """
    gets quality score- called by api
    """

  
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
    print(result_df)
    decisions = make_decisions(result_df)
    decisions["correctness_cols"]["gritbot"] = gritbot_score

    return final_score,decisions,lof_score,ee_score,missing_score,correctness_score,iqr_score,expectations_score