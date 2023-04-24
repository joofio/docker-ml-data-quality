import numpy as np
import pandas as pd


def preprocess_df(df, silo):

    # list(full_data["G_TERAPEUTICA"].unique())
    # transform all from SIM/NAO to S/N e similar
    # BACIA [nan 'ADEQUADA' 'LIMITE' 'INADEQUADA' 'A' 'L']
    standard_map = {
        "INADEQUADA": "Inadequada",
        "L": "Limite",
        "A": "Adequada",
        "X": "Sim",
        "S": "Sim",
        "N": "Não",
        -7: np.nan,
        "ADEQUADA": "Adequada",
        "LIMITE": "Limite",
        "DESCONHECIDO,RH_DESCONHECIDO": np.nan,
        "DESCONHECIDO,": np.nan,
        "SIM": "Sim",
        "NAO": "Não",
        "-1": np.nan,
        -1: np.nan,
        "NS": np.nan,
        ",": np.nan,
        "Sim": "Sim",
        "UNKNOWN": np.nan,
        "Desconhecida": np.nan,
        "Desconhecido": np.nan,
        "DESCONHECIDO": np.nan,
        " ": np.nan,
        "Desconhecido,": np.nan,
        "  ": np.nan,
    }

    def standardize_null(x, mapping):
        if x in mapping.keys():
            return mapping[x]
        if pd.isna(x):
            return np.nan
        return x

    for col in df.columns:
        df[col] = df[col].apply(standardize_null, mapping=standard_map)
    ### Mapping
    # transform into labels human readable
    label_map = {
        "APRESENTACAO_ADMISSAO": {
            "apr.cefala.3": "Cefálica",
            "apr.pelv.1": "Pélvica",
            "apr.esp.1": "Espádua",
            "apr.trans.3": "Transversa",
            "apr.desc.1": np.nan,
            "apr.face.2": "Face",
        }
    }

    df.replace(label_map, inplace=True)
    ## relabel apresentacao
    map_cols = [
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
    ]

    mapping = {
        "CHLN": {
            "10044": "pélvico modo nadegas",
            "10052": "situação transversa",
            "10080": "cefálica",
            "10078": "cefálica",
            "10061": "pélvico modo pés",
            "10064": "pélvica completa",
            "10066": "variável",
            "10006": "situação oblíqua",
            "10019": "pélvica",
            "10079": "pélvica",
            "10081": "apresentação composta",
            "10083": "cefálica",
            "10082": "pélvica",
        },
        "CHBV": {
            "1": "cefalica",
            "3": "cefalica, dorso anterior",
            "10000": "obliquo, dorso anterior",
            "8": "pelvica dorso-posterior",
            "4": "cefalica dorso-posterior",
            "5": "pélvica",
            "7": "pélvica dorso-anterior",
            "9": "transversa",
        },
        "HSO": {
            "1": "cefálica",
            "3": "cefálica, dorso anterior",
            "10000": "indiferente",
            "10002": "cefálica - dorso dta",
            "10003": "cefálica-dorso esq",
            "10004": "obliquo",
            "10005": "pélvico modo pés",
            "10006": "instável",
            "8": "pélvica dorso-posterior",
            "4": "cefálica dorso-posterior",
            "5": "pélvica",
            "7": "pélvica dorso-anterior",
            "9": "transversa",
        },
        "CHSJ": {
            "1": "cefálica",
            "3": "cefálica, dorso anterior",
            "7": "pélvica dorso-anterior",
            "8": "pélvica dorso-posterior",
            "10000": "cefálica insinuada",
            "10001": "pélvica",
            "10012": "cefálica e deflectida",
            "10038": "oblíqua",
            "10042": "cefálica muito insinuada",
            "10043": "situação transversa",
        },
        "CHTS": {
            "1": "cefalica",
            "3": "cefalica, dorso anterior",
            "10000": "cefalica dta",
            "10001": "pélvica esq",
            "10002": "cefalica dta",
            "10003": "pélvica dta",
            "10004": "espadua",
            "10005": "pélvica",
            "10006": "instavel",
            "4": "cefalica dorso-posterior",
            "5": "pélvica",
            "7": "pélvica dorso-anterior",
            "9": "transversa",
        },
        "CHEDV": {
            "1": "cefálica",
            "3": "cefálica, dorso-anterior",
            "4": "cefálica dorso-posterior",
            "5": "pélvica",
            "7": "pélvica dorso-anterior",
            "8": "pélvica dorso-posterior",
            "9": "transversa",
            "10001": "instável",
            "10002": "posterior alta",
            "10003": "pelve modo pés",
            "10004": "pelve muito insinuada",
            "10005": "cefálico dorso à esquerda",
            "10006": "cefálico dorso à direita",
            "10007": "pélvica dorso à direita",
            "10008": "pélvica dorso à esquerda",
            "10009": "cefálica insinuada",
        },
        "CHVNGE": {
            "1": "cefálica",
            "10022": "cefálico esquerdo",
            "10026": "apresentação composta",
            "10027": "Cefálica dorso anterior",
            "10000": "cefálica dorso à direita",
            "10001": "cefálica dorso à esquerda",
            "10002": "situação transversa, polo cefálico à esquerda",
            "10003": "pélvica",
            "10004": "situação transversa, polo cefálico à direita",
            "10005": "cefálica dorso à direita",
            "10006": "cefálica",
            "10007": "cefálica dorso à esquerda",
            "10008": "pélvica dorso à direita",
            "10010": "cefálica muito insinuada",
            "10011": "pelve franca, dorso à direita",
            "10012": "pelve franca, dorso à esquerda",
            "10013": "situação oblíqua com polo cefálico no quadrante inf esq",
            "10014": "situação oblíqua, com o pólo cefálico no quadr sup direito e pelve no QIE",
            "10015": "situação oblíqua",
            "10016": "pelve desdobrada",
            "10017": "pelve franca com dorso anterior esquerdo",
            "10018": "pelve, modo pés",
            "10019": "pelve completa",
            "10020": "cefálica muito insinuada",
            "10021": "pelve modo pés",
            "10023": "pelve franca com dorso anterior",
            "10024": "pelve modo nádegas",
            "10025": "situação oblíqua com polo cefálico no QSE",
            "3": "cefálica dorso-anterior",
            "4": "cefálica dorso-posterior",
            "7": "pélvica dorso-anterior",
            "8": "pélvica dorso-posterior",
            "9": "transverso",
        },
        "ULSM": {
            "1": "cefálica",
            "10017": "Espadua",
            "3": "cefálica, dorso anterior",
            "10000": "pélvica, dorso à esquerda",
            "10001": "pélvica",
            "10002": "instável",
            "10003": "cefálica, dorso à direita",
            "10004": "cefálica, dorso à esquerda",
            "10005": "pélvica",
            "10006": "transversa",
            "10008": "pélvica",
            "10009": "pélvica modo pés",
            "10010": "cefálica dorso-direita",
            "10011": "cefálica dorso-esquerda",
            "10014": "pélvica dorso-direita",
            "10015": "cefálica muito-inusitada",
            "10013": "pélvica dorso-direita",
            "8": "pélvica dorso-posterior",
            "4": "cefálica dorso-posterior",
            "5": "pélvica",
            "7": "pélvica dorso-anterior",
            "9": "transversa",
            "10012": "cefálica e deflectida",
            "10038": "oblíqua",
            "10043": "situação transversa",
        },
        "ULSAM": {
            "1": "cefálica",
            "3": "cefálica, dorso anterior",
            "4": "cefálica dorso-posterior",
            "5": "pélvica",
            "7": "pélvica dorso-anterior",
            "8": "pélvica dorso-posterior",
            "9": "transversa",
            "10000": "Instável",
            "10001": "cefálica, dorso à direita, posterior",
            "10002": "cefálico dorso anterior esquerdo",
            "10003": "transversa, dorso superior",
            "10004": "cefálica, dorso à direita",
            "10006": "cefálica, dorso à esquerda",
            "10007": "pélvica, dorso à direita",
            "10008": "Pélvica",
            "10009": "situação transversa, dorso inferior",
            "10010": "pelvica dorso á esquerda",
            "10011": "variável",
            "10012": "situação transversa",
        },
    }

    def mapping_apresentacao(x, silo):

        for map_col in map_cols:
            # print(x[map_col])
            # print(pd.isna(x[map_col]))
            if not pd.isna(x[map_col]):
                #   print(map_col)
                #  print(str(int(x[map_col])))
                # print(mapping[silo])
                try:
                    x[map_col] = mapping[silo][str(int(x[map_col]))]
                except:
                    print(silo, x[map_col])
                    x[map_col] = np.nan
        return x

    df = df.apply(mapping_apresentacao, axis=1, silo=silo)
    ### Mapping
    # transform into labels human readable
    l_label_map = {
        "Espadua": "espadua",
        "Cefálica dorso anterior": "cefálica dorso anterior",
        "cefálica, dorso anterior": "cefálica dorso anterior",
        "cefálico dorso à esquerda": "cefálica dorso à esquerda",
        "pelvica dorso-posterior": "pélvica dorso-posterior",
        "cefalica dta": "cefálica direita",
        "pelve modo pés": "pélvica modo pés",
        "situação oblíqua": "oblíqua",
        "cefálica dorso-posterior": "cefálica dorso posterior",
        "cefálica, dorso à esquerda": "cefálica dorso à esquerda",
        "cefálica - dorso dta": "cefálica dorso à direita",
        "cefálica dorso-esquerda": "cefálica dorso à esquerda",
        "cefálica dorso-direita": "cefálica dorso à direita",
        "cefálica dorso-anterior": "cefálica dorso anterior",
        "cefalica, dorso anterior": "cefálica dorso anterior",
        "cefalica dorso-posterior": "cefálica dorso posterior",
        "cefálica, dorso à direita": "cefálica dorso à direita",
        "cefálico dorso à direita": "cefálica dorso à direita",
        "cefálica-dorso esq": "cefálica dorso à esquerda",
        "cefálica, dorso-anterior": "cefálica dorso anterior",
        "pelve, modo pés": "pelve modo pés",
        "pélvico modo pés": "pélvica modo pés",
        "pélvica, dorso à direita": "pélvica dorso à direita",
        "Pélvica": "pélvica",
        "Instável": "instável",
        "transverso": "transversa",
        "situação transversa": "transversa",
        "pélvica dorso-direita": "pélvica dorso à direita",
        "instavel": "instável",
        "obliquo": "oblíqua",
        "cefalica": "cefálica",
    }
    for x, v in l_label_map.items():
        print(x, "||||", v)
    full_label_map = {}
    for col in map_cols:
        full_label_map[col] = l_label_map
    df.replace(full_label_map, inplace=True)
    # categoriess on GS return some errors, handle with this:
    def separate_gs(x):
        GS = x["GS"]
        if not type(GS) == str:
            return np.nan, np.nan
        if len(GS.split(",")) > 2:
            print(GS)
            return "check", "check"
        else:
            return GS.split(",")

    df[["GS_ABO", "GS_RH"]] = df.apply(separate_gs, axis=1, result_type="expand")
    df.drop(columns=["GS"], inplace=True)
    ### Mapping
    # transform into labels human readable
    label_map = {"GS_ABO": {"": np.nan}, "GS_RH": {"RH_DESCONHECIDO": np.nan}}
    df.replace(label_map, inplace=True)
    return df
