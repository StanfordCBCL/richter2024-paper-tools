import click
import wget
import os
from rich import print
import zipfile
import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool
from shutil import move

# python download.py /Volumes/richter/final_data/projects 0082_0001 0142_1001 0003_0001 0164_0001 0070_0001 0093_0001 0103_0001 0075_1001 0156_0001 0069_0001 0096_0001 0087_1001 0139_1001 0147_1001 0064_1001 0112_1001 0006_0001 0063_1001 0131_0000 0080_0001 0185_0001 0091_0001 0187_0002 0083_2002 0104_0001 0151_0001 0072_0001 0175_0000 0094_0001 0101_0001 0077_1001 0154_0001 0163_0001 0088_1001 0172_0001 0145_1001 0066_0001 0110_0001 0162_3001 0102_0001 0074_0001 0097_0001 0068_0001 0138_1001 0086_0001 0146_1001 0065_1001 0002_0001 0107_0001 0176_0000 0095_0001 0183_1002 0089_1001 0076_1001 0155_0001 0144_1001 0067_0001 0157_0000 0111_0001 0098_0001 0129_0000 0160_6001 0141_1001 0081_0001 0140_2001 0184_0001 0090_0001 0186_0002 0105_0001 0150_0001 0174_0000 0073_0001
# python download.py /Volumes/richter/final_data/projects 0090_0001 0186_0002 0105_0001 0150_0001 0174_0000 0073_0001

BASE_PATH = "https://www.vascularmodel.com/svprojects/"


def list_webdir(url, ext=""):
    page = requests.get(url).text
    soup = BeautifulSoup(page, "html.parser")
    return [
        url + "/" + node.get("href")
        for node in soup.find_all("a")
        if node.get("href").endswith(ext)
    ]


NEW_PROJECT_NAMES = [
    f.split("/")[-1].removesuffix(".zip")
    for f in list_webdir(BASE_PATH)
    if f.endswith(".zip")
]


def get_legacy_name(new_name):
    try:
        path = BASE_PATH + new_name + "/Meshes"
        content = list_webdir(path)
        legacy_name = [
            f.split("/")[-1].removesuffix(".vtp")
            for f in content
            if f.endswith(".vtp")
        ][0]
        print(legacy_name)
        return legacy_name
    except IndexError:
        None


# LEGACY_NAME_MAPPING = {}
# for new_name in NEW_PROJECT_NAMES:
#     legacy_name = get_legacy_name(new_name)
#     LEGACY_NAME_MAPPING[legacy_name] = new_name

LEGACY_NAME_MAPPING = {
    "0063_1001": "0001_H_AO_SVD",
    "0064_1001": "0002_H_AO_SVD",
    "0065_1001": "0003_H_AO_SVD",
    "0075_1001": "0004_H_AO_SVD",
    "0076_1001": "0005_H_AO_SVD",
    "0077_1001": "0006_H_AO_SVD",
    "0090_0001": "0007_H_AO_H",
    "0091_0001": "0008_H_AO_H",
    "0092_0001": "0009_H_AO_H",
    "0093_0001": "0010_H_AO_H",
    "0094_0001": "0011_H_AO_H",
    "0095_0001": "0012_H_AO_H",
    "0101_0001": "0013_H_AO_COA",
    "0102_0001": "0014_H_AO_COA",
    "0103_0001": "0015_H_AO_COA",
    "0104_0001": "0016_H_AO_COA",
    "0105_0001": "0017_H_AO_COA",
    "0106_0001": "0018_H_AO_COA",
    "0107_0001": "0019_H_AO_COA",
    "0111_0001": "0020_H_AO_COA",
    "0129_0000": "0021_H_AO_MFS",
    "0130_0000": "0022_H_AO_MFS",
    "0131_0000": "0023_H_AO_MFS",
    "0154_0001": "0024_H_AO_H",
    "0174_0000": "0025_H_AO_MFS",
    "0175_0000": "0026_H_AO_MFS",
    "0176_0000": "0027_H_AO_MFS",
    "0003_0001": "0028_H_ABAO_H",
    "0006_0001": "0029_H_ABAO_H",
    "0110_0001": "0030_H_ABAO_H",
    "0138_1001": "0031_H_ABAO_AAA",
    "0139_1001": "0032_H_ABAO_AAA",
    "0140_2001": "0033_H_ABAO_AAA",
    "0141_1001": "0034_H_ABAO_AAA",
    "0142_1001": "0035_H_ABAO_AAA",
    "0144_1001": "0036_H_ABAO_AAA",
    "0145_1001": "0037_H_ABAO_AAA",
    "0146_1001": "0038_H_ABAO_AAA",
    "0147_1001": "0039_H_ABAO_AAA",
    "0148_1001": "0040_H_ABAO_AAA",
    "0149_1001": "0041_H_ABAO_AAA",
    "0150_0001": "0042_H_ABAO_AAA",
    "0151_0001": "0043_H_ABAO_AAA",
    "0156_0001": "0044_H_ABAO_AAA",
    "0157_0000": "0045_H_ABAO_AAA",
    "0160_6001": "0046_H_ABAO_AIOD",
    "0161_0001": "0047_H_ABAO_AIOD",
    "0162_3001": "0048_H_ABAO_AIOD",
    "0163_0001": "0049_H_ABAO_AIOD",
    "0078_0001": "0050_H_CERE_H",
    "0079_0001": "0051_H_CERE_H",
    "0166_0001": "0052_H_CERE_H",
    "0167_0001": "0053_H_CERE_H",
    "0063_0001": "0054_H_PULMFON_TAT",
    "0064_0001": "0055_H_PULMFON_HLHS",
    "0065_0001": "0056_H_PULMFON_TAT",
    "0075_0001": "0057_H_PULMFON_TAT",
    "0076_0001": "0058_H_PULMFON_PAT",
    "0077_0001": "0059_H_PULMFON_TAT",
    "0096_0001": "0060_H_PULMGLN_SVD",
    "0097_0001": "0061_H_PULMGLN_SVD",
    "0098_0001": "0062_H_PULMGLN_SVD",
    "0099_0001": "0063_H_PULMGLN_SVD",
    "0125_0001": "0064_H_PULMFON_SVD",
    "0126_0001": "0065_H_PULMFON_SVD",
    "0002_0001": "0066_H_CORO_H",
    "0108_0001": "0067_H_CORO_KD",
    "0172_0001": "0068_H_CORO_KD",
    "0173_1001": "0069_H_CORO_KD",
    "0183_1002": "0070_H_CORO_KD",
    "0184_0001": "0071_H_CORO_KD",
    "0185_0001": "0072_H_CORO_KD",
    "0186_0002": "0073_H_CORO_H",
    "0187_0002": "0074_H_CORO_KD",
    "0188_0001": "0075_H_CORO_CAD",
    "0189_0001": "0076_H_CORO_CAD",
    "0005_1001": "0077_H_PULM_H",
    "0080_0001": "0078_H_PULM_H",
    "0081_0001": "0079_H_PULM_H",
    "0082_0001": "0080_H_PULM_H",
    "0083_2002": "0081_H_PULM_PAH",
    "0084_0001": "0082_H_PULM_H",
    "0085_1001": "0083_H_PULM_PAH",
    "0086_0001": "0084_H_PULM_H",
    "0087_1001": "0085_H_PULM_PAH",
    "0088_1001": "0086_H_PULM_PAH",
    "0089_1001": "0087_H_PULM_PAH",
    "0112_1001": "0088_H_PULM_PAH",
    "0118_1000": "0089_H_PULM_ALGS",
    "0119_0001": "0090_H_PULM_ALGS",
    "0134_0002": "0091_H_PULM_TOF",
    "0155_0001": "0092_H_PULM_H",
    "0066_0001": "0093_A_AO_H",
    "0067_0001": "0094_A_AO_H",
    "0068_0001": "0095_A_AO_H",
    "0069_0001": "0096_A_AO_COA",
    "0070_0001": "0097_A_AO_COA",
    "0071_0001": "0098_A_AO_COA",
    "0072_0001": "0099_A_AO_COA",
    "0073_0001": "0100_A_AO_COA",
    "0074_0001": "0101_A_AO_COA",
}


def get_download_link(model_name):
    return f"https://www.vascularmodel.com/svprojects/{model_name}.zip"


@click.command()
@click.argument("target_folder")
@click.argument("models", nargs=-1)
def main(target_folder, models):
    for model in models:
        try:
            new_name = LEGACY_NAME_MAPPING[model]
        except KeyError:
            print("Couldn't find model", model)
            continue

        print("Dowloading", model)
        source = get_download_link(new_name)
        target = os.path.join(target_folder, new_name)
        final_target = os.path.join(target_folder, model)
        wget.download(source, target + ".zip")
        print("\nUnzipping", model)
        with zipfile.ZipFile(target + ".zip", "r") as zip_ref:
            zip_ref.extractall(target_folder)
        move(target, final_target)
        print("Cleaning up", model)
        os.remove(target + ".zip")


if __name__ == "__main__":
    main()
