import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing from src/
ROOT_DIR = Path().parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src.open_map_utils as open_map_utils


def get_geo_data_for_each_row(row):
    full_addr = row["full_addr"]

    response = open_map_utils.openmap_search(
        searchVal=full_addr, returnGeom="Y", getAddrDetails="Y"
    )

    num_results = response.json()["found"]
    status_code = response.status_code

    # used to check progress of apply function
    print(row.name, num_results, status_code)

    if status_code != 200:
        return {
            "postal_code": pd.NA,
            "x": pd.NA,
            "y": pd.NA,
            "latitude": pd.NA,
            "longitude": pd.NA,
        }

    if num_results == 1:
        postal_code = response.json()["results"][0]["POSTAL"]
        x_coord = response.json()["results"][0]["X"]
        y_coord = response.json()["results"][0]["Y"]
        latitude = response.json()["results"][0]["LATITUDE"]
        longitude = response.json()["results"][0]["LONGITUDE"]
    else:
        indices = []
        postal_code = []

        for i, result in enumerate(response.json()["results"]):
            if (
                row["block"] in str(result["POSTAL"])[3:]
                and result["POSTAL"] not in postal_code
            ):
                postal_code.append(result["POSTAL"])
                indices.append(i)

        x_coord = [response.json()["results"][i]["X"] for i in indices]
        y_coord = [response.json()["results"][i]["Y"] for i in indices]
        latitude = [response.json()["results"][i]["LATITUDE"] for i in indices]
        longitude = [response.json()["results"][i]["LONGITUDE"] for i in indices]

    return {
        "postal_code": postal_code,
        "x": x_coord,
        "y": y_coord,
        "latitude": latitude,
        "longitude": longitude,
    }


def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield i // size, lst[i : i + size]  # (batch_index, slice)


def accessibility_score_one_point(pt, hawker_coords, lam=500):
    # pt is shapely Point in EPSG:3414 (meters)
    # hawker_coords is array shape (N,2) of x,y
    x0, y0 = pt.x, pt.y
    dx = hawker_coords[:, 0] - x0
    dy = hawker_coords[:, 1] - y0
    dists = np.sqrt(dx * dx + dy * dy)  # Euclidean distance in meters
    return np.exp(-dists / lam).sum()
