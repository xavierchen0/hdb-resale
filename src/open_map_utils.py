import asyncio
import os
import re
from pathlib import Path

import aiohttp
import pandas as pd
import requests
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

# Load .env's environment variables
ROOT_DIR = Path().parent.resolve()

load_dotenv(ROOT_DIR / ".env")


def _authenticate():
    """
    Returns the access token required for certain One Map API services.
    """
    # Store ONEMAP_EMAIL and ONEMAP_PWD in a .env file in root
    onemap_email = os.environ["ONEMAP_EMAIL"]
    onemap_pwd = os.environ["ONEMAP_PWD"]

    url = "https://www.onemap.gov.sg/api/auth/post/getToken"

    payload = {
        "email": onemap_email,
        "password": onemap_pwd,
    }

    response = requests.request("POST", url, json=payload)

    return response.json()["access_token"]


# Get access token from One Map
ACCESS_TOKEN = _authenticate()


def openmap_search(searchVal, returnGeom="Y", getAddrDetails="Y", pageNum=1):
    """
    https://www.onemap.gov.sg/apidocs/search

    Returns query results
    """
    url = "https://www.onemap.gov.sg/api/common/elastic/search"

    params = {
        "searchVal": searchVal,
        "returnGeom": returnGeom,
        "getAddrDetails": getAddrDetails,
        "pageNum": pageNum,
    }

    headers = {"Authorization": ACCESS_TOKEN}

    response = requests.get(url, params=params, headers=headers)

    return response


async def _fetch_one_hdb_addr(client, limiter, addr):
    """
    Internal function used in async fetching.
    """
    url = "https://www.onemap.gov.sg/api/common/elastic/search"
    params = {"searchVal": addr, "returnGeom": "Y", "getAddrDetails": "Y", "pageNum": 1}

    async with limiter:
        try:
            async with client.get(url, params=params) as response:
                if response.status != 200:
                    print(
                        f"Response status code not 200! Status Code = {response.status}"
                    )
                    return {
                        "full_addr": addr,
                        "postal_code": pd.NA,
                        "x": pd.NA,
                        "y": pd.NA,
                        "latitude": pd.NA,
                        "longitude": pd.NA,
                    }

                data = await response.json()
        except asyncio.TimeoutError:
            print("asyncio TimeoutError")
            return {
                "full_addr": addr,
                "postal_code": pd.NA,
                "x": pd.NA,
                "y": pd.NA,
                "latitude": pd.NA,
                "longitude": pd.NA,
            }

        if data.get("found", 0) == 0:
            print("Zero results found")
            return {
                "full_addr": addr,
                "postal_code": pd.NA,
                "x": pd.NA,
                "y": pd.NA,
                "latitude": pd.NA,
                "longitude": pd.NA,
            }

        if data.get("found", 0) == 1:
            return {
                "full_addr": addr,
                "postal_code": data["results"][0]["POSTAL"],
                "x": data["results"][0]["X"],
                "y": data["results"][0]["Y"],
                "latitude": data["results"][0]["LATITUDE"],
                "longitude": data["results"][0]["LONGITUDE"],
            }

        if data.get("found", 0) > 1:
            indices = []
            postal_code = []
            block = addr.split(" ", 1)[0]
            match = re.search(r"\d+", block)
            block_num_only = match.group() if match else None
            block_3digit = block_num_only.zfill(3)

            for i, r in enumerate(data["results"]):
                if (
                    block == r["BLK_NO"]
                    and block_3digit == r["POSTAL"][3:]
                    and r["POSTAL"] not in postal_code
                ):
                    postal_code.append(r["POSTAL"])
                    indices.append(i)

            x_coord = [data["results"][i]["X"] for i in indices]
            y_coord = [data["results"][i]["Y"] for i in indices]
            latitude = [data["results"][i]["LATITUDE"] for i in indices]
            longitude = [data["results"][i]["LONGITUDE"] for i in indices]

            return {
                "full_addr": addr,
                "postal_code": postal_code,
                "x": x_coord,
                "y": y_coord,
                "latitude": latitude,
                "longitude": longitude,
            }


async def many_openmap_hdb_search(addresses):
    """
    Using async to deal with high network costs due to API calls to Open Map.

    Result will be in the same order as addresses.
    """
    headers = {"Authorization": ACCESS_TOKEN}
    limiter = AsyncLimiter(max_rate=5000)  # Rate limit by OpenMap = 250 calls/min

    async with aiohttp.ClientSession(headers=headers) as client:
        tasks = [_fetch_one_hdb_addr(client, limiter, addr) for addr in addresses]

        results = await tqdm_asyncio.gather(
            *tasks, desc="OneMap Search API", total=len(tasks)
        )

    return pd.DataFrame(results)
