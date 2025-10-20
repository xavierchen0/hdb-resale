import os
from pathlib import Path

import requests
from dotenv import load_dotenv

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
