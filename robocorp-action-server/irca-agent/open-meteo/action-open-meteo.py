"""
A simple AI Action template for comparing timezones

Please checkout the base guidance on AI Actions in our main repository readme:
https://github.com/robocorp/robocorp/blob/master/README.md

"""

import json
import requests
from robocorp.actions import action

# https://github.com/public-apis/public-apis?tab=readme-ov-file#weather
# https://open-meteo.com/

OPEN_METEO_API_BASE_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORICAL_API_BASE_URL = "https://archive-api.open-meteo.com/v1/era5"

DATA_OPTIONS = {
    "temperature": "temperature_2m",
    "humidity_wind": "relative_humidity_2m,wind_speed_10m",
    "all": "temperature_2m,relative_humidity_2m,wind_speed_10m",
}


@action(is_consequential=False)
def get_current_weather_data(latitude: float, longitude: float, option: str) -> str:
    """
    Fetches current weather data from Open-Meteo API.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        option (str): Data retrieval option. Options include "temperature", "humidity_wind", and "all".

    Returns:
        str: Current weather data as a compact JSON string.
    """
    if option not in DATA_OPTIONS:
        return f"Invalid option. Available options are: {', '.join(DATA_OPTIONS.keys())}"

    data_fields = DATA_OPTIONS[option]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": data_fields,
    }

    response = requests.get(OPEN_METEO_API_BASE_URL, params=params)
    data = response.json()

    return json.dumps(data, separators=(",", ":"))


@action(is_consequential=False)
def get_hourly_weather_data(latitude: float, longitude: float, option: str) -> str:
    """
    Fetches hourly weather data from Open-Meteo API.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        option (str): Data retrieval option. Options include "temperature", "humidity_wind", and "all".

    Returns:
        str: Hourly weather data as a compact JSON string.
    """
    if option not in DATA_OPTIONS:
        return f"Invalid option. Available options are: {', '.join(DATA_OPTIONS.keys())}"

    data_fields = DATA_OPTIONS[option]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": data_fields,
    }

    response = requests.get(OPEN_METEO_API_BASE_URL, params=params)
    data = response.json()

    return json.dumps(data, separators=(",", ":"))


@action(is_consequential=False)
def get_past_weather_data(latitude: float, longitude: float, past_days: int, option: str) -> str:
    """
    Fetches past weather data for a specified number of past days from Open-Meteo API.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        past_days (int): Number of past days to fetch data for.
        option (str): Data retrieval option. Options include "temperature", "humidity_wind", and "all".

    Returns:
        str: Past weather data as a compact JSON string.
    """
    if option not in DATA_OPTIONS:
        return f"Invalid option. Available options are: {', '.join(DATA_OPTIONS.keys())}"

    data_fields = DATA_OPTIONS[option]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "past_days": past_days,
        "hourly": data_fields,
    }

    response = requests.get(OPEN_METEO_API_BASE_URL, params=params)
    data = response.json()

    return json.dumps(data, separators=(",", ":"))


@action(is_consequential=False)
def get_historical_weather_data(latitude: float, longitude: float, start_date: str, end_date: str, option: str) -> str:
    """
    Fetches historical weather data from Open-Meteo API for a specified date range.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        start_date (str): Start date for historical data in YYYY-MM-DD format.
        end_date (str): End date for historical data in YYYY-MM-DD format.
        option (str): Data retrieval option. Options include "temperature", "humidity_wind", and "all".

    Returns:
        str: Historical weather data as a compact JSON string.
    """
    if option not in DATA_OPTIONS:
        return f"Invalid option. Available options are: {', '.join(DATA_OPTIONS.keys())}"

    data_fields = DATA_OPTIONS[option]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": data_fields,
    }

    response = requests.get(OPEN_METEO_HISTORICAL_API_BASE_URL, params=params)
    data = response.json()

    return json.dumps(data, separators=(",", ":"))
