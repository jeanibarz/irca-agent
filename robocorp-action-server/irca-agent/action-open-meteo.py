"""
A simple AI Action template for comparing timezones

Please checkout the base guidance on AI Actions in our main repository readme:
https://github.com/robocorp/robocorp/blob/master/README.md

"""

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


@action
def get_current_weather_data(latitude: float, longitude: float) -> str:
    """
    Fetches current weather data from Open-Meteo API.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.

    Returns:
        str: Current weather data as a formatted string.
    """

    params = {"latitude": latitude, "longitude": longitude, "current": "temperature_2m,wind_speed_10m"}

    response = requests.get(OPEN_METEO_API_BASE_URL, params=params)
    data = response.json()

    return format_weather_data(data, data_type="current")


@action
def get_hourly_weather_data(latitude: float, longitude: float, option: str) -> str:
    """
    Fetches hourly weather data from Open-Meteo API.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        option (str): Predefined option for data retrieval. Available options:
                      - "temperature": Retrieves only the temperature data.
                      - "humidity_wind": Retrieves only the relative humidity and wind speed data.
                      - "all": Retrieves temperature, relative humidity, and wind speed data.
    Returns:
        str: Hourly weather data as a formatted string.
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

    # return data
    return format_weather_data(data, data_type="hourly")


@action
def get_past_weather_data(latitude: float, longitude: float, past_days: int, option: str) -> str:
    """
    Fetches past weather data from Open-Meteo API.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        past_days (int): Number of past days to fetch data for.
        option (str): Predefined option for data retrieval. Available options:
                      - "temperature": Retrieves only the temperature data.
                      - "humidity_wind": Retrieves only the relative humidity and wind speed data.
                      - "all": Retrieves temperature, relative humidity, and wind speed data.

    Returns:
        str: Past weather data as a formatted string.
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

    return format_weather_data(data, data_type="past")


@action
def get_historical_weather_data(latitude: float, longitude: float, start_date: str, end_date: str, option: str) -> str:
    """
    Fetches historical weather data from Open-Meteo API.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        start_date (str): Start date for historical data in YYYY-MM-DD format.
        end_date (str): End date for historical data in YYYY-MM-DD format.
        option (str): Predefined option for data retrieval. Available options:
                      - "temperature": Retrieves only the temperature data.
                      - "humidity_wind": Retrieves only the relative humidity and wind speed data.
                      - "all": Retrieves temperature, relative humidity, and wind speed data.

    Returns:
        str: Historical weather data as a formatted string.
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

    return format_weather_data(data, data_type="historical")


def format_weather_data(data, data_type):
    # Check if there's an error in the response
    if data.get("error"):
        return f"Error: {data.get('reason', 'Unknown error')}"

    if data_type in ["hourly", "past", "historical"]:
        weather_data = data.get("hourly", {})
        times = weather_data.get("time", [])

        title = {
            "hourly": "Hourly Weather Data:\n",
            "past": "Past Weather Data (Last Few Days):\n",
            "historical": "Historical Weather Data:\n",
        }.get(data_type, "Weather Data:\n")

        formatted = title
        for i, time in enumerate(times):
            formatted += f"Time: {time}"
            if "temperature_2m" in weather_data:
                formatted += f", Temperature: {weather_data['temperature_2m'][i]} °C"
            if "relative_humidity_2m" in weather_data:
                formatted += f", Humidity: {weather_data['relative_humidity_2m'][i]}%"
            if "wind_speed_10m" in weather_data:
                formatted += f", Wind Speed: {weather_data['wind_speed_10m'][i]} km/h"
            formatted += "\n"
    elif data_type == "current":
        current = data.get("current", {})
        formatted = (
            f"Current Weather Data:\n"
            f"- Time: {current.get('time')}\n"
            f"- Temperature: {current.get('temperature_2m')} °C\n"
            f"- Wind Speed: {current.get('wind_speed_10m')} km/h\n"
        )
    else:
        formatted = "Invalid data type."

    return formatted


def format_time_based_weather_data(weather_data, data_type):
    times = weather_data.get("time", [])
    temperatures = weather_data.get("temperature_2m", [])
    humidities = weather_data.get("relative_humidity_2m", [])
    wind_speeds = weather_data.get("wind_speed_10m", [])

    title = "Hourly Weather Data:\n" if data_type == "hourly" else "Past Weather Data (Last Few Days):\n"
    formatted = title
    for i, time in enumerate(times):
        formatted += (
            f"Time: {time}, "
            f"Temperature: {temperatures[i]} °C, "
            f"Humidity: {humidities[i]}%, "
            f"Wind Speed: {wind_speeds[i]} km/h\n"
        )
    return formatted
