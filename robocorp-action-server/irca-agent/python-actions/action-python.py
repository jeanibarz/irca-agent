from datetime import datetime
import json

import pytz

from robocorp.actions import action


@action(is_consequential=False)
def get_contextual_information():
    """
    Retrieves contextual information including the current date and time in the Europe/Paris timezone,
    and constant values for user's location, name, birthdate, and sex.

    This function does not require any input arguments.

    Returns:
        dict: A dictionary containing:
            - current_time (str): The current date and time in Europe/Paris timezone, formatted as 'YYYY-MM-DD HH:MM:SS TZ'.
            - location (str): The user's location, set as a constant ('31220, Saint-Julien, France').
            - name (str): The user's name, set as a constant ('Jean Ibarz').
            - birthdate (str): The user's birthdate, set as a constant ('15 Aout 1987').
            - sex (str): The user's sex, set as a constant ('M').
    """
    paris_tz = pytz.timezone("Europe/Paris")
    current_time = datetime.now(paris_tz)

    user_location = "31220, Saint-Julien, France"
    user_name = "Jean Ibarz"
    user_birthdate = "15 Aout 1987"
    user_sex = "M"
    user_personal_email = "ibarz.jean@gmail.com"

    data = {
        "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S %Z%z"),
        "location": user_location,
        "name": user_name,
        "birthdate": user_birthdate,
        "sex": user_sex,
        "personal_email": user_personal_email,
    }
    return json.dumps(data, separators=(",", ":"))
