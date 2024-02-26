"""
A simple AI Action template for generating passwords

Please checkout the base guidance on AI Actions in our main repository readme:
https://github.com/robocorp/robocorp/blob/master/README.md

"""

import json
import random
import string
from robocorp.actions import action


@action(is_consequential=False)
def generate_password(
    min_length: int = 8,
    max_length: int = 12,
    use_alphanumeric: bool = True,
    use_symbols: bool = True,
    num_symbols: int = 2,
    use_uppercase: bool = True,
    use_lowercase: bool = True,
) -> str:
    """
    Generates a random password based on the given criteria.

    Args:
        min_length (int): Minimum length of the password.
        max_length (int): Maximum length of the password.
        use_alphanumeric (bool): Include alphanumeric characters if True.
        use_symbols (bool): Include symbols if True.
        num_symbols (int): Number of symbols to include.
        use_uppercase (bool): Include uppercase characters if True.
        use_lowercase (bool): Include lowercase characters if True.

    Returns:
        str: A randomly generated password.
    """
    if min_length > max_length:
        return "Error: Minimum length cannot be greater than maximum length."

    characters = ""
    if use_alphanumeric:
        characters += string.ascii_letters + string.digits
    if use_symbols:
        characters += string.punctuation
    if use_uppercase:
        characters += string.ascii_uppercase
    if use_lowercase:
        characters += string.ascii_lowercase

    # Ensuring the password contains the specified number of symbols if required
    if use_symbols and num_symbols > 0:
        password_symbols = "".join(random.choice(string.punctuation) for _ in range(num_symbols))
    else:
        password_symbols = ""

    # Calculating the number of remaining characters to be filled
    remaining_length = random.randint(min_length, max_length) - len(password_symbols)
    if remaining_length < 0:
        return "Error: Number of symbols exceeds the maximum length of the password."

    # Generating the remaining part of the password
    password_rest = "".join(random.choice(characters) for _ in range(remaining_length))
    password = list(password_symbols + password_rest)
    random.shuffle(password)

    return json.dumps({"password": "".join(password)}, separators=(",", ":"))
