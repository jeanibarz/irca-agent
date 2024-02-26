import json
import re
import requests
from enum import Enum
from copy import deepcopy


# Define GPT categories
class GPTCategory(Enum):
    LOCATION_AND_GEOGRAPHY_MANAGER = "location-and-geography-manager"
    MAIL_MANAGER = "mail-manager"
    DATA_AND_INFORMATION_MANAGER = "data-and-information-manager"


# Define GPTs
GPTs = [
    ("contextual-information-toolbox", GPTCategory.LOCATION_AND_GEOGRAPHY_MANAGER),
    ("contextual-information-toolbox", GPTCategory.MAIL_MANAGER),
    ("contextual-information-toolbox", GPTCategory.DATA_AND_INFORMATION_MANAGER),
]

# Terminal output containing the URL
terminal_output = """
root@docker-desktop:/workspace/robocorp-action-server/contextual-information-toolbox# action-server start --expose

  ⚡️ Starting Action Server v0.0.20

Using datadir: /root/robocorp/.action_server/contextual-information-toolbox_ce364fd6
Logs may be found at: /root/robocorp/.action_server/contextual-information-toolbox_ce364fd6/server_log.txt.
Action package seems ok. Bootstrapping RCC environment (please wait, this can take a long time).
Python interpreter path: /opt/robocorp/ht/720c7a8_a78733b_ead76f62/bin/python3
> Resume previous expose URL https://ninety-nine-tough-moths.robocorp.link Y/N? [Y]y

  ⚡️ Local Action Server: http://localhost:8080
  🌍 Public URL: https://thirty-three-purple-warthogs.robocorp.link
"""

# Regular expression pattern to extract the URL
url_pattern = r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"

# Extracting all URLs from the terminal output
urls = re.findall(url_pattern, terminal_output)


if urls:
    # Using the last URL from the list
    url = urls[-1]
    print("Extracted URL:", url)

    # Retrieving content from the extracted URL
    response = requests.get(url)
    if response.status_code == 200:
        import json

        # Parse the content into a dictionary
        parsed_content = json.loads(response.text)

        # Iterate over each GPT
        for gpt_toolbox, gpt_category in GPTs:
            print("GPT Category:", gpt_category.value)

            # Make a deep copy of parsed_content
            filtered_content = deepcopy(parsed_content)

            # Filter paths based on GPT category and replace them in the copied content
            for path, details in parsed_content["paths"].items():
                if gpt_toolbox not in path and gpt_category.value not in path:
                    del filtered_content["paths"][path]

            # Dump the filtered content to a file
            output_filename = f"{gpt_category.value}_content.json"
            with open(output_filename, "w") as f:
                json.dump(filtered_content, f, separators=(",", ":"))

            print(f"Filtered content dumped to {output_filename}")
    else:
        print("Failed to retrieve content from the URL. Status code:", response.status_code)
else:
    print("No URL found in the terminal output.")
