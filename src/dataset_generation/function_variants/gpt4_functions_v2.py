functions = [
    {
        "name": "get_random_joke",
        "description": "Fetch a random joke from an online joke database",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["general", "knock-knock", "programming"],
                    "description": "The category of the joke",
                }
            },
            "required": [],
        },
    },
    {
        "name": "solve_polynomial",
        "description": "Find the roots of a polynomial equation",
        "parameters": {
            "type": "object",
            "properties": {
                "coefficients": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Coefficients of the polynomial, starting with the highest degree",
                }
            },
            "required": ["coefficients"],
        },
    },
    {
        "name": "remote_device_control",
        "description": "Send a control signal to a specified remote device",
        "parameters": {
            "type": "object",
            "properties": {
                "device_id": {
                    "type": "string",
                    "description": "Unique identifier of the device",
                },
                "command": {
                    "type": "string",
                    "description": "Control command to be sent",
                },
            },
            "required": ["device_id", "command"],
        },
    },
    {
        "name": "add_numbers",
        "description": "Add a list of numbers together",
        "parameters": {
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "A list of numbers to be added",
                }
            },
            "required": ["numbers"],
        },
    },
    {
        "name": "control_home_appliance",
        "description": "Control a home appliance like a TV or radio",
        "parameters": {
            "type": "object",
            "properties": {
                "appliance": {
                    "type": "string",
                    "enum": ["TV", "radio"],
                    "description": "The appliance to control",
                },
                "action": {
                    "type": "string",
                    "enum": ["turn_on", "turn_off", "adjust_volume"],
                    "description": "The action to perform on the appliance",
                },
            },
            "required": ["appliance", "action"],
        },
    },
    {
        "name": "get_scheduled_alarms",
        "description": "Retrieve a list of scheduled alarms from a device",
        "parameters": {
            "type": "object",
            "properties": {
                "device_id": {
                    "type": "string",
                    "description": "Unique identifier of the device",
                }
            },
            "required": ["device_id"],
        },
    },
    {
        "name": "track_package",
        "description": "Track a package with a given tracking number",
        "parameters": {
            "type": "object",
            "properties": {
                "tracking_number": {
                    "type": "string",
                    "description": "The tracking number of the package",
                }
            },
            "required": ["tracking_number"],
        },
    },
    {
        "name": "start_waze_to_destination",
        "description": "Start navigation to a specified destination using Waze",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": "The destination address or coordinates",
                }
            },
            "required": ["destination"],
        },
    },
    {
        "name": "validate_json",
        "description": "Check if a given JSON string is syntactically correct",
        "parameters": {
            "type": "object",
            "properties": {
                "json_string": {
                    "type": "string",
                    "description": "The JSON string to validate",
                }
            },
            "required": ["json_string"],
        },
    },
    {
        "name": "convert_json_to_xml",
        "description": "Convert a JSON object to XML format",
        "parameters": {
            "type": "object",
            "properties": {
                "json_object": {
                    "type": "string",
                    "description": "The JSON object to convert",
                }
            },
            "required": ["json_object"],
        },
    },
    {
        "name": "send_text_to_printer",
        "description": "Send text to a connected printer for printing",
        "parameters": {
            "type": "object",
            "properties": {
                "printer_id": {
                    "type": "string",
                    "description": "Identifier of the printer",
                },
                "text": {"type": "string", "description": "Text to be printed"},
            },
            "required": ["printer_id", "text"],
        },
    },
    {
        "name": "read_out_loud",
        "description": "Read a provided text out loud using text-to-speech",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to be read out loud"}},
            "required": ["text"],
        },
    },
    {
        "name": "discord_get_channel_users",
        "description": "Get a list of users in a specific Discord channel",
        "parameters": {
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "Identifier of the Discord channel",
                }
            },
            "required": ["channel_id"],
        },
    },
    {
        "name": "discord_check_voice_status",
        "description": "Check the voice status of a specific Discord user",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Identifier of the Discord user",
                }
            },
            "required": ["user_id"],
        },
    },
    {
        "name": "discord_mute_user",
        "description": "Mute a user in a Discord channel",
        "parameters": {
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "Identifier of the Discord channel",
                },
                "user_id": {
                    "type": "string",
                    "description": "Identifier of the user to mute",
                },
            },
            "required": ["channel_id", "user_id"],
        },
    },
    {
        "name": "discord_send_voice_message",
        "description": "Send a text message as a voice message to a Discord user",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Identifier of the recipient Discord user",
                },
                "message": {
                    "type": "string",
                    "description": "Text message to be converted to voice",
                },
            },
            "required": ["user_id", "message"],
        },
    },
    {
        "name": "discord_send_private_message",
        "description": "Send a private message to a Discord user",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Identifier of the recipient Discord user",
                },
                "message": {"type": "string", "description": "Message to be sent"},
            },
            "required": ["user_id", "message"],
        },
    },
    {
        "name": "discord_send_channel_message",
        "description": "Send a message in a specific Discord channel",
        "parameters": {
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "Identifier of the Discord channel",
                },
                "message": {"type": "string", "description": "Message to be sent"},
            },
            "required": ["channel_id", "message"],
        },
    },
    {
        "name": "discord_generate_channel_summary",
        "description": "Generate a summary of the last N minutes of activity in a Discord text channel",
        "parameters": {
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "Identifier of the Discord channel",
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "Number of minutes to summarize",
                },
            },
            "required": ["channel_id", "duration_minutes"],
        },
    },
    {
        "name": "discord_start_speech_to_text",
        "description": "Start speech-to-text transcription in a given Discord channel",
        "parameters": {
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "Identifier of the Discord channel",
                }
            },
            "required": ["channel_id"],
        },
    },
    {
        "name": "discord_kick_user",
        "description": "Kick a specific user from a Discord server",
        "parameters": {
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Identifier of the Discord server",
                },
                "user_id": {
                    "type": "string",
                    "description": "Identifier of the user to kick",
                },
            },
            "required": ["server_id", "user_id"],
        },
    },
    {
        "name": "discord_invite_user_by_email",
        "description": "Invite a user to a Discord server via email",
        "parameters": {
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Identifier of the Discord server",
                },
                "email": {
                    "type": "string",
                    "description": "Email address of the user to invite",
                },
            },
            "required": ["server_id", "email"],
        },
    },
    {
        "name": "discord_check_user_connection",
        "description": "Check if a Discord user is currently connected",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Identifier of the Discord user",
                }
            },
            "required": ["user_id"],
        },
    },
    {
        "name": "gmail_send_email",
        "description": "Send an email using a Gmail account",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Email address of the recipient",
                },
                "subject": {"type": "string", "description": "Subject of the email"},
                "body": {"type": "string", "description": "Body content of the email"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "gmail_get_inbox",
        "description": "Retrieve a list of emails from the Gmail inbox",
        "parameters": {
            "type": "object",
            "properties": {
                "number_of_emails": {
                    "type": "integer",
                    "description": "Number of emails to retrieve",
                }
            },
            "required": ["number_of_emails"],
        },
    },
    {
        "name": "spotify_play_song",
        "description": "Play a specific song on Spotify",
        "parameters": {
            "type": "object",
            "properties": {
                "song_id": {
                    "type": "string",
                    "description": "Identifier of the song on Spotify",
                }
            },
            "required": ["song_id"],
        },
    },
    {
        "name": "spotify_create_playlist",
        "description": "Create a new playlist in Spotify",
        "parameters": {
            "type": "object",
            "properties": {
                "playlist_name": {
                    "type": "string",
                    "description": "Name of the new playlist",
                },
                "description": {
                    "type": "string",
                    "description": "Description of the new playlist",
                },
            },
            "required": ["playlist_name"],
        },
    },
    {
        "name": "twitter_post_tweet",
        "description": "Post a tweet on Twitter",
        "parameters": {
            "type": "object",
            "properties": {"content": {"type": "string", "description": "Content of the tweet"}},
            "required": ["content"],
        },
    },
    {
        "name": "twitter_get_user_timeline",
        "description": "Retrieve the tweet timeline of a specific Twitter user",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Identifier of the Twitter user",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of tweets to retrieve",
                },
            },
            "required": ["user_id", "count"],
        },
    },
    {
        "name": "uber_request_ride",
        "description": "Request a ride using Uber",
        "parameters": {
            "type": "object",
            "properties": {
                "pickup_location": {
                    "type": "string",
                    "description": "Pickup location for the ride",
                },
                "destination": {
                    "type": "string",
                    "description": "Destination of the ride",
                },
            },
            "required": ["pickup_location", "destination"],
        },
    },
    {
        "name": "netflix_search_show_or_movie",
        "description": "Search for a show or movie on Netflix",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for the show or movie",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "google_maps_get_directions",
        "description": "Get directions between two locations using Google Maps",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "Starting point for the directions",
                },
                "destination": {
                    "type": "string",
                    "description": "End point for the directions",
                },
                "mode_of_transport": {
                    "type": "string",
                    "enum": ["driving", "walking", "bicycling", "transit"],
                    "description": "Mode of transport for the directions",
                },
            },
            "required": ["origin", "destination"],
        },
    },
    {
        "name": "fitness_tracker_get_steps",
        "description": "Retrieve the number of steps walked from a fitness tracker",
        "parameters": {
            "type": "object",
            "properties": {
                "device_id": {
                    "type": "string",
                    "description": "Identifier of the fitness tracker",
                },
                "date": {
                    "type": "string",
                    "format": "date",
                    "description": "Date for which to retrieve steps",
                },
            },
            "required": ["device_id", "date"],
        },
    },
    {
        "name": "food_delivery_order_status",
        "description": "Check the status of a food delivery order",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Identifier of the food delivery order",
                }
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "smart_home_adjust_lighting",
        "description": "Adjust the lighting in a smart home environment",
        "parameters": {
            "type": "object",
            "properties": {
                "room": {
                    "type": "string",
                    "description": "Room where the lighting needs to be adjusted",
                },
                "brightness_level": {
                    "type": "integer",
                    "description": "Desired brightness level",
                },
            },
            "required": ["room", "brightness_level"],
        },
    },
    {
        "name": "flight_status_check",
        "description": "Check the status of a flight",
        "parameters": {
            "type": "object",
            "properties": {
                "flight_number": {
                    "type": "string",
                    "description": "Flight number to check status for",
                },
                "date": {
                    "type": "string",
                    "format": "date",
                    "description": "Date of the flight",
                },
            },
            "required": ["flight_number", "date"],
        },
    },
]
