all_available_functions = [
    {
        "name": "get_exchange_rate",
        "description": "Get the exchange rate between two currencies",
        "parameters": {
            "type": "object",
            "properties": {
                "base_currency": {
                    "type": "string",
                    "description": "The currency to convert from",
                },
                "target_currency": {
                    "type": "string",
                    "description": "The currency to convert to",
                },
            },
            "required": ["base_currency", "target_currency"],
        },
    },
    {
        "name": "get_news_headlines",
        "description": "Get the latest news headlines",
        "parameters": {
            "type": "object",
            "properties": {
                "country": {
                    "type": "string",
                    "description": "The country for which to fetch news",
                }
            },
            "required": ["country"],
        },
    },
    {
        "name": "generate_password",
        "description": "Generate a random password",
        "parameters": {
            "type": "object",
            "properties": {
                "length": {
                    "type": "integer",
                    "description": "The length of the password",
                },
                "include_symbols": {
                    "type": "boolean",
                    "description": "Whether to include symbols in the password",
                },
            },
            "required": ["length"],
        },
    },
    {
        "name": "create_task",
        "description": "Create a new task in a task management system",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The title of the task"},
                "due_date": {
                    "type": "string",
                    "format": "date",
                    "description": "The due date of the task",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "The priority of the task",
                },
            },
            "required": ["title", "due_date", "priority"],
        },
    },
    {
        "name": "calculate_median",
        "description": "Calculate the median of a list of numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "A list of numbers",
                }
            },
            "required": ["numbers"],
        },
    },
    {
        "name": "calculate_loan_payment",
        "description": "Calculate the monthly payment for a loan",
        "parameters": {
            "type": "object",
            "properties": {
                "principal": {
                    "type": "number",
                    "description": "The principal amount of the loan",
                },
                "interest_rate": {
                    "type": "number",
                    "description": "The annual interest rate for the loan",
                },
                "loan_term": {
                    "type": "integer",
                    "description": "The term of the loan in years",
                },
            },
            "required": ["principal", "interest_rate", "loan_term"],
        },
    },
    {
        "name": "convert_temperature",
        "description": "Convert temperature from one unit to another",
        "parameters": {
            "type": "object",
            "properties": {
                "temperature": {
                    "type": "number",
                    "description": "The temperature value",
                },
                "from_unit": {
                    "type": "string",
                    "description": "The unit to convert from",
                },
                "to_unit": {"type": "string", "description": "The unit to convert to"},
            },
            "required": ["temperature", "from_unit", "to_unit"],
        },
    },
    {
        "name": "get_current_date",
        "description": "Get the current date",
        "parameters": {},
    },
    {
        "name": "get_movie_details",
        "description": "Get details of a movie",
        "parameters": {
            "type": "object",
            "properties": {
                "movie_title": {
                    "type": "string",
                    "description": "The title of the movie",
                }
            },
            "required": ["movie_title"],
        },
    },
    {
        "name": "calculate_tip",
        "description": "Calculate the tip amount for a bill",
        "parameters": {
            "type": "object",
            "properties": {
                "bill_amount": {
                    "type": "number",
                    "description": "The total bill amount",
                },
                "tip_percentage": {
                    "type": "number",
                    "description": "The percentage of tip",
                },
            },
            "required": ["bill_amount", "tip_percentage"],
        },
    },
    {
        "name": "track_calories",
        "description": "Track daily calorie intake",
        "parameters": {
            "type": "object",
            "properties": {
                "meal": {
                    "type": "string",
                    "description": "The meal for which calories are being tracked",
                },
                "calories": {
                    "type": "number",
                    "description": "The number of calories consumed",
                },
                "date": {
                    "type": "string",
                    "format": "date",
                    "description": "The date for which calories are being tracked",
                },
            },
            "required": ["meal", "calories", "date"],
        },
    },
    {
        "name": "create_contact",
        "description": "Create a new contact",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The name of the contact"},
                "email": {
                    "type": "string",
                    "description": "The email address of the contact",
                },
            },
            "required": ["name", "email"],
        },
    },
    {
        "name": "create_calendar_event",
        "description": "Create a new calendar event",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The title of the event"},
                "start_time": {
                    "type": "string",
                    "description": "The start time of the event in the format YYYY-MM-DD HH:MM",
                },
                "end_time": {
                    "type": "string",
                    "description": "The end time of the event in the format YYYY-MM-DD HH:MM",
                },
            },
            "required": ["title", "start_time", "end_time"],
        },
    },
    {
        "name": "search_movie",
        "description": "Search for a movie by title",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The title of the movie"}
            },
            "required": ["title"],
        },
    },
    {
        "name": "get_stock_price",
        "description": "Get the current stock price",
        "parameters": {
            "type": "object",
            "properties": {
                "stock_symbol": {
                    "type": "string",
                    "description": "The symbol of the stock, e.g. AAPL",
                }
            },
            "required": ["stock_symbol"],
        },
    },
    {
        "name": "calculate_age",
        "description": "Calculate the age of a person",
        "parameters": {
            "type": "object",
            "properties": {
                "birth_date": {
                    "type": "string",
                    "description": "The birth date of the person",
                }
            },
            "required": ["birth_date"],
        },
    },
    {
        "name": "calculate_discounted_price",
        "description": "Calculate the discounted price based on a given original price and discount percentage",
        "parameters": {
            "type": "object",
            "properties": {
                "original_price": {
                    "type": "number",
                    "description": "The original price before discount",
                },
                "discount_percentage": {
                    "type": "number",
                    "description": "The percentage of the discount",
                },
            },
            "required": ["original_price", "discount_percentage"],
        },
    },
    {
        "name": "encrypt_text",
        "description": "Encrypt the given text",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to be encrypted"},
                "encryption_algorithm": {
                    "type": "string",
                    "description": "The encryption algorithm to be used",
                },
            },
            "required": ["text", "encryption_algorithm"],
        },
    },
    {
        "name": "search_books",
        "description": "Search for books based on title, author, or genre",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The title of the book"},
                "author": {"type": "string", "description": "The author of the book"},
                "genre": {"type": "string", "description": "The genre of the book"},
            },
            "required": [],
        },
    },
    {
        "name": "calculate_distance",
        "description": "Calculate the distance between two points",
        "parameters": {
            "type": "object",
            "properties": {
                "lat1": {"type": "number", "description": "The latitude of point 1"},
                "lon1": {"type": "number", "description": "The longitude of point 1"},
                "lat2": {"type": "number", "description": "The latitude of point 2"},
                "lon2": {"type": "number", "description": "The longitude of point 2"},
            },
            "required": ["lat1", "lon1", "lat2", "lon2"],
        },
    },
    {
        "name": "play_music",
        "description": "Play music from a specific genre or artist",
        "parameters": {
            "type": "object",
            "properties": {
                "genre": {
                    "type": "string",
                    "description": "The genre of music to play",
                },
                "artist": {
                    "type": "string",
                    "description": "The artist whose music to play",
                },
            },
            "required": ["genre", "artist"],
        },
    },
    {
        "name": "get_lyrics",
        "description": "Get the lyrics of a song",
        "parameters": {
            "type": "object",
            "properties": {
                "song": {"type": "string", "description": "The name of the song"},
                "artist": {"type": "string", "description": "The artist of the song"},
            },
            "required": ["song", "artist"],
        },
    },
    {
        "name": "calculate_bmi",
        "description": "Calculate the BMI (Body Mass Index)",
        "parameters": {
            "type": "object",
            "properties": {
                "weight": {
                    "type": "number",
                    "description": "The weight of the person in kilograms",
                },
                "height": {
                    "type": "number",
                    "description": "The height of the person in meters",
                },
            },
            "required": ["weight", "height"],
        },
    },
    {
        "name": "check_email_availability",
        "description": "Check if an email address is available or already taken",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The email address to be checked",
                }
            },
            "required": ["email"],
        },
    },
    {
        "name": "calculate_gpa",
        "description": "Calculate the GPA (Grade Point Average)",
        "parameters": {
            "type": "object",
            "properties": {
                "grades": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "course": {
                                "type": "string",
                                "description": "The name of the course",
                            },
                            "grade": {
                                "type": "string",
                                "description": "The grade received in the course",
                            },
                            "credit_hours": {
                                "type": "number",
                                "description": "The number of credit hours for the course",
                            },
                        },
                        "required": ["course", "grade", "credit_hours"],
                    },
                }
            },
            "required": ["grades"],
        },
    },
    {
        "name": "get_definition",
        "description": "Get the definition of a word",
        "parameters": {
            "type": "object",
            "properties": {
                "word": {
                    "type": "string",
                    "description": "The word to get the definition of",
                }
            },
            "required": ["word"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {
                    "type": "string",
                    "description": "The email address of the recipient",
                },
                "subject": {
                    "type": "string",
                    "description": "The subject of the email",
                },
                "message": {
                    "type": "string",
                    "description": "The content of the email",
                },
            },
            "required": ["recipient", "subject", "message"],
        },
    },
    {
        "name": "analyze_website",
        "description": "Analyze the content and performance of a website",
        "parameters": {
            "type": "object",
            "properties": {
                "website_url": {
                    "type": "string",
                    "description": "The URL of the website to analyze",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The metrics to analyze for the website",
                },
            },
            "required": ["website_url", "metrics"],
        },
    },
    {
        "name": "generate_qr_code",
        "description": "Generate a QR code for the given data",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "The data to be encoded in the QR code",
                }
            },
            "required": ["data"],
        },
    },
]
