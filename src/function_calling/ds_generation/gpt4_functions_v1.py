all_available_functions = [
    {
        "name": "news_service",
        "description": "Provide both the latest news headlines and a personalized news digest based on user preferences and selected categories",
        "parameters": {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "News categories for personalized digest",
                },
                "country": {
                    "type": "string",
                    "description": "Country for local news headlines",
                },
                "sourcePreferences": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Preferred news sources",
                },
            },
            "required": [],
        },
    },
    {
        "name": "reminder_service",
        "description": "Set various types of reminders including general tasks, wellbeing activities, and more, with the ability to categorize or specify the type of reminder",
        "parameters": {
            "type": "object",
            "properties": {
                "time": {"type": "string", "description": "The time for the reminder"},
                "message": {
                    "type": "string",
                    "description": "The message or description of the reminder",
                },
                "type": {
                    "type": "string",
                    "description": "Type of reminder (e.g., 'general', 'wellbeing')",
                },
            },
            "required": ["time", "message"],
        },
    },
    {
        "name": "translate_text",
        "description": "Translate a given text into a specified language",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to be translated"},
                "targetLanguage": {
                    "type": "string",
                    "description": "The target language code (e.g., 'en' for English, 'es' for Spanish)",
                },
            },
            "required": ["text", "targetLanguage"],
        },
    },
    {
        "name": "get_local_weather",
        "description": "Provide the current weather information for a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Latitude of the location",
                },
                "longitude": {
                    "type": "number",
                    "description": "Longitude of the location",
                },
            },
            "required": ["latitude", "longitude"],
        },
    },
    {
        "name": "create_image",
        "description": "Generate a digital image based on a textual description",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "A detailed description of the image to be created",
                },
                "style": {
                    "type": "string",
                    "description": "Optional. Specifies the style or artistic influence for the image",
                },
            },
            "required": ["description"],
        },
    },
    {
        "name": "calculate_expense",
        "description": "Calculate expenses based on entered data and provide budgeting advice",
        "parameters": {
            "type": "object",
            "properties": {
                "expenses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "amount": {
                                "type": "number",
                                "description": "Amount of the expense",
                            },
                            "category": {
                                "type": "string",
                                "description": "Category of the expense (e.g., 'groceries', 'utilities')",
                            },
                        },
                        "required": ["amount", "category"],
                    },
                    "description": "A list of expenses",
                }
            },
            "required": ["expenses"],
        },
    },
    {
        "name": "fitness_tracker",
        "description": "Track fitness activities and provide summaries and recommendations",
        "parameters": {
            "type": "object",
            "properties": {
                "activityType": {
                    "type": "string",
                    "description": "Type of activity (e.g., 'running', 'swimming')",
                },
                "duration": {
                    "type": "number",
                    "description": "Duration of the activity in minutes",
                },
                "intensity": {
                    "type": "string",
                    "description": "Intensity of the activity (e.g., 'low', 'medium', 'high')",
                },
            },
            "required": ["activityType", "duration"],
        },
    },
    {
        "name": "book_appointment",
        "description": "Book an appointment with service providers like doctors, salons, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "serviceType": {
                    "type": "string",
                    "description": "Type of service required (e.g., 'doctor', 'haircut')",
                },
                "provider": {
                    "type": "string",
                    "description": "Name of the provider or establishment",
                },
                "date": {
                    "type": "string",
                    "description": "Preferred date for the appointment",
                },
                "time": {
                    "type": "string",
                    "description": "Preferred time for the appointment",
                },
            },
            "required": ["serviceType", "provider", "date", "time"],
        },
    },
    {
        "name": "personal_finance_advice",
        "description": "Get advice on personal finance including savings, investments, and budgeting",
        "parameters": {
            "type": "object",
            "properties": {
                "income": {"type": "number", "description": "Monthly or annual income"},
                "expenses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "amount": {
                                "type": "number",
                                "description": "Amount of the expense",
                            },
                            "type": {
                                "type": "string",
                                "description": "Type of expense (e.g., 'rent', 'food', 'utilities')",
                            },
                        },
                        "required": ["amount", "type"],
                    },
                    "description": "List of monthly or annual expenses",
                },
                "goals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Financial goals (e.g., 'save for a car', 'plan a vacation')",
                },
            },
            "required": ["income", "expenses"],
        },
    },
    {
        "name": "daily_planner",
        "description": "Organize daily tasks and schedules",
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "Description of the task",
                            },
                            "time": {
                                "type": "string",
                                "description": "Scheduled time for the task",
                            },
                        },
                        "required": ["task", "time"],
                    },
                    "description": "List of tasks and their scheduled times",
                }
            },
            "required": ["tasks"],
        },
    },
    {
        "name": "language_learning",
        "description": "Provide language learning exercises, tips, and progress tracking",
        "parameters": {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "description": "The language to learn or improve",
                },
                "level": {
                    "type": "string",
                    "description": "Current proficiency level in the language (e.g., 'beginner', 'intermediate')",
                },
                "goal": {
                    "type": "string",
                    "description": "Specific learning goal or area of focus (e.g., 'vocabulary', 'conversation')",
                },
            },
            "required": ["language", "level"],
        },
    },
    {
        "name": "event_ticketing",
        "description": "Find and book tickets for events like concerts, theater, or sports",
        "parameters": {
            "type": "object",
            "properties": {
                "eventType": {
                    "type": "string",
                    "description": "Type of event (e.g., 'concert', 'sports', 'theater')",
                },
                "date": {
                    "type": "string",
                    "description": "Preferred date or date range for the event",
                },
                "location": {
                    "type": "string",
                    "description": "City or venue for the event",
                },
            },
            "required": ["eventType", "date", "location"],
        },
    },
    {
        "name": "project_tracker",
        "description": "Oversee home renovation projects including timeline, milestones, and task delegation",
        "parameters": {
            "type": "object",
            "properties": {
                "projectName": {
                    "type": "string",
                    "description": "Name of the renovation project",
                },
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "taskName": {
                                "type": "string",
                                "description": "Name of the task",
                            },
                            "deadline": {
                                "type": "string",
                                "description": "Deadline for the task",
                            },
                            "assignedTo": {
                                "type": "string",
                                "description": "Person or contractor assigned to the task",
                            },
                        },
                        "required": ["taskName", "deadline"],
                    },
                    "description": "List of tasks associated with the project",
                },
            },
            "required": ["projectName", "tasks"],
        },
    },
    {
        "name": "educational_content_recommender",
        "description": "Suggest study materials, online courses, and resources based on the user's academic or personal learning goals",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Subject or field of study",
                },
                "level": {
                    "type": "string",
                    "description": "Academic or proficiency level",
                },
                "focusAreas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific areas within the subject to focus on",
                },
            },
            "required": ["subject", "level"],
        },
    },
    {
        "name": "health_monitor",
        "description": "Track and analyze key health parameters, providing alerts and health advice based on the data",
        "parameters": {
            "type": "object",
            "properties": {
                "parametersTracked": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of health parameters being monitored (e.g., 'blood pressure', 'sugar levels')",
                },
                "frequency": {
                    "type": "string",
                    "description": "Frequency of monitoring (e.g., 'daily', 'weekly')",
                },
            },
            "required": ["parametersTracked", "frequency"],
        },
    },
    {
        "name": "career_advisory",
        "description": "Provide career development advice including job alerts, course recommendations, and networking opportunities",
        "parameters": {
            "type": "object",
            "properties": {
                "industry": {
                    "type": "string",
                    "description": "User's industry or field of interest",
                },
                "experienceLevel": {
                    "type": "string",
                    "description": "User's current career level or years of experience",
                },
                "goals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Career goals or objectives",
                },
            },
            "required": ["industry", "experienceLevel"],
        },
    },
    {
        "name": "sustainability_tips",
        "description": "Provide daily tips and recommendations for living a sustainable and eco-friendly lifestyle",
        "parameters": {
            "type": "object",
            "properties": {
                "interestAreas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Areas of interest in sustainability (e.g., 'energy', 'waste reduction')",
                }
            },
            "required": ["interestAreas"],
        },
    },
    {
        "name": "fashion_curator",
        "description": "Suggest fashion items and accessories based on user's taste, budget, and the latest trends",
        "parameters": {
            "type": "object",
            "properties": {
                "stylePreferences": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "User's style preferences or desired looks",
                },
                "budget": {"type": "number", "description": "Budget for new items"},
                "occasion": {
                    "type": "string",
                    "description": "Specific occasion or purpose for the outfit (e.g., 'casual', 'business', 'event')",
                },
            },
            "required": ["stylePreferences"],
        },
    },
    {
        "name": "safety_monitor",
        "description": "Monitor safety and well-being of individuals, alerting caregivers or emergency services if needed",
        "parameters": {
            "type": "object",
            "properties": {
                "monitoringType": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Types of monitoring required (e.g., 'fall detection', 'activity levels')",
                },
                "alertRecipients": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Contacts for emergency or alert notifications",
                },
            },
            "required": ["monitoringType", "alertRecipients"],
        },
    },
    {
        "name": "commuting_assistant",
        "description": "Provide real-time traffic updates, route recommendations, and contextual commuting advice based on user's velocity, usual locations, and preferences",
        "parameters": {
            "type": "object",
            "properties": {
                "currentLocation": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number"},
                        "longitude": {"type": "number"},
                    },
                    "description": "User's current location",
                },
                "destination": {
                    "type": "string",
                    "description": "Intended destination",
                },
                "preferences": {
                    "type": "object",
                    "properties": {
                        "modeOfTransport": {"type": "string"},
                        "usualRoutes": {"type": "array", "items": {"type": "string"}},
                    },
                    "description": "Commuting preferences including mode of transport and usual routes",
                },
            },
            "required": ["currentLocation"],
        },
    },
    {
        "name": "diet_management",
        "description": "Suggest recipes, manage dietary preferences and restrictions, and integrate with grocery list for holistic diet planning",
        "parameters": {
            "type": "object",
            "properties": {
                "dietaryRestrictions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of dietary restrictions (e.g., 'gluten-free', 'vegan')",
                },
                "mealType": {
                    "type": "string",
                    "description": "Type of meal (e.g., 'breakfast', 'lunch', 'dinner')",
                },
                "ingredientsAvailable": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ingredients available for recipe suggestions",
                },
            },
            "required": [],
        },
    },
    {
        "name": "errand_optimizer",
        "description": "Optimize the order and timing of errands based on location and personal preferences",
        "parameters": {
            "type": "object",
            "properties": {
                "errands": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Location of the errand",
                            },
                            "type": {
                                "type": "string",
                                "description": "Type of errand (e.g., 'grocery', 'post office')",
                            },
                        },
                        "required": ["location", "type"],
                    },
                    "description": "List of errands to complete",
                },
                "currentLocation": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number"},
                        "longitude": {"type": "number"},
                    },
                    "description": "User's current location",
                },
            },
            "required": ["errands", "currentLocation"],
        },
    },
    {
        "name": "personal_alert_system",
        "description": "Monitor user's safety and send alerts based on location, time, and other safety parameters",
        "parameters": {
            "type": "object",
            "properties": {
                "safetyParameters": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of safety parameters to monitor (e.g., 'location', 'time')",
                },
                "alertRecipients": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Contacts for emergency or alert notifications",
                },
            },
            "required": ["safetyParameters", "alertRecipients"],
        },
    },
    {
        "name": "travel_diary",
        "description": "Compile and organize travel data into a journal with photos, notes, and weather logs",
        "parameters": {
            "type": "object",
            "properties": {
                "trips": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Travel location",
                            },
                            "date": {"type": "string", "description": "Date of travel"},
                            "notes": {
                                "type": "string",
                                "description": "Personal notes or diary entry",
                            },
                        },
                        "required": ["location", "date"],
                    },
                    "description": "List of trips and associated details",
                }
            },
            "required": ["trips"],
        },
    },
    {
        "name": "home_automation",
        "description": "Integrate and control home devices based on user's proximity, schedule, and preferences",
        "parameters": {
            "type": "object",
            "properties": {
                "deviceType": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Types of home devices to control (e.g., 'thermostat', 'lights')",
                },
                "controlPreferences": {
                    "type": "object",
                    "properties": {
                        "time": {"type": "string"},
                        "location": {"type": "string"},
                        "userSetting": {"type": "string"},
                    },
                    "description": "Preferences for how and when to control each device",
                },
            },
            "required": ["deviceType", "controlPreferences"],
        },
    },
    {
        "name": "study_progress_tracker",
        "description": "Monitor and update on the user's learning progress, suggesting areas for improvement",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "The subject or topic being studied",
                },
                "goals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific study goals or milestones",
                },
                "progressMetrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Metrics or indicators of progress (e.g., quiz scores, chapters completed)",
                },
            },
            "required": ["subject", "goals"],
        },
    },
    {
        "name": "gardening_tips_provider",
        "description": "Offer personalized gardening advice and care tips based on plant types and local weather conditions",
        "parameters": {
            "type": "object",
            "properties": {
                "plantTypes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Types of plants in the garden",
                },
                "localWeather": {
                    "type": "object",
                    "properties": {
                        "temperature": {"type": "number"},
                        "humidity": {"type": "number"},
                        "precipitation": {"type": "number"},
                    },
                    "description": "Current local weather conditions",
                },
            },
            "required": ["plantTypes", "localWeather"],
        },
    },
    {
        "name": "pet_care_scheduler",
        "description": "Create and manage a personalized pet care routine including feeding, exercise, and vet appointments",
        "parameters": {
            "type": "object",
            "properties": {
                "petType": {
                    "type": "string",
                    "description": "Type of pet (e.g., 'dog', 'cat')",
                },
                "careTasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "schedule": {"type": "string"},
                        },
                        "description": "Specific care tasks and their schedule",
                    },
                },
            },
            "required": ["petType", "careTasks"],
        },
    },
    {
        "name": "interactive_storyteller",
        "description": "Create and narrate interactive and educational stories based on child's age and preferences",
        "parameters": {
            "type": "object",
            "properties": {
                "childAge": {"type": "number", "description": "Age of the child"},
                "interests": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Child's interests or favorite topics",
                },
                "educationalGoals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Educational goals or learning areas",
                },
            },
            "required": ["childAge", "interests"],
        },
    },
    {
        "name": "grocery_list_manager",
        "description": "Manage and predict grocery needs, suggesting recipes and reminding when to restock",
        "parameters": {
            "type": "object",
            "properties": {
                "pantryItems": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "item": {"type": "string"},
                            "quantity": {"type": "number"},
                        },
                    },
                    "description": "Current items and their quantities in the pantry",
                },
                "restockAlerts": {
                    "type": "boolean",
                    "description": "Whether to alert when items are low and need restocking",
                },
            },
            "required": ["pantryItems"],
        },
    },
    {
        "name": "audio_entertainment_curator",
        "description": "Analyze preferences and context to suggest music and podcast lists",
        "parameters": {
            "type": "object",
            "properties": {
                "mood": {"type": "string", "description": "Current mood of the user"},
                "activity": {
                    "type": "string",
                    "description": "Current or planned activity (e.g., 'working', 'exercising')",
                },
                "preferences": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Musical genres or podcast types preferred",
                },
            },
            "required": ["mood", "activity"],
        },
    },
    {
        "name": "vacation_planner",
        "description": "Manage all aspects of vacation planning from transportation to activities",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": "Vacation destination",
                },
                "travelDates": {"type": "string", "description": "Dates of the trip"},
                "preferences": {
                    "type": "object",
                    "properties": {
                        "accommodation": {"type": "string"},
                        "activities": {"type": "string"},
                    },
                    "description": "Travel preferences including accommodation and activities",
                },
            },
            "required": ["destination", "travelDates"],
        },
    },
    {
        "name": "digital_legacy_organizer",
        "description": "Assist in organizing, backing up, and planning for digital assets and online presence",
        "parameters": {
            "type": "object",
            "properties": {
                "assets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Types of digital assets to manage (e.g., 'social media', 'digital files')",
                },
                "backupPlan": {
                    "type": "boolean",
                    "description": "Whether to create a plan for regular backups",
                },
            },
            "required": ["assets"],
        },
    },
    {
        "name": "special_event_tracker",
        "description": "Monitor important personal dates and provide suggestions for celebrations or gifts",
        "parameters": {
            "type": "object",
            "properties": {
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "eventType": {"type": "string"},
                            "date": {"type": "string"},
                        },
                    },
                    "description": "List of special events and their dates",
                },
                "reminderLeadTime": {
                    "type": "number",
                    "description": "How many days in advance to provide reminders",
                },
            },
            "required": ["events", "reminderLeadTime"],
        },
    },
]

print([f["name"] for f in all_available_functions])
