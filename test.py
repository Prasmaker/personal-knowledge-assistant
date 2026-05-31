from dotenv import load_dotenv
import os
from groq import Groq

#loading the .env file first and create an openai client
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

#SEND and check a sample message
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",         # the model 
    messages=[
        {"role": "user", "content": "Say hello in one sentence."}
    ]
)

print(response.choices[0].message.content)

