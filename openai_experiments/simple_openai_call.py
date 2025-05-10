from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "Where is Tamriel?"}
  ]
)

print(completion.choices[0].message);