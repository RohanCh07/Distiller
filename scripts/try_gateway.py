"""Send a few Banking77-style requests through the gateway and print who answered.

Usage (with the gateway running):  python scripts/try_gateway.py
Needs: pip install openai
"""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")  # API key is read from OPENAI_API_KEY

MESSAGES = [
    "My card still hasn't arrived, it's been 2 weeks",
    "How do I top up using Apple Pay?",
    "I was charged twice for the same coffee",
    "What is the exchange rate you use?",
    "I forgot my passcode",
]

for text in MESSAGES:
    raw = client.chat.completions.with_raw_response.create(
        model="gpt-4o-mini",  # change to the teacher model you use
        messages=[
            {
                "role": "system",
                "content": "Classify the bank customer's message into exactly one intent. Reply with the intent name only.",
            },
            {"role": "user", "content": text},
        ],
        extra_headers={"X-Distiller-Task": "banking77"},
    )
    reply = raw.parse()
    print(
        f"{raw.headers.get('x-distiller-route', '?'):8}  {reply.choices[0].message.content:35}  <- {text}"
    )
