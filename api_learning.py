from openai import OpenAI
import os

client = OpenAI()

response = client.responses.create(
    model="gpt-5.6-luna",
    input="Elplain APIs in one sentence",
)

print(response.output_text)

response.text
