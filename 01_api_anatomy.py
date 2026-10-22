from openai import OpenAI
import time
import argparse

client = OpenAI()

start = time.perf_counter()

parser = argparse.ArgumentParser(
    description="Send prompts to OpenAI using responses API"
)

parser.add_argument("input_text", help="Enter a prompt to send to the model")

args = parser.parse_args()

input_text = args.input_text

response = client.responses.create(
    model='gpt-5.6-luna',
    input= input_text
)

elapsed = time.perf_counter() - start


out_text = response.output_text
out_model = response.model
out_status = response.status
usage = response.usage
rid = response._request_id


print("ELAPSED TIME")
print(f"{elapsed:.2f} seconds")
print(f"Output text:\n{out_text}")
print(f"model used:\n{out_model}")
print(f"status:\n{out_status}")
print(f"usage:\n{usage}")
print(f"Request id:\n{rid}")
print("FULL RESPONSE JSON:\n")
print(response.model_dump_json(indent=2))

