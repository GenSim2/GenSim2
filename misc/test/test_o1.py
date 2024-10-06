# from openai import OpenAI
import openai

# client = OpenAI()

response = openai.ChatCompletion.create(
    model="gpt-4-1106-preview",
    messages=[
        {
            "role": "user",
            "content": "Write a bash script that takes a matrix represented as a string with format '[1,2],[3,4],[5,6]' and prints the transpose in the same format.",
        }
    ],
)
import ipdb

ipdb.set_trace()
print(response.choices[0].message.content)
