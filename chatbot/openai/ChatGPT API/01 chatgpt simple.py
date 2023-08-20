import openai

openai.api_key = "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"

completion = openai.ChatCompletion.create(model="text-davinci-003", messages=[{"role": "user", "content": "Give me 3 ideas for apps I could build with openai apis "}])
print(completion.choices[0].message.content)