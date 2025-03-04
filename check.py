import google.generativeai as genai
import os
import google.generativeai as genai

genai.configure(api_key="AIzaSyA1OATmIt66nzqtg3aqZHgO2fPezkcsQTQ")

# List available models
for model in genai.list_models():
    print(model.name)