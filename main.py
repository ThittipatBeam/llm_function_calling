import os
from dotenv import load_dotenv
import requests
from typing import List

from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from pydantic import BaseModel, Field

from huggingface_hub import login

load_dotenv()

token = os.getenv('HUGGINGFACEHUB_API_TOKEN', '')
weather_api = os.getenv('OPEN_WEATHER_API' , '')
llm_model = os.getenv('LLM_MODEL', '')

login(token=token)

def get_current_weather(city_name: str):

    """
    Create a API call to get the current weather base on city name

    Example call:
        get_current_weather("bangkok")
    
    Args:
        city_name : The city name that user want to get the current weather from 

    Returns:
        str: The API response of the current weather on the specific city or an error message if the API call threw an error
    """

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={weather_api}&units=metric"

    try:

        api_response = requests.get(url)
        print(api_response.content)
        return "API collect weather data successfully!"
    
    except Exception as e:

        print(f"An error occurred: {e}")


# @st.cache_resource
def load_local_model():
        
    return HuggingFacePipeline.from_model_id(
            model_id=llm_model,
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 1024,
                "cache_dir": "/mnt/c/Users/beam_/Desktop/llm_function-calling/model"
                # "top_k": 50,
                # "temperature": 0.4
            },
        )


llm = load_local_model()


available_tools = {
    "get current weather": get_current_weather
}

tool_descriptions = [f"{name}:\n{func.__doc__}\n\n" for name, func in available_tools.items()]


class ToolCall(BaseModel):
    name: str = Field(description= "Name of the function to run")
    args: dict = Field(description= "Arguments for the function call (empty if no arguments are needed for the tool call)")

class ToolCallOrResponse(BaseModel):
    tool_calls: List[ToolCall] = Field(description="List of tool calls, empty array if you don't need to invoke a tool")
    content: str = Field(description="Response to the user if a tool doesn't need to be invoked")


tool_text = f"""
You always respond with a JSON object that has two required keys.

tool_calls: List[ToolCall] = Field(description="List of tool calls, empty array if you don't need to invoke a tool")
content: str = Field(description="Response to the user if a tool doesn't need to be invoked")

Here is the type for ToolCall (object with two keys):
    name: str = Field(description="Name of the function to run (NA if you don't need to invoke a tool)")
    args: dict = Field(description="Arguments for the function call (empty array if you don't need to invoke a tool or if no arguments are needed for the tool call)")

Don't start your answers with "Here is the JSON response", just give the JSON.

The tools you have access to are:

{"".join(tool_descriptions)}

Any message that starts with "Thought:" is you thinking to yourself. This isn't told to the user so you still need to communicate what you did with them.
Don't repeat an action. If a thought tells you that you already took an action for a user, don't do it again.
"""  


def prompt_ai(messages, nested_calls=0, invoked_tools=[]):

    if nested_calls > 3:
        raise Exception("Failsafe - AI is failing too much!")
    
    # First, prompt the AI with the latest user message
    parser = JsonOutputParser(pydantic_object=ToolCallOrResponse)
    chatbot = ChatHuggingFace(llm=llm) | parser

    try:
        ai_response = chatbot.invoke(messages)
    except:
        return prompt_ai(messages, nested_calls + 1)
    
    print(ai_response)



if __name__ == "__main__":

    a = prompt_ai(messages="สวัสดีครับ")
    print(a)
    print("beam")



