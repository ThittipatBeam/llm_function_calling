import os
from dotenv import load_dotenv
import requests
from typing import List
import json
from datetime import datetime

import streamlit as st

import torch
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage

from pydantic import BaseModel, Field

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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
        return api_response.content
    
    except Exception as e:

        print(f"An error occurred: {e}")


@st.cache_resource
def load_local_model():

    # return HuggingFaceEndpoint(
    #         repo_id=llm_model,
    #         task="text-generation",
    #         max_new_tokens=1024,
    #         do_sample=False
    #     )

    # return HuggingFacePipeline.from_model_id(
    #         model_id=llm_model,
    #         task="text-generation",
    #         pipeline_kwargs={
    #             "max_new_tokens": 1024,
    #             "cache_dir": "/mnt/c/Users/beam_/Desktop/llm_function-calling/model"
    #             # "top_k": 50,
    #             # "temperature": 0.4
    #         },
    #     )

    os.getenv("CUDA_VISIBLE_DEVICES", '')

    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    model = AutoModelForCausalLM.from_pretrained(llm_model, device_map="auto")
    pipe = pipeline("text-generation",
                    model=model, 
                    tokenizer=tokenizer,
                    max_new_tokens=1024,
                    return_full_text=False
                    )
    hf = HuggingFacePipeline(pipeline=pipe)
    
    return hf


llm = load_local_model()


available_tools = {
    "get_current_weather": get_current_weather
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

    has_tool_calls = len(ai_response["tool_calls"]) > 0
    if has_tool_calls:
        # Next, for each tool the AI wanted to call, call it and add the tool result to the list of messages
        for tool_call in ai_response["tool_calls"]:
            if str(tool_call) not in invoked_tools:
                tool_name = tool_call["name"].lower()
                selected_tool = available_tools[tool_name]
                tool_output = selected_tool(**tool_call["args"])

                messages.append(HumanMessage(content=f"Thought: - I called {tool_name} with args {tool_call['args']} and got back: {tool_output}."))  
                invoked_tools.append(str(tool_call))
            else:
                return ai_response          

        # Prompt the AI again now that the result of calling the tool(s) has been added to the chat history
        return prompt_ai(messages, nested_calls + 1, invoked_tools)

    return ai_response


def prompt_ai_test(messages):

    chatbot = ChatHuggingFace(llm=llm)
    ai_response = chatbot.invoke(messages)

    print(ai_response)


def main():
    st.title("Weather Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=f"You are a personal assistant who helps get the current weather base on the user input. The current date is: {datetime.now().date()}.\n{tool_text}")
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        message_json = json.loads(message.json())
        message_type = message_json["type"]
        message_content = message_json["content"]
        if message_type in ["human", "ai", "system"] and not message_content.startswith("Thought:"):
            with st.chat_message(message_type):
                st.markdown(message_content)        

    # React to user input
    if prompt := st.chat_input("What would you like to do today?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append(HumanMessage(content=prompt))

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            ai_response = prompt_ai(st.session_state.messages)
            st.markdown(ai_response['content'])
        
        st.session_state.messages.append(AIMessage(content=ai_response['content']))
    

if __name__ == "__main__":
    main()

