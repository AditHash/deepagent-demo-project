
from deepagents import create_deep_agent

import os
from dotenv import load_dotenv
load_dotenv()


API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "openai:gpt-5-nano")

# OpenAI's built-in web search — no extra install or API key needed
internet_search = {"type": "web_search"}



# System prompt to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
other wise just normal chat. 
"""

agent = create_deep_agent(
    model=MODEL_NAME,
    tools=[internet_search],
    system_prompt=research_instructions,
)

# agent = create_deep_agent(
#     model=MODEL_NAME,
#     system_prompt="You are a research assistant.",
# )
result = agent.invoke({"messages": "Hi"})
print(result)