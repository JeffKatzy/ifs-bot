from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from .response_part_finder_template import (ResponsePartDescription,
                                            part_finder_template)

load_dotenv()

llm = ChatOpenAI(model = 'gpt-4-turbo-preview')

response_part_finder_agent = part_finder_template | llm.bind_tools(tools = [ResponsePartDescription],
tool_choice = 'ResponsePartDescription')


if __name__ == "__main__":
    input_text = """Well if I'm being honest I would like it to go away.  And I am quite annoyed with it, because it gets in the way of my happiness a lot."""

    human_message = HumanMessage(
        content=input_text
    )
    res = response_part_finder_agent.invoke(input = {'messages': [human_message]})
