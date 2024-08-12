from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from .symptom_finder_template import (SymptomDescription,
                                      symptom_finder_template)

load_dotenv()

llm = ChatOpenAI(model = 'gpt-4-turbo-preview')

symptom_finder_agent = symptom_finder_template | llm.bind_tools(tools = [SymptomDescription],
tool_choice = 'SymptomDescription')


if __name__ == "__main__":
    input_text = """Right now, as I am thinking about it, I find it in the throat, and that is quite common, like a constriction in the throat.  Also will feel it basically right over the heart, like a constriction in the left side of my chest."""

    human_message = HumanMessage(
        content=input_text
    )
    res = symptom_finder_agent.invoke(input = {'messages': [human_message]})
    
