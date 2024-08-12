from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from .part_finder_template import PartDescription, part_finder_template

load_dotenv()

llm = ChatOpenAI(model = 'gpt-4-turbo-preview')

part_finder_agent = part_finder_template | llm.bind_tools(tools = [PartDescription],
tool_choice = 'PartDescription')



if __name__ == "__main__":
    input_text = """
    So I think that my anger is a byproduct of fear.  And I think that the fear may be worth digging into.  And I have found that anxiety has become more pronounced and this vague sense of ease that something bad may or will happen, is something that I've 
    noticed more and more as depression has become less of an ongoing battle for me.  And I would like to explore that.
    """

    bad_input_text = """
    Well I really need my kids to help me.  They always shirk their responsibilities.  And I don't know why.
    """
    human_message = HumanMessage(
        content=bad_input_text
    )
    res = part_finder_agent.invoke(input = {'messages': [human_message]})