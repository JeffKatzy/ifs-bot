import json

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

from src.agents.part_finder import PartDescription, part_finder_agent
from src.agents.response_part_finder import (ResponsePartDescription,
                                             response_part_finder_agent)
from src.agents.symptom_finder import SymptomDescription, symptom_finder_agent

load_dotenv()

class Node:
    def parse_part(self, state):
        messages = state["messages"]
        response = part_finder_agent.invoke(messages)
        parser = PydanticToolsParser(tools=[PartDescription])
        part_description = parser.invoke(response)[0]
        content = json.dumps(part_description.dict())
        tool_message = ToolMessage(content = content, tool_call_id = response.tool_calls[0]['id'])
        ai_message = self._get_part_finder_ai_message(part_description)
        return {"messages": [tool_message, ai_message]}

    def get_symptom(self, state):
        return {'messages': []}

    def get_responding_part(self, state):
        return {'messages': []}

    def parse_symptom(self, state):
        messages = state["messages"]
        human_symptom_message = messages[-1:]
        response = symptom_finder_agent.invoke(human_symptom_message)
        parser = PydanticToolsParser(tools=[SymptomDescription])
        symptom_description = parser.invoke(response)[0]
        content = json.dumps(symptom_description.dict())
        tool_message = ToolMessage(content = content, tool_call_id = response.tool_calls[0]['id'])
        return {"messages": [tool_message]}

    def parse_responding_parts(self, state):
        messages = state["messages"]
        response_to_part = messages[-1:]
        response = response_part_finder_agent.invoke(response_to_part)    
        parser = PydanticToolsParser(tools=[ResponsePartDescription])
        response_part_description = parser.invoke(response)[0]
        content = json.dumps(response_part_description.dict())
        tool_message = ToolMessage(content = content, tool_call_id = response.tool_calls[0]['id'])
        content = response_part_description.sympathize_with_responding_parts
        ask_for_space = response_part_description.ask_responding_part_for_space
        ai_msg = AIMessage(content = ask_for_space)
        return {"messages": [tool_message, ai_msg]}

    def _get_part_finder_ai_message(self, part_description):
        encouragment_text = part_description.next_response.encouragement_that_you_will_help
        can_feel_text = part_description.next_response.ask_if_can_feel_part
        where_text = part_description.next_response.ask_physically_where
        ai_msg = AIMessage(content = " NEXT ".join([encouragment_text + ' ' + can_feel_text, where_text]))
        return ai_msg
