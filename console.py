# thread = {"configurable": {"thread_id": "3"}}
from src.agents.identify_self.identify_template import *
from src.agents.part_finder.part_finder import part_finder_agent
from src.agents.response_part_finder.response_part_finder import \
    response_part_finder_agent
from src.workflow.graph import Graph

graph = Graph()
graph.build()