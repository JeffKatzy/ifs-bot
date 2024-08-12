from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from src.workflow.nodes import Node


class Graph():
    def __init__(self):
        self.workflow = StateGraph(MessagesState)

    def add_nodes(self):
        node = Node()
        self.workflow.add_node("parse_part", node.parse_part)
        self.workflow.add_node("get_symptom", node.get_symptom)
        self.workflow.add_node("parse_symptom", node.parse_symptom)
        self.workflow.add_node("get_responding_part", node.get_responding_part)
        self.workflow.add_node("parse_responding_parts", node.parse_responding_parts)

    def add_edges(self):
        self.workflow.set_entry_point("parse_part")
        self.workflow.add_edge("parse_part", "get_symptom")
        self.workflow.add_edge("get_symptom", "parse_symptom")
        self.workflow.add_edge("parse_symptom", "get_responding_part")
        self.workflow.add_edge("get_responding_part", "parse_responding_parts")

    def build(self):
        self.add_nodes()
        self.add_edges()
        memory = MemorySaver()
        return self.workflow.compile(checkpointer=memory,
        interrupt_before=["get_symptom", 
        "get_responding_part"])


