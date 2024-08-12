from langchain_core.messages import HumanMessage

from src.workflow import Graph

app = Graph().build()

thread = {"configurable": {"thread_id": "3"}}

input_text = """
    So I think that my anger is a byproduct of fear.  And I think that the fear may be worth digging into.  And I have found that anxiety has become more pronounced and this vague sense of ease that something bad may or will happen, is something that I've 
    noticed more and more as depression has become less of an ongoing battle for me.  And I would like to explore that.
    """
inputs = [HumanMessage(content=input_text)]

for event in app.stream({"messages": inputs}, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

input_text = "Currently I feel it as tension in the jaw."
inputs = [HumanMessage(content=input_text)]

snapshot = app.get_state(thread)
snapshot.values["messages"] += inputs

app.update_state(thread, snapshot.values, as_node="get_symptom")

for event in app.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

input_text = """Well if I'm being honest I would like it to go away.  And I am quite annoyed with it, because it gets in the way of my happiness a lot."""
inputs = [HumanMessage(content=input_text)]
app.update_state(thread, {"messages": inputs}, as_node="get_responding_part")

for event in app.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()
