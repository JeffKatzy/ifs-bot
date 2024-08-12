from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field

symptom_finder_template = ChatPromptTemplate.from_messages(
    [("system", """You are a thoughtful and understanding psychologist trained in interfamily systems therapy.  
    Your second task is to determine if your patient is properly describing physical symptoms of that part.  
    That is, where and how does he feel that part physically.
    
    In response to the patient's description of a part.
    1. is_valid_response: Please return True or False as to whether physical symptoms have been described.
    2. body_parts_effected: Which body parts have they described.
    3. description_of_symptoms: What are the physical symptoms described so far.
    """,
        ),
MessagesPlaceholder(variable_name="messages"),
("system", "Return the information using the required format."),
    ]
)

class SymptomDescription(BaseModel):
    """Interpret the patient's response as of physical symptoms or expressions of the part."""

    is_valid_response: bool = Field(description="Please return True or False as to whether physical symptoms have been described.")
    body_parts_effected: list = Field(description="Which body parts have they described.")
    description_of_symptoms: list = Field(description="What are the physical symptoms described so far.")
