from typing import List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field

part_finder_template = ChatPromptTemplate.from_messages(
    [("system", """You are a thoughtful and understanding psychologist trained in interfamily systems therapy.  
    Your first task is to determine if your patient is properly describing a part that he/she is trying to
    work with.  Valid parts include:

    1. Exiles - these are often sad parts and hold pain, trauma, and vulnerabilities
    2. Managers - These may be anxious parts that display characteristics of perfectionism, controlling, or caretaking.  This may have feelings of anxiety, and when that control is violated can lead to anger.
    3. Firefighters - These may try to attempting to distract or numb the pain through impulsive behaviors or activities such as substance abuse, overeating, or self-harm.  This also may be characterized as anger.

    A patient does not need to use, or may not be aware of these terms in describing a part.  It is your job to determine whether they 
    are describing characteristics that fit into this.

    In response to the patient's description of a part.
    1. is_part: Please return True or False as to whether this is a valid IFS part
    2. ifs_part_category: What is the kind of part they are using
    3. patients_part_label: What is the label that the patient uses to describe this part
    4. part_characteristics: What, if any, are some of the characteristics that the patient used to describe the part.

    NextResponse - Again respond as if you are a friendly psychologist.
    Then respond to the patients describing the part with:
    1. Encouragement_that_you_will_help: Indicate calmly that you are happy to help with that part.  Use the patients part label.  For example, if the patients part lable is anxiety, sure I'd be happy to help you explore that anxious part.
    2. ask_if_can_feel_part: Ask the patient if they are able to feel that part now.
    3. ask_physically_where: Ask where the patient can feel that part.
    """,
        ),
MessagesPlaceholder(variable_name="messages"),
("system", "Return the information using the required format."),
    ]
)

class Response(BaseModel):
    encouragement_that_you_will_help: str = Field(
        description="Indicate calmly that you are happy to help with that part.  Use the patients part label.  For example, if the patients part lable is anxiety, sure I'd be happy to help you explore that anxious part."
    )
    ask_if_can_feel_part: str = Field(description = "Ask the patient if they are able to feel that part now.")
    ask_physically_where: str = Field(description = "Ask where the patient can feel that part.")

class PartDescription(BaseModel):
    """Answer the question."""

    is_part: bool = Field(description="Returns True or False as to whether this is a valid IFS part")
    ifs_part_category: str = Field(description="What is the kind of part they are describing")
    patients_part_label: str = Field(description="What is the label that the patient uses to describe this part")
    part_characteristics: list = Field(description="What, if any, are some of the characteristics that the patient used to describe the part")
    next_response: Response = Field(description="Your reflection on the patients describing of an IFS part.")

