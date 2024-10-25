from pydantic import BaseModel, ValidationError, Field
from langchain_openai import ChatOpenAI
# from langchain.chains import SimpleChain
from langchain_core.tools import tool
from typing import Optional, List, Dict
from langgraph.prebuilt import ToolNode
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

#  Get input data (coordinate with group)
#  PDF parsing
#  Data template
#  Prompting
#  Information extractor
#  Validator


class ExampleModel(BaseModel):
    name: str
    age: Optional[int] = None
    address: Dict[str, str]
    items: List[str]


example_objects = []


@tool
def validate_data(data: dict) -> Optional[ExampleModel]:
    """Validates an ExampleModel object using the .model_validate method."""
    try:
        ExampleModel.model_validate(data)
        example_objects.append(data)
    except ValidationError as exc:
        print(repr(exc.errors()[0]['type']))




@tool
def create_example_object(name: str, age: Optional[int], 
                          address: Dict[str, str], items: List[str]):
    """Creates a pydantic ExampleModel object from function parameters."""
    example_object = ExampleModel(name=name, age=age, address=address, items=items)
    return example_object

tools = [validate_data, create_example_object]

tool_node = ToolNode(tools)

model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key="API_KEY",  # if you prefer to pass api key in directly instaed of using env vars
    )

i = 0
# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls or len(example_objects) == 0:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    # Collect valid objects returned by `validate_data`
    for msg in response.tool_calls:
        if msg.tool_name == 'validate_data' and msg.tool_output is not None:
            example_objects.append(msg.tool_output.dict())
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer)
source_data = "John Doe, 28, lives in New York with his cat and owns a bike and a guitar."
# Use the Runnable
final_state = app.invoke(
    {"messages": [HumanMessage(content=f"Create 3 valid example objects from the source data. source data:{source_data}")]},
    config={"configurable": {"thread_id": 42}}
)

print(final_state["messages"][-1].content)
print(example_objects)
