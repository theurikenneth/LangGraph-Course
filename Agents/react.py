from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_community.llms import LlamaCpp
import json
import re

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers together"""
    return a + b 

@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

tools = [add, subtract, multiply]

llm = LlamaCpp(
    model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8,
    n_batch=256,
    verbose=False,
    max_tokens=512,
    temperature=0.7,
    stop=["[/INST]", "</s>"]
)

def create_tool_prompt(tools):
    """Create a prompt that describes available tools with their exact parameter names"""
    tool_descriptions = []
    for tool in tools:
        params = tool.args_schema.model_json_schema()['properties']
        param_details = []
        for param_name, param_info in params.items():
            param_type = param_info.get('type', 'any')
            param_details.append(f"{param_name}: {param_type}")
        
        params_str = ", ".join(param_details)
        tool_descriptions.append(f"- {tool.name}({params_str}): {tool.description}")
    
    return "\n".join(tool_descriptions)

def model_call(state: AgentState) -> AgentState:
    system_prompt = f"""You are my AI assistant, please answer my query to the best of your ability.

You have access to these tools:
{create_tool_prompt(tools)}

When you need to use a tool, respond EXACTLY in this format:
TOOL: tool_name
ARGS: {{"param_name1": value1, "param_name2": value2}}

You can call multiple tools if needed. After using tools, provide a final answer to the user.

Otherwise, respond normally to answer the user's question."""
    
    # Format messages for Llama-2
    formatted_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
    
    # Add conversation history
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            formatted_prompt += f"{msg.content} [/INST] "
        elif isinstance(msg, AIMessage):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                formatted_prompt += f"{msg.content} </s>[INST] "
            else:
                formatted_prompt += f"{msg.content} </s>[INST] "
    
    # Remove last [INST] if present
    if formatted_prompt.endswith("[INST] "):
        formatted_prompt = formatted_prompt[:-7]
    
    response = llm.invoke(formatted_prompt)
    response = response.strip()
    
    # Check if response contains a tool call
    tool_match = re.search(r'TOOL:\s*(\w+)\s*ARGS:\s*({.*?})', response, re.DOTALL)
    
    if tool_match:
        tool_name = tool_match.group(1)
        try:
            tool_args = json.loads(tool_match.group(2))
            ai_message = AIMessage(
                content=response,
                additional_kwargs={
                    "tool_calls": [{
                        "name": tool_name,
                        "args": tool_args
                    }]
                }
            )
            ai_message.tool_calls = [{
                "name": tool_name,
                "args": tool_args,
                "id": "call_1"
            }]
        except:
            ai_message = AIMessage(content=response)
    else:
        ai_message = AIMessage(content=response)
    
    return {"messages": [ai_message]}

def should_continue(state: AgentState): 
    messages = state["messages"]
    last_message = messages[-1]
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls: 
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            print(f"\n{message.content}")

inputs = {"messages": [HumanMessage(content="Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
print_stream(app.stream(inputs, stream_mode="values"))
