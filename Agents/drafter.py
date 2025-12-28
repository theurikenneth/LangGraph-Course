from typing import Annotated, Sequence, TypedDict
from langchain_community.llms import LlamaCpp
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import json
import re

# This is the global variable to store document content
document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str) -> str:
    """Updates the document with the provided content."""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is:\n{document_content}"


@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the process.
    
    Args:
        filename: Name for the text file.
    """

    global document_content

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."
    
    except Exception as e:
        return f"Error saving document: {str(e)}"
    

tools = [update, save]

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

def our_agent(state: AgentState) -> AgentState:
    system_prompt_text = f"""You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
- If the user wants to update or modify content, use the 'update' tool with the complete updated content.
- If the user wants to save and finish, you need to use the 'save' tool.
- Make sure to always show the current document state after modifications.

The current document content is: {document_content}

You have access to these tools:
{create_tool_prompt(tools)}

When you need to use a tool, respond EXACTLY in this format:
TOOL: tool_name
ARGS: {{"param_name": "value"}}

Otherwise, respond normally to help the user."""

    if not state["messages"]:
        user_input = "I'm ready to help you create a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
        response = AIMessage(content=user_input)
        print(f"\n🤖 AI: {user_input}")
        return {"messages": [user_message, response]}

    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    # Format messages for Llama-2
    formatted_prompt = f"[INST] <<SYS>>\n{system_prompt_text}\n<</SYS>>\n\n"
    
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            formatted_prompt += f"{msg.content} [/INST] "
        elif isinstance(msg, AIMessage):
            formatted_prompt += f"{msg.content} </s>[INST] "
        elif isinstance(msg, ToolMessage):
            formatted_prompt += f"Tool result: {msg.content} </s>[INST] "
    
    # Add current user message
    formatted_prompt += f"{user_input} [/INST] "
    
    response_text = llm.invoke(formatted_prompt)
    response_text = response_text.strip()
    
    # Check if response contains a tool call
    tool_match = re.search(r'TOOL:\s*(\w+)\s*ARGS:\s*({.*?})', response_text, re.DOTALL)
    
    if tool_match:
        tool_name = tool_match.group(1)
        try:
            tool_args = json.loads(tool_match.group(2))
            response = AIMessage(
                content=response_text,
                additional_kwargs={
                    "tool_calls": [{
                        "name": tool_name,
                        "args": tool_args
                    }]
                }
            )
            response.tool_calls = [{
                "name": tool_name,
                "args": tool_args,
                "id": "call_1"
            }]
            print(f"\n🤖 AI: {response_text}")
            print(f"🔧 USING TOOLS: {[tool_name]}")
        except Exception as e:
            response = AIMessage(content=response_text)
            print(f"\n🤖 AI: {response_text}")
    else:
        response = AIMessage(content=response_text)
        print(f"\n🤖 AI: {response_text}")

    return {"messages": list(state["messages"]) + [user_message, response]}


def should_call_tools(state: AgentState) -> str:
    """Check if the agent wants to call a tool"""
    messages = state["messages"]
    if not messages:
        return "agent"
    
    last_message = messages[-1]
    
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    else:
        return "agent"


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""

    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    # This looks for the most recent tool message....
    for message in reversed(messages):
        # ... and checks if this is a ToolMessage resulting from save
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end" # goes to the end edge which leads to the endpoint
        
    return "continue"

def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_call_tools,
    {
        "tools": "tools",
        "agent": "agent",
    },
)

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()