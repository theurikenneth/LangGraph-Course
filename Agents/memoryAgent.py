from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_community.llms import LlamaCpp

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

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

def format_messages_for_llama2(messages):
    """Format messages in Llama-2's expected format"""
    formatted = "[INST] <<SYS>>\nYou are a helpful AI assistant.\n<</SYS>>\n\n"
    
    first_user = True
    for msg in messages:
        if isinstance(msg, HumanMessage):
            if first_user:
                formatted += f"{msg.content} [/INST] "
                first_user = False
            else:
                formatted += f"[INST] {msg.content} [/INST] "
        elif isinstance(msg, AIMessage):
            formatted += f"{msg.content} </s>"
    
    return formatted

def process(state: AgentState) -> AgentState:
    # Format all messages for Llama-2
    prompt = format_messages_for_llama2(state["messages"])
    
    # LlamaCpp.invoke() returns a string
    response = llm.invoke(prompt)
    
    # Clean up response
    response = response.strip()
    
    # Append AI response to messages
    state["messages"].append(AIMessage(content=response))
    
    print(f"\nAI: {response}")

    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")

with open("logging.txt", "w") as file:
    file.write("Your Conversation Log: \n")

    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content} \n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")


