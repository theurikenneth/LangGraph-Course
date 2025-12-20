from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_community.llms import LlamaCpp

llm = LlamaCpp(
    model_path="./models/Phi-3-mini-4k-instruct-q4.gguf",
    n_ctx=2048,
    n_threads=8,   # set to number of CPU cores
    n_batch=256,
    verbose=False,
    max_tokens=512,
    temperature=0.7,
    stop=["<|end|>", "<|user|>"]
)

class AgentState(TypedDict):
    messages: List[HumanMessage]

def process(state: AgentState) -> AgentState:
    # Get the last message content
    user_message = state["messages"][-1].content
    
    # Format for Phi-3
    prompt = f"<|system|>\nYou are a helpful AI assistant.<|end|>\n<|user|>\n{user_message}<|end|>\n<|assistant|>\n"
    
    # LlamaCpp.invoke() returns a string
    response = llm.invoke(prompt)
    response = response.strip()
    
    print(f"\nAI: {response}")

    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("Enter: ")
while user_input.lower() != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")


