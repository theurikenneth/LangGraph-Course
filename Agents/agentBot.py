from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from llama_cpp import Llama

llm = Llama(
    model_path="./models/Phi-3-mini-4k-instruct-q4.gguf",
    n_ctx=2048,
    n_threads=8,   # set to number of CPU cores
    n_batch=256,
    verbose=False
)

class AgentState(TypedDict):
    messages: List[HumanMessage]

def process(state: AgentState) -> AgentState:
    prompt = "\n".join([m.content for m in state["messages"]])

    output = llm(
        prompt,
        max_tokens=256,
        temperature=0.7
    )

    response = output["choices"][0]["text"]
    print(f"\nAI: {response}\n")

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


