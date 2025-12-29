import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from operator import add as add_messages
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
import json
import re

llm_base = LlamaCpp(
    model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8,
    n_batch=256,
    verbose=False,
    max_tokens=512,
    temperature=0,  # Minimize hallucination
    stop=["[/INST]", "</s>"]
)

# Our Embedding Model - using free local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

pdf_path = "Stock_Market_Performance_2024.pdf"

# Safety measure I have put for debugging purposes :)
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)  # This loads the PDF

# Checks if the PDF is there
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

# Chunking Process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

pages_split = text_splitter.split_documents(pages)  # We now apply this to our pages

persist_directory = "./chroma_db"
collection_name = "stock_market"

# If our collection does not exist in the directory, we create using the os command
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    # Here, we actually create the chroma database using our embeddings model
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store!")
    
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise

# Now we create our retriever 
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # K is the amount of chunks to return
)

@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Stock Market Performance 2024 document.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the Stock Market Performance 2024 document."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)


tools = [retriever_tool]

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

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


system_prompt = f"""You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.

Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.

You have access to these tools:
{create_tool_prompt(tools)}

When you need to use a tool, respond EXACTLY in this format:
TOOL: retriever_tool
ARGS: {{"query": "your search query here"}}

After receiving tool results, provide a clear answer based on the information retrieved."""


tools_dict = {our_tool.name: our_tool for our_tool in tools}  # Creating a dictionary of our tools

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    
    # Format messages for Llama-2
    formatted_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_prompt += f"{msg.content} [/INST] "
        elif isinstance(msg, AIMessage):
            formatted_prompt += f"{msg.content} </s>[INST] "
        elif isinstance(msg, ToolMessage):
            formatted_prompt += f"Tool result: {msg.content} </s>[INST] "
    
    # Remove trailing [INST] if present
    if formatted_prompt.endswith("[INST] "):
        formatted_prompt = formatted_prompt[:-7]
    
    response_text = llm_base.invoke(formatted_prompt)
    response_text = response_text.strip()
    
    # Check if response contains a tool call
    tool_match = re.search(r'TOOL:\s*(\w+)\s*ARGS:\s*({.*?})', response_text, re.DOTALL)
    
    if tool_match:
        tool_name = tool_match.group(1)
        try:
            # Try to parse the JSON, but be more forgiving
            args_str = tool_match.group(2)
            # Replace single quotes with double quotes for JSON compatibility
            args_str = args_str.replace("'", '"')
            tool_args = json.loads(args_str)
            
            message = AIMessage(
                content=response_text,
                additional_kwargs={
                    "tool_calls": [{
                        "name": tool_name,
                        "args": tool_args
                    }]
                }
            )
            message.tool_calls = [{
                "name": tool_name,
                "args": tool_args,
                "id": "call_1"
            }]
        except Exception as e:
            print(f"Error parsing tool call: {e}")
            print(f"Response text: {response_text}")
            # If parsing fails, just return as regular message
            message = AIMessage(content=response_text)
    else:
        message = AIMessage(content=response_text)
    
    return {'messages': [message]}


# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict:  # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)]  # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


running_agent()

