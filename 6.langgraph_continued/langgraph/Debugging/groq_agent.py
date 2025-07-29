from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import END, START, MessageGraph
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

model = ChatGroq(temperature=0, model="llama3-8b-8192")

def make_default_graph():
    """Make a simple LLM agent"""
    graph_workflow = StateGraph(State)
    def call_model(state):
        return {"messages": [model.invoke(state["messages"])]}
    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge("agent", END)
    graph_workflow.add_edge(START, "agent")
    agent = graph_workflow.compile()
    return agent

def make_alternative_graph():
    """Make a tool-calling agent"""
    @tool
    def add(a: float, b: float):
        """Adds two numbers."""
        return a + b
    
    @tool
    def subtract(a: float, b: float):
        """Subtracts b from a."""
        return a - b
    
    @tool
    def multiply(a: float, b: float):
        """Multiplies two numbers."""
        return a * b
    
    @tool
    def divide(a: float, b: float):
        """Divides a by b."""
        if b == 0:
            return "Error: Division by zero is not allowed."
        return a / b
    
    @tool
    def power(a: float, b: float):
        """Raises a to the power of b."""
        return a ** b
    
    @tool
    def get_word_count(text: str):
        """Counts the number of words in a given text."""
        return len(text.split())
    
    @tool
    def reverse_string(text: str):
        """Reverses a given string."""
        return text[::-1]
    
    @tool
    def to_uppercase(text: str):
        """Converts text to uppercase."""
        return text.upper()

    tools = [add, subtract, multiply, divide, power, get_word_count, reverse_string, to_uppercase]
    tool_node = ToolNode(tools)
    model_with_tools = model.bind_tools(tools)

    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def should_continue(state: State):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END

    graph_workflow = StateGraph(State)
    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)
    graph_workflow.add_edge("tools", "agent")
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_conditional_edges("agent", should_continue)
    agent = graph_workflow.compile()
    return agent

# Export the graph for LangGraph Studio
agent = make_alternative_graph()