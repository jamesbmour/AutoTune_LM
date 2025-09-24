"""
LangGraph Todo List Manager with Reminder Subgraph

Demonstrates LangGraph subgraphs and state management through an intelligent 
todo list manager with LLM-powered reminders.
"""

import json
from typing import List, Dict, Any, Optional, TypedDict, Callable
from datetime import datetime
import os

# Mock LangChain components for demonstration
try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, END
    from langgraph.graph.state import CompiledStateGraph
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    
    class MockMessage:
        def __init__(self, content: str):
            self.content = content
    
    class HumanMessage(MockMessage): pass
    class SystemMessage(MockMessage): pass
    
    class MockLLM:
        def invoke(self, messages):
            return MockMessage(json.dumps([
                {"task_id": i+1, "priority_score": 8-i, 
                 "reminder_text": f"Task {i+1} priority reminder", 
                 "suggested_action": f"Focus on task {i+1}"}
                for i in range(4)
            ]))
    
    class ChatOllama(MockLLM): 
        def __init__(self, **kwargs): pass
    class ChatOpenAI(MockLLM): 
        def __init__(self, **kwargs): pass
    
    class StateGraph:
        def __init__(self, state_schema):
            self.nodes = {}
            self.edges = []
            self.entry_point = None
            self.finish_point = None
        
        def add_node(self, name: str, func: Callable):
            self.nodes[name] = func
        
        def add_edge(self, from_node: str, to_node: str):
            self.edges.append((from_node, to_node))
        
        def set_entry_point(self, node: str):
            self.entry_point = node
        
        def set_finish_point(self, node: str):
            self.finish_point = node
        
        def compile(self):
            return CompiledStateGraph(self)
    
    class CompiledStateGraph:
        def __init__(self, graph):
            self.graph = graph
        
        def invoke(self, state):
            current_state = state.copy()
            if self.graph.entry_point in self.graph.nodes:
                current_state = self.graph.nodes[self.graph.entry_point](current_state)
                visited = {self.graph.entry_point}
                for from_node, to_node in self.graph.edges:
                    if from_node in visited and to_node in self.graph.nodes:
                        node = self.graph.nodes[to_node]
                        if isinstance(node, CompiledStateGraph):
                            current_state = node.invoke(current_state)
                        else:
                            current_state = node(current_state)
                        visited.add(to_node)
            return current_state
    
    END = "END"


# --- State Schema ---
class TodoItem(TypedDict):
    id: int
    title: str
    description: str
    priority: str
    created_at: str
    completed: bool

class ReminderData(TypedDict):
    task_id: int
    reminder_text: str
    priority_score: int
    suggested_action: str

class TodoState(TypedDict):
    todos: List[TodoItem]
    reminders: List[ReminderData]
    current_action: str
    user_input: str
    llm_response: str
    error_message: Optional[str]

# --- LLM Setup ---
def get_llm():
    """Initialize LLM client with fallback."""
    if LANGCHAIN_AVAILABLE:
        try:
            return ChatOllama(model="llama3.2", base_url="http://127.0.0.1:11434", temperature=0.7)
        except Exception:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=api_key)
    print("⚠️  Using mock LLM for demonstration")
    return ChatOllama()

# --- Main Graph Nodes ---
def add_todo_node(state: TodoState) -> TodoState:
    """Add a new todo item to the list."""
    user_input = state.get("user_input", "")
    if not user_input.strip():
        state["error_message"] = "Please provide todo details."
        return state
    
    new_id = max([todo["id"] for todo in state["todos"]], default=0) + 1
    new_todo: TodoItem = {
        "id": new_id,
        "title": user_input[:50],
        "description": user_input,
        "priority": "medium",
        "created_at": datetime.now().isoformat(),
        "completed": False
    }
    
    state["todos"].append(new_todo)
    state["current_action"] = f"Added todo: {new_todo['title']}"
    state["error_message"] = None
    return state

def update_state_node(state: TodoState) -> TodoState:
    """Update shared state and prepare for reminder subgraph."""
    active_todos = [todo for todo in state["todos"] if not todo["completed"]]
    state["current_action"] = f"Prepared {len(active_todos)} active todos for reminder processing"
    return state

def display_results_node(state: TodoState) -> TodoState:
    """Display the updated todo list with reminders."""
    todos = state["todos"]
    reminders = state["reminders"]
    
    display_text = "\n=== TODO LIST WITH REMINDERS ===\n"
    for todo in todos:
        status = "✓" if todo["completed"] else "○"
        display_text += f"\n{status} [{todo['id']}] {todo['title']}\n"
        display_text += f"   Priority: {todo['priority']} | Created: {todo['created_at'][:10]}\n"
        
        reminder = next((r for r in reminders if r["task_id"] == todo["id"]), None)
        if reminder:
            display_text += f"   🔔 Reminder: {reminder['reminder_text']}\n"
            display_text += f"   📈 Priority Score: {reminder['priority_score']}/10\n"
            display_text += f"   💡 Suggestion: {reminder['suggested_action']}\n"
    
    if not todos:
        display_text += "\nNo todos found. Add some tasks to get started!\n"
    
    display_text += "\n" + "="*40 + "\n"
    state["llm_response"] = display_text
    state["current_action"] = "Displayed todo list with reminders"
    return state

# --- Reminder Subgraph Nodes ---
def analyze_tasks_node(state: TodoState) -> TodoState:
    """Analyze tasks using LLM for prioritization."""
    active_todos = [todo for todo in state["todos"] if not todo["completed"]]
    if not active_todos:
        state["current_action"] = "No active tasks to analyze"
        return state
    
    task_context = "Current active tasks:\n"
    for todo in active_todos:
        task_context += f"- ID {todo['id']}: {todo['title']} (Priority: {todo['priority']})\n"
    
    system_prompt = """Analyze tasks and provide priority_score (1-10), reminder_text, and suggested_action for each.
    Respond in JSON format: [{"task_id": int, "priority_score": int, "reminder_text": str, "suggested_action": str}]"""
    
    try:
        llm = get_llm()
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=task_context)]
        response = llm.invoke(messages)
        state["current_action"] = "Analyzed tasks with LLM"
        state["llm_response"] = response.content
    except Exception as e:
        state["error_message"] = f"LLM analysis failed: {str(e)}"
        state["current_action"] = "LLM analysis failed"
    return state

def generate_reminders_node(state: TodoState) -> TodoState:
    """Generate structured reminders from LLM response."""
    llm_response = state.get("llm_response", "")
    if not llm_response:
        state["error_message"] = "No LLM response to process"
        return state
    
    try:
        response_text = llm_response.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:-3]
        
        reminders_data = json.loads(response_text)
        reminders = [ReminderData(
            task_id=item.get("task_id", 0),
            reminder_text=item.get("reminder_text", "Complete this task"),
            priority_score=item.get("priority_score", 5),
            suggested_action=item.get("suggested_action", "Start working on it")
        ) for item in reminders_data]
        
        state["reminders"] = reminders
        state["current_action"] = f"Generated {len(reminders)} reminders"
        state["error_message"] = None
        
    except json.JSONDecodeError:
        # Fallback: create basic reminders
        active_todos = [todo for todo in state["todos"] if not todo["completed"]]
        fallback_reminders = [ReminderData(
            task_id=todo["id"],
            reminder_text=f"Don't forget: {todo['title']}",
            priority_score=5,
            suggested_action="Review and take the next step"
        ) for todo in active_todos]
        
        state["reminders"] = fallback_reminders
        state["current_action"] = f"Generated {len(fallback_reminders)} fallback reminders"
    
    return state

# --- Graph Construction ---
def create_reminder_subgraph() -> CompiledStateGraph:
    """Create the reminder subgraph for task analysis and reminder generation."""
    subgraph = StateGraph(TodoState)
    subgraph.add_node("analyze_tasks", analyze_tasks_node)
    subgraph.add_node("generate_reminders", generate_reminders_node)
    subgraph.add_edge("analyze_tasks", "generate_reminders")
    subgraph.set_entry_point("analyze_tasks")
    subgraph.set_finish_point("generate_reminders")
    return subgraph.compile()

def create_main_graph() -> CompiledStateGraph:
    """Create the main todo management graph."""
    reminder_subgraph = create_reminder_subgraph()
    main_graph = StateGraph(TodoState)
    main_graph.add_node("add_todo", add_todo_node)
    main_graph.add_node("update_state", update_state_node)
    main_graph.add_node("reminder_subgraph", reminder_subgraph)
    main_graph.add_node("display_results", display_results_node)
    main_graph.add_edge("add_todo", "update_state")
    main_graph.add_edge("update_state", "reminder_subgraph")
    main_graph.add_edge("reminder_subgraph", "display_results")
    main_graph.set_entry_point("add_todo")
    main_graph.set_finish_point("display_results")
    return main_graph.compile()

# --- Main Application ---
class TodoManager:
    """Interactive Todo List Manager with LangGraph and LLM integration."""
    
    def __init__(self):
        self.graph = create_main_graph()
        self.state: TodoState = {
            "todos": [], "reminders": [], "current_action": "",
            "user_input": "", "llm_response": "", "error_message": None
        }
    
    def add_todo(self, task_description: str) -> str:
        """Add a new todo and generate reminders."""
        self.state["user_input"] = task_description
        result = self.graph.invoke(self.state)
        self.state.update(result)
        return result.get("llm_response", "Todo added successfully!")
    
    def list_todos(self) -> str:
        """List all todos with current reminders."""
        if not self.state["todos"]:
            return "No todos found. Add some tasks to get started!"
        
        reminder_subgraph = create_reminder_subgraph()
        updated_state = reminder_subgraph.invoke(self.state)
        self.state.update(updated_state)
        display_state = display_results_node(self.state)
        return display_state.get("llm_response", "Failed to display todos")
    
    def complete_todo(self, todo_id: int) -> str:
        """Mark a todo as completed."""
        for todo in self.state["todos"]:
            if todo["id"] == todo_id:
                todo["completed"] = True
                return f"Completed todo: {todo['title']}"
        return f"Todo with ID {todo_id} not found"

# --- Example Usage ---
def demo():
    """Demonstration of the Todo Manager with LangGraph subgraphs."""
    print("🚀 LangGraph Todo Manager Demo\n")
    
    manager = TodoManager()
    sample_todos = [
        "Complete the quarterly report for the finance team",
        "Review and approve the new marketing campaign",
        "Schedule dentist appointment for next week",
        "Learn about LangGraph subgraph implementation"
    ]
    
    print("Adding sample todos...")
    for todo in sample_todos:
        result = manager.add_todo(todo)
        print(f"✓ Added: {todo[:30]}...")
    
    print("\nGenerating intelligent reminders...")
    todos_display = manager.list_todos()
    print(todos_display)
    
    print("\nCompleting a task...")
    completion_result = manager.complete_todo(1)
    print(f"✓ {completion_result}")
    
    print("\nUpdated todo list:")
    updated_display = manager.list_todos()
    print(updated_display)

if __name__ == "__main__":
    demo()