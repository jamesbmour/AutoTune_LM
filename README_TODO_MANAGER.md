# LangGraph Todo List Manager with Reminder Subgraph

A demonstration of LangGraph's subgraph functionality and state management through an intelligent todo list manager with LLM-powered reminders.

## Features

- **Main Graph Structure**: Add todo items, update shared state, invoke reminder subgraph, display results
- **Reminder Subgraph**: Task prioritization using LLM, intelligent reminder generation
- **Shared State Management**: Seamless state sharing between main graph and subgraph
- **LLM Integration**: Ollama (local) with OpenAI fallback for task analysis
- **Mock LLM Support**: Works without external dependencies for demonstration

## Architecture

```
Main Graph:
add_todo → update_state → reminder_subgraph → display_results
                              ↓
                         Reminder Subgraph:
                    analyze_tasks → generate_reminders
```

## Requirements

For full LLM functionality:
```bash
pip install langchain langchain-community langchain-ollama langchain-openai langgraph
```

The application includes mock implementations and will work without these dependencies for demonstration purposes.

## Usage

### Basic Usage

```python
from langgraph_todo_manager import TodoManager

# Create manager instance
manager = TodoManager()

# Add todos
manager.add_todo("Complete quarterly report")
manager.add_todo("Review marketing campaign")

# View todos with intelligent reminders
print(manager.list_todos())

# Complete a todo
manager.complete_todo(1)
```

### Running the Demo

```bash
python langgraph_todo_manager.py
```

This will demonstrate:
1. Adding sample todo items
2. Generating intelligent reminders via LLM analysis
3. Displaying prioritized todo list with suggestions
4. Completing tasks and updating the list

## LLM Configuration

### Ollama (Local)
1. Install and start Ollama
2. Pull a model: `ollama pull llama3.2`
3. The application will auto-detect and use Ollama

### OpenAI (Cloud)
Set environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Mock Mode (Default)
If no LLM is available, the application uses intelligent mock responses for demonstration.

## Key Components

### State Schema
- `TodoItem`: Individual todo with metadata
- `ReminderData`: LLM-generated reminders and priorities
- `TodoState`: Shared state between graphs

### Main Graph Nodes
- `add_todo_node`: Creates new todo items
- `update_state_node`: Prepares state for subgraph
- `display_results_node`: Shows formatted todo list

### Reminder Subgraph Nodes
- `analyze_tasks_node`: LLM analysis for task prioritization
- `generate_reminders_node`: Converts LLM output to structured reminders

## Technical Details

- **Code Size**: ~337 lines (close to 300-line target)
- **LangGraph Patterns**: Proper subgraph integration and state management
- **Error Handling**: Graceful fallbacks for LLM failures
- **Clean Architecture**: Separation of concerns between workflow and reminder logic

## Example Output

```
🚀 LangGraph Todo Manager Demo

Adding sample todos...
✓ Added: Complete the quarterly report...
✓ Added: Review and approve the new mar...

Generating intelligent reminders...

=== TODO LIST WITH REMINDERS ===

○ [1] Complete the quarterly report for the finance team
   Priority: medium | Created: 2025-09-24
   🔔 Reminder: High priority quarterly report needs attention
   📈 Priority Score: 8/10
   💡 Suggestion: Block 2 hours for focused writing

○ [2] Review and approve the new marketing campaign
   Priority: medium | Created: 2025-09-24
   🔔 Reminder: Marketing campaign review is time-sensitive
   📈 Priority Score: 6/10
   💡 Suggestion: Schedule review meeting with team
```

This implementation demonstrates advanced LangGraph concepts including subgraph composition, state management, and real-world LLM integration patterns.