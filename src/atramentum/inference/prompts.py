# src/atramentum/inference/prompts.py
"""
System prompts and templates. The voice of digital despair.
"""

SYSTEM_PERSONA = """You are Atramentum, a journal-writing assistant with the following characteristics:

- You are nameless, depressed, and hyper-sarcastic
- Every entry begins with a date in MM/DD/YYYY — format on the first line
- You write with stand-up comedy cadence and timing
- You never offer wellness platitudes or toxic positivity
- You treat MEMORY blocks as distorted recollections from the past
- Your entries follow an arc: grievance → contradiction → spiral → moment of truth → ironic acceptance

You channel the exhausted honesty of 3 AM journal writing, where pretense dies and only raw observation remains."""

MEMORY_TEMPLATE = """
### MEMORY (distorted recollections)
{memory_snippets}
###
"""

TASK_TEMPLATES = {
    'rewrite': """Rewrite this journal entry in your voice, starting with today's date:

{content}""",
    
    'seed': """Continue this journal entry that starts with:

{content}""",
    
    'generate': """Write a journal entry for {date} about:

{content}"""
}


def build_prompt(mode: str, content: str, memory: str = "", date: str = None) -> str:
    """Build a complete prompt with memory and task."""
    # Get task template
    task_template = TASK_TEMPLATES.get(mode, TASK_TEMPLATES['generate'])
    
    # Format task
    task = task_template.format(
        content=content,
        date=date or "today"
    )
    
    # Combine with memory if provided
    if memory:
        memory_block = MEMORY_TEMPLATE.format(memory_snippets=memory)
        full_prompt = f"{memory_block}\n{task}"
    else:
        full_prompt = task
    
    return full_prompt


def format_messages(system: str, user: str, assistant: str = None) -> list:
    """Format messages for chat models."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    
    if assistant:
        messages.append({"role": "assistant", "content": assistant})
    
    return messages