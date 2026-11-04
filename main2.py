
import os
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from deepagents import create_deep_agent
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

load_dotenv()

console = Console()

MODEL_NAME = os.getenv("MODEL_NAME", "openai:gpt-5-nano")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

internet_search = {"type": "web_search"}

research_instructions = """
You are an expert research assistant.

Research the user's topic using web search.
Prefer official and reliable sources.
Verify important factual claims.
Provide source URLs.

Keep research focused and avoid unnecessary searches.

Write a well-structured Markdown report with:
1. Executive Summary
2. Key Findings
3. Detailed Analysis
4. Conclusion
5. Sources
"""

agent = create_deep_agent(
    model=MODEL_NAME,
    tools=[internet_search],
    system_prompt=research_instructions,
)


def get_text(message):
    """Extract visible text without reasoning or tool payloads."""
    parts = []

    for block in message.content_blocks:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))

    return "".join(parts)


def safe_serialize(obj):
    """Make LangChain objects JSON serializable."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")

    if isinstance(obj, dict):
        return {
            str(key): safe_serialize(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [safe_serialize(item) for item in obj]

    return obj


def remove_encrypted_data(obj):
    """Remove large opaque reasoning payloads."""
    if isinstance(obj, dict):
        return {
            key: remove_encrypted_data(value)
            for key, value in obj.items()
            if key not in {
                "encrypted_content",
                "signature",
            }
        }

    if isinstance(obj, list):
        return [remove_encrypted_data(item) for item in obj]

    return obj


def show_usage(messages):
    """Display token usage from completed AI messages."""
    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    web_actions = []

    for message in messages:
        if message.type != "ai":
            continue

        usage = message.usage_metadata or {}

        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)

        output_details = usage.get(
            "output_token_details", {}
        ) or {}

        reasoning_tokens += output_details.get(
            "reasoning", 0
        )

        # Provider-hosted web search events
        content = message.content
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue

                if block.get("type") == "web_search_call":
                    web_actions.append({
                        "status": block.get("status"),
                        "action": block.get("action"),
                    })

    table = Table(title="Execution Statistics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Input Tokens", f"{input_tokens:,}")
    table.add_row("Output Tokens", f"{output_tokens:,}")
    table.add_row(
        "Reasoning Tokens",
        f"{reasoning_tokens:,}"
    )
    table.add_row(
        "Total Tokens",
        f"{input_tokens + output_tokens:,}"
    )
    table.add_row(
        "Web Search Actions",
        str(len(web_actions))
    )

    console.print(table)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "web_search_actions": web_actions,
    }


def run_research(query):
    console.print(
        Panel(
            f"Model: {MODEL_NAME}\nQuery: {query}",
            title="Deep Agents Research",
        )
    )

    inputs = {
        "messages": [
            {"role": "user", "content": query}
        ]
    }

    result = None
    streamed_text = False
    started_text = False

    # Receive multiple event types from one execution.
    for event in agent.stream(
        inputs,
        stream_mode=["messages", "updates", "values"],
        version="v2",
        config={"recursion_limit": 30},
    ):
        event_type = event["type"]
        data = event["data"]

        if event_type == "messages":
            message_chunk, metadata = data

            if message_chunk.type != "AIMessageChunk":
                continue

            text = get_text(message_chunk)

            if text:
                if not started_text:
                    console.print(
                        "\n[bold green]Live Response[/]\n"
                    )
                    started_text = True

                print(text, end="", flush=True)
                streamed_text = True

        elif event_type == "updates":
            # Avoid printing full message contents again.
            for node_name, update in data.items():
                console.print(
                    f"\n[dim]Completed step: {node_name}[/]"
                )

        elif event_type == "values":
            # Full state snapshot. Keep the last one.
            result = data

    if result is None:
        raise RuntimeError(
            "No final agent state received."
        )

    messages = result.get("messages", [])

    # Find the last AI message with visible text.
    final_report = ""

    for message in reversed(messages):
        if message.type == "ai":
            final_report = get_text(message)
            if final_report:
                break

    if not streamed_text:
        console.print("\n[bold]Final Report[/bold]")
        console.print(Markdown(final_report))

    console.print("\n\n")
    statistics = show_usage(messages)

    # Export formatted output.
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    report_path = OUTPUT_DIR / f"report_{timestamp}.md"
    json_path = OUTPUT_DIR / f"result_{timestamp}.json"

    report_path.write_text(
        final_report,
        encoding="utf-8"
    )

    # Serialize complete result, excluding encrypted
    # reasoning payloads.
    formatted_result = safe_serialize(result)
    formatted_result = remove_encrypted_data(
        formatted_result
    )

    export = {
        "model": MODEL_NAME,
        "query": query,
        "statistics": statistics,
        "result": formatted_result,
    }

    json_path.write_text(
        json.dumps(
            export,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    console.print(
        f"[green]Report saved:[/] {report_path}"
    )
    console.print(
        f"[green]JSON saved:[/] {json_path}"
    )

    return result


if __name__ == "__main__":
    query = input("Enter research topic: ").strip()

    if not query:
        raise ValueError("Research topic is required")

    result = run_research(query)
