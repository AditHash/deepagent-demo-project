
import os
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from deepagents import create_deep_agent
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()

console = Console()
MODEL = os.getenv("MODEL_NAME", "openai:gpt-5-nano")

OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

research_instructions = """
You are Mini SalesDig, an AI-powered B2B research agent.

Research the target company and produce a professional
sales intelligence report.

Research:
1. Company overview, industry and headquarters
2. Products, services and target customers
3. Existing technology, cloud and AI initiatives
4. Potential business challenges
5. Three to five AI/GenAI opportunities
6. Recommended solutions and business benefits
7. Suggested sales pitch

Rules:
- Use web search for company research.
- Prefer official company sources.
- Verify important factual claims.
- Distinguish confirmed facts from hypotheses.
- Never invent company information.
- Avoid repetitive searches.
- Include source URLs.
- Write the final report in Markdown.
"""

agent = create_deep_agent(
    model=MODEL,
    tools=[{"type": "web_search"}],
    system_prompt=research_instructions,
)


def extract_text(message):
    return "".join(
        block.get("text", "")
        for block in message.content_blocks
        if block.get("type") == "text"
    )


def research_company(company, website):
    prompt = f"""
    Research this company for B2B sales intelligence.

    Company: {company}
    Website: {website or "Find the official website"}

    Identify realistic AI/ML/GenAI opportunities.
    Generate an evidence-based report with citations.
    """

    console.print(f"\n[cyan]Researching {company}...[/cyan]\n")

    result = None
    started = False

    for event in agent.stream(
        {"messages": [{"role": "user", "content": prompt}]},
        stream_mode=["messages", "values"],
        version="v2",
        config={"recursion_limit": 30},
    ):
        if event["type"] == "messages":
            chunk, metadata = event["data"]

            if chunk.type != "AIMessageChunk":
                continue

            text = extract_text(chunk)

            if text:
                if not started:
                    console.print("[green]Generating report...[/green]\n")
                    started = True

                print(text, end="", flush=True)

        elif event["type"] == "values":
            result = event["data"]

    if result is None:
        raise RuntimeError("Agent returned no final state")

    messages = result["messages"]

    # Final report
    report = ""
    for message in reversed(messages):
        if message.type == "ai":
            report = extract_text(message)
            if report:
                break

    # Statistics and source citations
    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    searches = []
    sources = {}

    for message in messages:
        if message.type != "ai":
            continue

        usage = message.usage_metadata or {}

        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)
        reasoning_tokens += (
            usage.get("output_token_details") or {}
        ).get("reasoning", 0)

        for block in message.content if isinstance(message.content, list) else []:
            if not isinstance(block, dict):
                continue

            if block.get("type") == "web_search_call":
                searches.append(block.get("action", {}))

            if block.get("type") == "text":
                for citation in block.get("annotations", []):
                    if citation.get("type") == "url_citation":
                        url = citation.get("url")
                        if url:
                            sources[url] = {
                                "title": citation.get("title"),
                                "url": url,
                            }

    statistics = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": input_tokens + output_tokens,
        "web_actions": len(searches),
    }

    # Save output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in company.lower()
    )

    md_path = OUTPUT_DIR / f"{slug}_{timestamp}.md"
    json_path = OUTPUT_DIR / f"{slug}_{timestamp}.json"

    md_path.write_text(report, encoding="utf-8")

    # Export readable result including message history.
    # Exclude encrypted reasoning payloads.
    serialized = result
    if isinstance(result, dict):
        serialized = {
            **result,
            "messages": [
                message.model_dump(
                    mode="json",
                    exclude={"content": {"__all__": {"encrypted_content"}}},
                )
                for message in messages
            ],
        }

    export = {
        "company": company,
        "website": website,
        "model": MODEL,
        "report": report,
        "statistics": statistics,
        "search_actions": searches,
        "sources": list(sources.values()),
        "result": serialized,
    }

    json_path.write_text(
        json.dumps(export, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    console.print("\n\n[bold cyan]Execution Statistics[/bold]")
    for key, value in statistics.items():
        console.print(f"{key}: {value:,}")

    console.print(f"\n[green]Report:[/green] {md_path}")
    console.print(f"[green]JSON:[/green] {json_path}")

    return report, result


if __name__ == "__main__":
    company = input("Company name: ").strip()
    website = input("Website (optional): ").strip()

    if not company:
        raise ValueError("Company name is required")

    report, result = research_company(company, website)
