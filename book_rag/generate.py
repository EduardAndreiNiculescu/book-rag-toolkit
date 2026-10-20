"""Framework prompts: retrieve the most relevant book passages, then brief an LLM.

Each framework pairs a retrieval query with a prompt template. Without an API
key the compiled prompt (instructions + retrieved passages) is written to a
file you can paste into any chat model.
"""
from __future__ import annotations

import os
from pathlib import Path

from .search import ChunkSearch

FRAMEWORKS = {
    "offer": {
        "title": "Hormozi Grand Slam Offer Generator",
        "search_query": "grand slam offer value equation pricing risk reversal bonus guarantee",
        "system_instruction": "You are a world-class marketer and expert in Alex Hormozi's '$100M Offers' framework. Your goal is to construct an irresistible 'Grand Slam Offer' for a business.",
        "user_template": """Based on the following product/service details, construct an irresistible Grand Slam Offer:
- Product/Service: {product}
- Target Audience: {audience}
- Price point (if known): {price}

Use these insights from "$100M Offers":
{context}

Structure your response to include:
1. The Core Offer (solving the primary problem)
2. Pricing Strategy (High-ticket premium value vs commodity)
3. The Value Equation breakdown (Dream Outcome, Likelihood of Achievement, Time Delay, Effort/Sacrifice)
4. Bonuses (what else is included to increase perceived value)
5. Guarantees (Risk Reversal)
6. Scarcity & Urgency elements
"""
    },
    "outreach": {
        "title": "Outreach & Lead Gen Campaign Designer",
        "search_query": "warm outreach cold outreach warm leads direct message advertising lead magnet",
        "system_instruction": "You are an outreach and lead generation expert skilled in Alex Hormozi's '$100M Leads' framework.",
        "user_template": """Create an outreach campaign structure and script templates based on:
- Business/Product: {product}
- Target Audience: {audience}
- Goal (e.g., book a call, opt-in): {price}

Use these insights from "$100M Leads":
{context}

Provide:
1. Lead Magnet definition (how to provide value first)
2. Warm Outreach framework (scripts for reaching out to existing contacts/network)
3. Cold Outreach template (email or direct message)
4. Strategy for scaling the campaign (more, better, or new channels)
"""
    },
    "viral": {
        "title": "Virality & Contagious Hook Generator",
        "search_query": "stepps social currency triggers emotion public practical value stories word of mouth",
        "system_instruction": "You are a virality expert trained on Jonah Berger's 'Contagious' (STEPPS) and Ryan Holiday's 'Trust Me, I'm Lying' frameworks.",
        "user_template": """Generate viral hooks and promotional strategies for:
- Product/Service: {product}
- Target Audience: {audience}
- Campaign Goal: {price}

Use these insights from "Contagious" and "Trust Me, I'm Lying":
{context}

Draft ideas for:
1. Social Currency (making the customer look good/cool by sharing)
2. Triggers (linking the product/message to everyday stimuli)
3. Emotion (how to evoke high-arousal emotions like awe, anger, or excitement)
4. Practical Value (highly useful tips/guides that people want to pass along)
5. Stories (wrapping the core brand message inside a narrative)
"""
    },
    "persuasion": {
        "title": "Cialdini Persuasion Copywriter",
        "search_query": "reciprocity commitment consistency social proof liking authority scarcity persuasion",
        "system_instruction": "You are an expert copywriter skilled in Robert Cialdini's principles of influence and persuasion.",
        "user_template": """Write high-converting sales copy incorporating Cialdini's persuasion principles for:
- Product/Service: {product}
- Target Audience: {audience}

Use these insights from "Influence":
{context}

Draft copy/sections highlighting:
1. Reciprocity (what free value or goodwill can you offer first)
2. Commitment & Consistency (getting small initial agreements or micro-commitments)
3. Social Proof (strategizing testimonials or proof elements)
4. Liking (building affinity and connection)
5. Authority (demonstrating status, credentials, or expertise)
6. Scarcity (creating authentic limits)
"""
    }
}


def build_prompt(engine: ChunkSearch, framework: str, product: str, audience: str, details: str = "", k: int = 4) -> tuple[str, str]:
    cfg = FRAMEWORKS[framework]
    context = "".join(
        f"\n--- {c.get('metadata', {}).get('title', c.get('source_file', ''))} "
        f"(chapter: {c.get('metadata', {}).get('chapter', '?')}) ---\n{c.get('raw_text', c.get('text', ''))}\n"
        for _, c in engine.search(cfg["search_query"], limit=k)
    )
    return cfg["system_instruction"], cfg["user_template"].format(product=product, audience=audience, price=details, context=context)


def call_llm(system: str, user: str) -> str | None:
    """Uses Gemini or OpenAI if a key is set (model overridable via LLM_MODEL); otherwise None."""
    if os.environ.get("GEMINI_API_KEY"):
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(os.environ.get("LLM_MODEL", "gemini-2.5-flash"), system_instruction=system)
        return model.generate_content(user).text
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI

        resp = OpenAI().chat.completions.create(
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return resp.choices[0].message.content
    return None
