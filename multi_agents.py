import os
import asyncio
from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.llms.groq import Groq
from tavily import AsyncTavilyClient


# ---------- Initialization ----------

def get_init_llm():
    """Initialize Groq LLM."""
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    return Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key)


def init_tavily_client():
    """Initialize Tavily API client."""
    load_dotenv()
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    return AsyncTavilyClient(api_key=tavily_api_key)


# ---------- Tools ----------

async def search_web(query: str, client: AsyncTavilyClient) -> str:
    """Search the web for a given query."""
    results = await client.search(query)
    return str(results)


async def record_search_results(ctx: Context, search_results: str) -> str:
    """Record search results in workflow state."""
    async with ctx.store.edit_state as ctx_state:
        ctx_state["state"]["search_results"] = search_results
    return "Search results recorded."


async def write_post(ctx: Context, post_article_content: str) -> str:
    """Write a blog post based on search results."""
    async with ctx.store.edit_state as ctx_state:
        ctx_state["state"]["post_article_content"] = post_article_content
    return "Post article written."


async def improve_seo(ctx: Context, improved_post_article: str) -> str:
    """Improve the SEO of the article and store it."""
    async with ctx.store.edit_state as ctx_state:
        ctx_state["state"]["improved_post_article"] = improved_post_article
    return f"SEO improved and final article stored."

# ---------- Workflow ----------

def initialize_workflow():
    """Set up agents and workflow."""
    llm = get_init_llm()
    client = init_tavily_client()

    # Define tools
    web_search_tool = FunctionTool.from_defaults(
        fn=lambda query: asyncio.run(search_web(query, client)),
        name="web_search",
        description="Search the web for a given topic."
    )

    record_search_results_tool = FunctionTool.from_defaults(
        fn=record_search_results,
        name="record_search_results",
        description="Record search results for later use."
    )

    write_post_tool = FunctionTool.from_defaults(
        fn=write_post,
        name="write_post",
        description="Write a blog post based on search results."
    )

    improve_seo_tool = FunctionTool.from_defaults(
        fn=improve_seo,
        name="improve_seo",
        description="Improve SEO of the blog post."
    )

    # Define agents
    search_agent = FunctionAgent(
        name="SearchAgent",
        description="Search the web for relevant information.",
        system_prompt=(
            "You are a search agent that finds relevant information about the given topic. "
            "After collecting notes, hand off to WritePostAgent."
        ),
        llm=llm,
        tools=[web_search_tool, record_search_results_tool],
        can_handoff_to=["WritePostAgent"],
    )

    write_post_agent = FunctionAgent(
        name="WritePostAgent",
        description="Generate a blog post based on search results.",
        system_prompt=(
            "You are a writer agent that drafts a blog post in markdown format. "
            "When done, hand off to SeoReviewerAgent for optimization."
        ),
        llm=llm,
        tools=[write_post_tool],
        can_handoff_to=["SeoReviewerAgent"],
    )

    seo_agent = FunctionAgent(
        name="SeoReviewerAgent",
        description="Review and improve SEO.",
        system_prompt=(
            "You are an SEO reviewer agent that improves the article's structure and SEO quality."
        ),
        llm=llm,
        tools=[improve_seo_tool],
        can_handoff_to=[],
    )

    # Define workflow
    workflow = AgentWorkflow(
        agents=[search_agent, write_post_agent, seo_agent],
        root_agent=search_agent.name,
        initial_state={
            "search_results": "",
            "post_article_content": "",
            "improved_post_article": "",
        },
    )

    return workflow


def execute_workflow(agent_workflow, user_query: str):
    """Run the agent workflow synchronously for a given query and return the article."""

    async def _run():
        # Run the workflow
        result = await agent_workflow.run(
            user_msg=f"Write a detailed, markdown-formatted blog post about {user_query}"
        )

        # Depending on the version, result may have .response or .state
        improved_article = ""
        post_article = ""
        search_results = ""

        if hasattr(result, "state"):
            improved_article = result.state.get("improved_post_article", "")
            post_article = result.state.get("post_article_content", "")
            search_results = result.state.get("search_results", "")

        # Try to use the final model's response if no state was stored
        if hasattr(result, "response") and isinstance(result.response, str):
            fallback_output = result.response
        else:
            fallback_output = str(result)

        # Fallback priority
        if improved_article:
            return improved_article
        elif post_article:
            return post_article
        elif fallback_output:
            return fallback_output
        else:
            return f"No article text found.\n\nDebug info:\nState: {getattr(result, 'state', {})}\nSearch: {search_results}"

    return asyncio.run(_run())

