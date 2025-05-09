import os
import chainlit as cl
import logging
from typing import List, Dict, cast
from dotenv import load_dotenv
from aiohttp import ClientError
import asyncio
from agents import Agent, Runner, RunConfig, RunContextWrapper, AsyncOpenAI, OpenAIChatCompletionsModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def setup_config() -> tuple[Agent, RunConfig]:
    """Set up the triage agent and configuration."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    external_client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-1.5-flash",
        openai_client=external_client,
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    # Setup Agents
    spanish_agent = Agent(
        name="spanish_agent",
        instructions="You translate the user's message to Spanish",
        handoff_description="An English to Spanish translator",
        model=model
    )

    french_agent = Agent(
        name="french_agent",
        instructions="You translate the user's message to French",
        handoff_description="An English to French translator",
        model=model
    )

    italian_agent = Agent(
        name="italian_agent",
        instructions="You translate the user's message to Italian",
        handoff_description="An English to Italian translator",
        model=model
    )

    # Triage Agent
    triage_agent = Agent(
        name="triage_agent",
        instructions=(
            "You are a translation agent. You use the tools given to you to translate. "
            "If asked for multiple translations, you call the relevant tools in order. "
            "You never translate on your own, you always use the provided tools."
        ),
        tools=[
            spanish_agent.as_tool(
                tool_name="translate_to_spanish",
                tool_description="Translate the user's message to Spanish",
            ),
            french_agent.as_tool(
                tool_name="translate_to_french",
                tool_description="Translate the user's message to French",
            ),
            italian_agent.as_tool(
                tool_name="translate_to_italian",
                tool_description="Translate the user's message to Italian",
            ),
        ],
        model=model
    )

    return triage_agent, config

@cl.on_chat_start
async def start():
    """Initialize the session and send a welcome message."""
    triage_agent, config = setup_config()
    cl.user_session.set("triage_agent", triage_agent)
    cl.user_session.set("config", config)
    cl.user_session.set("chat_history", [])
    await cl.Message(content="Welcome to the Panaversity AI Assistant!").send()

@cl.on_message
async def main(message: cl.Message):
    """Process incoming messages and generate responses."""
    # Send a thinking message
    msg = cl.Message(content="Thinking...")
    await msg.send()

    # Check and reinitialize session state if necessary
    triage_agent = cl.user_session.get("triage_agent")
    config = cl.user_session.get("config")
    if not triage_agent or not config:
        logger.warning("Session state missing, reinitializing...")
        triage_agent, config = setup_config()
        cl.user_session.set("triage_agent", triage_agent)
        cl.user_session.set("config", config)

    triage_agent = cast(Agent, triage_agent)
    config = cast(RunConfig, config)

    # Retrieve and update chat history
    history: List[Dict[str, str]] = cl.user_session.get("chat_history") or []
    history.append({"role": "user", "content": message.content})

    try:
        logger.info("Calling agent with history: %s", history)
        result = Runner.run_sync(triage_agent, history, run_config=config)
        response_content = result.final_output

        # Update the thinking message with the response
        msg.content = response_content
        await msg.update()

        # Append the assistant's response to the history
        history.append({"role": "developer", "content": response_content})
        # WORKAROUND: Using 'developer' due to a bug in the agents library.
        # Should be 'assistant' when the library is fixed.

        # Update the session with the new history
        cl.user_session.set("chat_history", history)

        # Log the interaction
        logger.info("User: %s", message.content)
        logger.info("Assistant: %s", response_content)

    except (ClientError, asyncio.TimeoutError) as e:
        logger.error("Network error: %s", str(e))
        msg.content = f"Network error: {str(e)}"
        await msg.update()
    except ValueError as e:
        logger.error("Invalid input: %s", str(e))
        msg.content = f"Invalid input: {str(e)}"
        await msg.update()
    except Exception as e:
        logger.error("Unexpected error: %s", str(e))
        msg.content = f"Unexpected error: {str(e)}"
        await msg.update()