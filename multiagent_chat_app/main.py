import os
import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import httpx
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = 'XXXX'
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

app = FastAPI()

# Allow CORS for local fronten
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    sender: str
    text: str

class ChatRequest(BaseModel):
    username: str
    message: str
    history: List[Message]

def build_agent_prompt(agent: str, history: List[Message], user_message: str, username: str = None) -> str:
    base_prompts = {
        "john": (
            "You are Agent John, an AI chef. Introduce yourself as Agent John, an AI agent for culinary expertise. "
            "Respond in a natural, conversational, and context-aware way. Always keep your responses to the user precise and concise—never overload with unnecessary or excessive information. "
            "Only include sections (Meal Plan, Cooking Steps, Grocery List, Substitutions, Budget Estimate, Clarification) that are relevant for the current step. "
            "Do NOT fill all sections by default—only show content that is ready or needed. If you need more info, just ask for it. "
            "Be concise, friendly, and avoid repeating unnecessary boilerplate. "
            "When ready for review, tag @sara. "
            "If Agent Sara provides nutrition or portion control advice, update your plan accordingly and always provide the budget in INR. "
            "Always use markdown for lists and structure, but keep the conversation human and adaptive."
        ),
        "sara": (
            "You are Agent Sara, an AI nutritionist. Introduce yourself as Agent Sara, an AI agent for nutrition and dietary review. "
            "Your job is to review meal plans for allergen compliance, budget ceiling, kcal (calorie) count, balanced diet, portion control, and overall nutritional adequacy. Always provide portion control advice as part of your review. "
            "Respond in a natural, conversational, and context-aware way. Always keep your responses to the user precise and concise—never overload with unnecessary or excessive information. "
            "Only include sections (Review, Required Fixes, Approval) that are relevant for the current step. "
            "Do NOT fill all sections by default—only show content that is ready or needed. If you need more info, just ask for it. "
            "Be concise, friendly, and avoid repeating unnecessary boilerplate. "
            "When ready for rectification, tag @john. When plan is ready, tag the user. "
            "Always use markdown for lists and structure, but keep the conversation human and adaptive."
        ),
        "aisha": (
            "You are Agent Aisha, an AI household budget planner. Introduce yourself as Agent Aisha, an AI agent for planning and organizing household budgets. "
            "Help the user organize expenses, plan savings, and create a monthly or weekly budget. "
            "Respond in a natural, conversational, and context-aware way. Keep your responses concise and actionable. "
            "When your plan is ready for review, tag @mack. "
            "If Agent Mack provides feedback, update your plan accordingly. "
            "Always use markdown for lists and structure, but keep the conversation human and adaptive."
        ),
        "mack": (
            "You are Agent Mack, an AI household budget reviewer. Introduce yourself as Agent Mack, an AI agent for reviewing and improving household budgets. "
            "Your job is to review Aisha's plans for accuracy, completeness, and suggest improvements or corrections. "
            "Respond in a natural, conversational, and context-aware way. Keep your responses concise and actionable. "
            "When your review is complete, tag @aisha for any fixes or tag the user if the plan is ready. "
            "Always use markdown for lists and structure, but keep the conversation human and adaptive."
        ),
    }
    chat_history = "\n".join([f"{m.sender}: {m.text}" for m in history])
    user_line = f"User's name is: {username}" if username else ""
    return f"{base_prompts[agent]}\n\n{user_line}\n\nChat so far:\n{chat_history}\n\nNew message: {user_message}\n"

def bold_agent_mentions(text: str) -> str:
    # Bold @john and @sara tags for agent-to-agent communication
    import re
    return re.sub(r'(@john|@sara)', r'**\1**', text, flags=re.IGNORECASE)

def format_markdown(text: str) -> str:
    # Ensure double newlines after headings and before lists, and between sections
    import re
    text = bold_agent_mentions(text)
    # Add newline after headings (fix regex)
    text = re.sub(r'(\*\*[^\*]+\*\*:)', r'\1\n\n', text)
    # Add newline before numbered and bullet lists
    text = re.sub(r'(\n)([0-9]+\.)', r'\1\n\2', text)
    text = re.sub(r'(\n)(- )', r'\1\n\2', text)
    # Remove accidental double spaces after colons
    text = re.sub(r': +', ': ', text)
    # Remove extra single newlines between sections
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

async def call_gemini(prompt: str, delay: float = 1.5) -> str:
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    try:
        await asyncio.sleep(delay)  # Simulate typing delay
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(GEMINI_API_URL, headers=headers, json=data)
            resp.raise_for_status()
            result = resp.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            text = format_markdown(text)
            return text
    except httpx.ReadTimeout:
        return "Sorry, the AI service is taking too long to respond. Please try again later."

def is_household_channel(channel: str) -> bool:
    channel = channel.strip().lower().replace('#', '').replace('-', ' ').replace('_', ' ')
    channel = ' '.join(channel.split())
    return channel in ["household budget", "household"]

@app.post("/chat")
async def chat(req: ChatRequest):
    # Only handle multi-agent logic for culinary channel; for household budget, just return dummy UI messages
    if hasattr(req, 'channel') and req.channel and is_household_channel(req.channel):
        # Dummy UI messages for household channel
        return {"history": [
            Message(sender="aisha", text=f"Hi {req.username}, I'm Agent Aisha, your household budget planner. How can I help you organize your expenses or plan your savings today?"),
            Message(sender="mack", text=f"Hi {req.username}, I'm Agent Mack, your budget reviewer. I'll review your plans for accuracy and suggest improvements!")
        ]}
    # Always use 'user' as sender for user messages in history for frontend compatibility
    # Avoid duplicate user messages in history
    history = []
    last_user = None
    for m in req.history:
        if m.sender.lower() == req.username.lower():
            if last_user == m.text:
                continue  # skip duplicate
            history.append(Message(sender="user", text=m.text))
            last_user = m.text
        else:
            history.append(m)
            last_user = None
    # Only append the new user message if it's not a duplicate of the last
    if not history or not (history[-1].sender == "user" and history[-1].text == req.message):
        history.append(Message(sender="user", text=req.message))

    responses = []
    msg_lower = req.message.lower()
    john_mentioned = ("@john" in msg_lower) or ("john" in msg_lower)
    sara_mentioned = ("@sara" in msg_lower) or ("sara" in msg_lower)

    def count_agent_turns(hist, user):
        count = 0
        for m in reversed(hist):
            if m.sender == "user":
                break
            if m.sender in ["john", "sara"]:
                count += 1
        return count

    max_agent_turns = 2
    agent_turns = count_agent_turns(history, "user")

    # Helper to address user by name only in greeting or as needed
    def address_user(text, is_greeting=False):
        if is_greeting and req.username:
            return f"Hi {req.username}, {text}".strip()
        return text

    # Multi-agent logic: John (chef) should get user input (max 2 turns), then get plan reviewed by Sara (nutritionist), then John incorporates changes and provides final response.
    # Only the relevant agent should be shown as typing (frontend now handles this)
    if john_mentioned and not sara_mentioned and agent_turns < max_agent_turns:
        # John responds to user (max 2 turns)
        responses.append({"sender": "john", "text": "typing..."})
        prompt = build_agent_prompt("john", history, req.message, req.username)
        john_reply = await call_gemini(prompt, delay=2.0)
        is_greeting = len([m for m in history if m.sender == "john"]) == 0
        john_reply = address_user(john_reply.strip(), is_greeting)
        responses[-1]["text"] = john_reply
        agent_turns += 1
        # If John is ready for review, trigger Sara's review
        if ("plan is ready" in john_reply.lower() or "@sara" in john_reply.lower()) and agent_turns <= max_agent_turns:
            await asyncio.sleep(1.5)
            history2 = history + [Message(sender="john", text=john_reply)]
            responses.append({"sender": "sara", "text": "typing..."})
            prompt2 = build_agent_prompt("sara", history2, john_reply, req.username)
            sara_reply = await call_gemini(prompt2, delay=2.0)
            is_greeting = len([m for m in history2 if m.sender == "sara"]) == 0
            sara_reply = address_user(sara_reply.strip(), is_greeting)
            responses[-1]["text"] = sara_reply
            # John incorporates Sara's feedback and provides final response
            await asyncio.sleep(1.5)
            history3 = history2 + [Message(sender="sara", text=sara_reply)]
            responses.append({"sender": "john", "text": "typing..."})
            prompt3 = build_agent_prompt("john", history3, sara_reply, req.username)
            john_reply2 = await call_gemini(prompt3, delay=2.0)
            john_reply2 = address_user(john_reply2.strip(), False)
            responses[-1]["text"] = john_reply2
    elif sara_mentioned and not john_mentioned and agent_turns < max_agent_turns:
        # Sara responds to user (max 2 turns)
        responses.append({"sender": "sara", "text": "typing..."})
        prompt = build_agent_prompt("sara", history, req.message, req.username)
        sara_reply = await call_gemini(prompt, delay=2.0)
        is_greeting = len([m for m in history if m.sender == "sara"]) == 0
        sara_reply = address_user(sara_reply.strip(), is_greeting)
        responses[-1]["text"] = sara_reply
        agent_turns += 1
        # If Sara tags John, trigger John's response
        if ("@john" in sara_reply.lower() or "john" in sara_reply.lower()) and agent_turns <= max_agent_turns:
            await asyncio.sleep(1.5)
            history2 = history + [Message(sender="sara", text=sara_reply)]
            responses.append({"sender": "john", "text": "typing..."})
            prompt2 = build_agent_prompt("john", history2, sara_reply, req.username)
            john_reply = await call_gemini(prompt2, delay=2.0)
            john_reply = address_user(john_reply.strip(), False)
            responses[-1]["text"] = john_reply
    elif john_mentioned and sara_mentioned and agent_turns < max_agent_turns:
        # If both are mentioned, start with John, then Sara, then John again
        responses.append({"sender": "john", "text": "typing..."})
        prompt_john = build_agent_prompt("john", history, req.message, req.username)
        john_reply = await call_gemini(prompt_john, delay=2.0)
        is_greeting = len([m for m in history if m.sender == "john"]) == 0
        john_reply = address_user(john_reply.strip(), is_greeting)
        responses[-1]["text"] = john_reply
        agent_turns += 1
        await asyncio.sleep(1.5)
        history2 = history + [Message(sender="john", text=john_reply)]
        responses.append({"sender": "sara", "text": "typing..."})
        prompt_sara = build_agent_prompt("sara", history2, req.message, req.username)
        sara_reply = await call_gemini(prompt_sara, delay=2.0)
        is_greeting = len([m for m in history2 if m.sender == "sara"]) == 0
        sara_reply = address_user(sara_reply.strip(), is_greeting)
        responses[-1]["text"] = sara_reply
        # John incorporates Sara's feedback and provides final response
        await asyncio.sleep(1.5)
        history3 = history2 + [Message(sender="sara", text=sara_reply)]
        responses.append({"sender": "john", "text": "typing..."})
        prompt3 = build_agent_prompt("john", history3, sara_reply, req.username)
        john_reply2 = await call_gemini(prompt3, delay=2.0)
        john_reply2 = address_user(john_reply2.strip(), False)
        responses[-1]["text"] = john_reply2
    # Only return agent messages, not synthetic user messages
    return {"history": history + [Message(**r) for r in responses]}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/chat/channel/{channel}")
def get_channel_intro(channel: str, username: str = "{username}"):
    def greet_agent(agent, username):
        if agent == "john":
            return f"Hi {username}, I'm Agent John, your AI chef. I can help you with meal plans, recipes, and budgeting. Tag me with @john or just 'John' to get started!"
        if agent == "sara":
            return f"Hi {username}, I'm Agent Sara, your AI nutritionist. I'll review your meal plans for nutrition, allergies, and portion control. Tag me with @sara or just 'Sara' for advice!"
        if agent == "aisha":
            return f"Hi {username}, I'm Agent Aisha, your household budget planner. I can help you organize expenses and plan savings. Tag me with @aisha or just 'Aisha' to get started!"
        if agent == "mack":
            return f"Hi {username}, I'm Agent Mack, your budget reviewer. I'll review your plans for accuracy and suggest improvements. Tag me with @mack or just 'Mack' for feedback!"
        return ""
    if is_household_channel(channel):
        return {
            "messages": [
                {"sender": "aisha", "text": greet_agent("aisha", username)},
                {"sender": "mack", "text": greet_agent("mack", username)}
            ]
        }
    elif channel.strip().lower() in ["culinary", "# culinary"]:
        return {
            "messages": [
                {"sender": "john", "text": greet_agent("john", username)},
                {"sender": "sara", "text": greet_agent("sara", username)}
            ]
        }
    else:
        return {"messages": []}

@app.post("/chat/join")
async def chat_join(req: Request):
    data = await req.json()
    channel = data.get("channel", "culinary")
    username = data.get("username", "{username}")
    def greet_agent(agent, username):
        if agent == "john":
            return f"Hi {username}, I'm Agent John, your AI chef. I can help you with meal plans, recipes, and budgeting. Tag me with @john or just 'John' to get started!"
        if agent == "sara":
            return f"Hi {username}, I'm Agent Sara, your AI nutritionist. I'll review your meal plans for nutrition, allergies, and portion control. Tag me with @sara or just 'Sara' for advice!"
        if agent == "aisha":
            return f"Hi {username}, I'm Agent Aisha, your household budget planner. I can help you organize expenses and plan savings. Tag me with @aisha or just 'Aisha' to get started!"
        if agent == "mack":
            return f"Hi {username}, I'm Agent Mack, your budget reviewer. I'll review your plans for accuracy and suggest improvements. Tag me with @mack or just 'Mack' for feedback!"
        return ""
    if is_household_channel(channel):
        return {
            "messages": [
                {"sender": "aisha", "text": greet_agent("aisha", username)},
                {"sender": "mack", "text": greet_agent("mack", username)}
            ]
        }
    elif channel.strip().lower() in ["culinary", "# culinary"]:
        return {
            "messages": [
                {"sender": "john", "text": greet_agent("john", username)},
                {"sender": "sara", "text": greet_agent("sara", username)}
            ]
        }
    else:
        return {"messages": []}

@app.post("/chat/message")
async def chat_message(req: Request):
    data = await req.json()
    # Map frontend payload to backend ChatRequest
    username = data.get("username", "user")
    message = data.get("text", "")
    history = data.get("history", [])
    # Convert history to Message objects if present
    msg_objs = [Message(**m) if isinstance(m, dict) else m for m in history]
    chat_req = ChatRequest(username=username, message=message, history=msg_objs)
    return await chat(chat_req)

# Serve the frontend
@app.get("/")
async def serve_frontend():
    """Serve the main chat interface"""
    return FileResponse("index.html")

@app.get("/index.html")
async def serve_index():
    """Serve the index.html file directly"""
    return FileResponse("index.html")
