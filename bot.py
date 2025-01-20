#!/usr/bin/env python3
"""
Telegram AI Therapist Bot using OpenAI's ChatCompletion.
This bot:
 - Responds to /start and /help with usage instructions.
 - Resets conversation context with /new.
 - Forwards all other text messages to OpenAI for "therapeutic" style replies.
 - Maintains up to 10 messages of context per user session.

Deployment:
-----------
1. Set the environment variables TELEGRAM_BOT_TOKEN and OPENAI_API_KEY.
2. (Optional) Create a .env file in the same directory with:
    TELEGRAM_BOT_TOKEN=your-telegram-bot-token
    OPENAI_API_KEY=your-openai-api-key

3. Install dependencies from requirements.txt:
    pip install -r requirements.txt

4. Run the bot:
    python bot.py

5. (For hosting on Selectel/PQ hosting) Follow the instructions later in this file.
"""

import os
import logging
import openai

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# Load environment variables from .env file if it exists (for local dev convenience).
load_dotenv()

# Retrieve credentials from environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# In-memory dictionary to store user conversations
# Key: chat_id, Value: list of conversation messages (dicts for ChatCompletion)
user_conversations = {}

# Maximum number of messages to store per user
MAX_CONTEXT_MESSAGES = 10


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for /start command. Sends a welcome message and basic instructions.
    """
    welcome_text = (
        "Hello! I'm your AI Therapist Bot. I am here to listen and offer supportive responses.\n\n"
        "You can talk to me about anything on your mind, and I'll do my best to help. "
        "If you need more detailed instructions, type /help."
    )
    await update.message.reply_text(welcome_text)

    # Initialize user conversation in memory if not exists
    chat_id = update.effective_chat.id
    if chat_id not in user_conversations:
        user_conversations[chat_id] = []


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for /help command. Provides instructions on how to use the bot.
    """
    help_text = (
        "Here are some commands you can use:\n\n"
        "/start - Welcome message.\n"
        "/help - This help message.\n"
        "/new - Reset your conversation context.\n\n"
        "Otherwise, just type your message, and I'll reply!"
    )
    await update.message.reply_text(help_text)


async def new_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for /new command. Clears the conversation context for the user.
    """
    chat_id = update.effective_chat.id
    user_conversations[chat_id] = []
    await update.message.reply_text("Conversation context has been reset. Feel free to start anew!")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for regular text messages.
    Forwards message to OpenAI ChatCompletion and returns the AI's response.
    """
    chat_id = update.effective_chat.id
    user_text = update.message.text

    # Log user message
    logging.info(f"User ({chat_id}) says: {user_text}")

    # Initialize or retrieve conversation context
    if chat_id not in user_conversations:
        user_conversations[chat_id] = []

    conversation_history = user_conversations[chat_id]

    # Append user's message to the conversation
    conversation_history.append({"role": "user", "content": user_text})

    # Truncate the conversation history if it exceeds MAX_CONTEXT_MESSAGES
    if len(conversation_history) > MAX_CONTEXT_MESSAGES:
        conversation_history = conversation_history[-MAX_CONTEXT_MESSAGES:]

    # Prepare messages for ChatCompletion
    # We can add a system prompt to help the model act as a therapist:
    system_prompt = (
        "You are a supportive, empathetic AI therapist. You listen attentively and provide helpful, "
        "gentle, and understanding responses. You are not a medical professional, but you offer "
        "a comforting presence and coping strategies."
    )

    messages_for_openai = [
        {"role": "system", "content": system_prompt}
    ] + conversation_history

    # Send the request to OpenAI
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages_for_openai,
            temperature=0.7,
        )
        ai_reply = response["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"OpenAI API Error: {e}")
        await update.message.reply_text(
            "I'm sorry, but I'm having trouble connecting to my brain right now. "
            "Please try again later."
        )
        return

    # Append AI response to the conversation
    conversation_history.append({"role": "assistant", "content": ai_reply})

    # Update the global conversation list with truncated version
    user_conversations[chat_id] = conversation_history[-MAX_CONTEXT_MESSAGES:]

    # Send the AI's reply to the user
    await update.message.reply_text(ai_reply)


def main():
    """
    Main entry point that sets up the Bot and starts polling.
    
    Instructions to run:
    1) Install dependencies: pip install -r requirements.txt
    2) Set environment variables or create a .env file with TELEGRAM_BOT_TOKEN and OPENAI_API_KEY
    3) Run this script: python bot.py
    """
    # Build the application
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("new", new_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot
    application.run_polling()


if __name__ == "__main__":
    main()
