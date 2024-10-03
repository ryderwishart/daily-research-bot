import os
import discord
import openai
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import Json
import numpy as np
import requests
import schedule
import time
import asyncio
import re
from openai import OpenAI
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import ast  # Add this import at the top of the file
import io  # Add this import

# Download necessary NLTK data (you may need to run this once)
nltk.download('punkt')
nltk.download('stopwords')

# Load the system prompt from the bot_instructions file
with open('./bot_instructions', 'r') as file:
    system_prompt = file.read().strip()

# Ensure the system prompt is not empty
if not system_prompt:
    raise ValueError("The bot_instructions file is empty or could not be read.")

# Load the environment variables from the .env file
load_dotenv()

# Add this line near the top of the file, after the imports
global recent_messages

# Initialize the OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Connect to your PostgreSQL database
conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    dbname=os.getenv('DB_DATABASE'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    port=os.getenv('DB_PORT')
)
cur = conn.cursor()

# Create a new Discord client
intents = discord.Intents.default()
intents.typing = False
intents.presences = False
intents.message_content = True
discord_client = discord.Client(intents=intents)

MAX_MESSAGE_LENGTH = 2000
MAX_INPUT_LENGTH = 1500  # Adjust as needed to ensure the total length stays within limits

# Modify the declaration of recent_messages
recent_messages = []

def get_markdown_content(url):
    full_url = f"https://r.jina.ai/{url}"
    response = requests.get(full_url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to retrieve markdown content from {full_url}")
        return None

async def get_huggingface_papers():
    url = "https://huggingface.co/api/daily_papers"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        data = response.json()
        
        papers = []
        for paper in data:
            title = paper['paper']['title']
            link = f"https://huggingface.co/papers/{paper['paper']['id']}"
            papers.append({'title': title, 'link': link})
        
        return papers
    except requests.RequestException as e:
        print(f"Error fetching papers from Hugging Face API: {e}")
        return []

async def get_elvissaravia_papers():
    url = "https://nlp.elvissaravia.com/p/top-ml-papers-of-the-week-fbc"
    markdown_content = get_markdown_content(url)
    if markdown_content is None:
        print("Failed to retrieve markdown content from Elvis Saravia's blog.")
        return []
    
    papers = []
    sections = markdown_content.split('\n\n')
    for section in sections:
        # Look for lines starting with bold text
        bold_match = re.search(r'\*\*(.*?)\*\*', section)
        if bold_match:
            title = bold_match.group(1)
        else:
            # If no bold text, take the first 100 characters as the title
            title = section[:100].strip()
        
        # Look for paper link
        link_match = re.search(r'\[paper\]\((.*?)\)', section.strip())
        if link_match:
            link = link_match.group(1)
            papers.append({'title': title, 'link': link})
    
    return papers

def get_paper_content(paper_url):
    markdown_content = get_markdown_content(paper_url)
    if markdown_content is None:
        print(f"Failed to retrieve markdown content for {paper_url}")
        return None
    # Now, we need to extract the abstract from the markdown content
    # Let's assume that the abstract is under the '## Abstract' heading
    abstract = ""
    lines = markdown_content.split('\n')
    in_abstract = False
    for line in lines:
        if '## Abstract' in line:
            in_abstract = True
            continue
        elif line.startswith('## ') and in_abstract:
            # Reached the next section
            break
        elif in_abstract:
            abstract += line + '\n'
    if abstract.strip() == "":
        # If no abstract found, use the entire markdown content as fallback
        abstract = markdown_content
    return abstract.strip()

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

def get_embedding(text, model="text-embedding-3-small", max_tokens=8000):
    text = text.replace("\n", " ")
    
    # First, try removing stopwords
    filtered_text = remove_stopwords(text)
    
    # If still too long, truncate
    if len(filtered_text.split()) > max_tokens:
        filtered_text = ' '.join(filtered_text.split()[:max_tokens])
    
    try:
        return openai_client.embeddings.create(input=[filtered_text], model=model).data[0].embedding
    except openai.BadRequestError as e:
        if "maximum context length" in str(e):
            # If still too long, truncate further
            truncated_text = ' '.join(filtered_text.split()[:max_tokens//2])
            return openai_client.embeddings.create(input=[truncated_text], model=model).data[0].embedding
        else:
            raise e

async def store_summaries(papers):
    for paper in papers:
        # Get the content (e.g., abstract) of the paper
        content = get_paper_content(paper['link'])
        if content is None:
            continue
        embedding = get_embedding(content)
        # Store the embedding as a JSON string
        embedding_json = Json(embedding)
        try:
            await asyncio.to_thread(cur.execute, "INSERT INTO summaries (title, snippet, link, embedding) VALUES (%s, %s, %s, %s)",
                        (paper['title'], content, paper['link'], embedding_json))
            print(f"Stored summary for paper: {paper['title']}")
        except Exception as e:
            print(f"Error storing paper '{paper['title']}': {e}")
    await asyncio.to_thread(conn.commit)

async def daily_task():
    huggingface_papers = await get_huggingface_papers()
    elvissaravia_papers = await get_elvissaravia_papers()
    all_papers = huggingface_papers + elvissaravia_papers
    print(f"Retrieved {len(huggingface_papers)} papers from Hugging Face and {len(elvissaravia_papers)} papers from Elvis Saravia's blog.")
    await store_summaries(all_papers)
    print("Daily summaries stored successfully.")

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

async def get_similar_messages(content, threshold=0.7, top_k=10):
    embedding = get_embedding(content)
    try:
        # First, check if the pgvector extension is available
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        if cur.fetchone() is None:
            print("pgvector extension is not installed. Falling back to basic search.")
            # Fallback to a basic search without vector comparison
            cur.execute("""
                SELECT title, snippet, link
                FROM summaries
                ORDER BY id DESC
                LIMIT %s
            """, (top_k,))
        else:
            # Use pgvector for similarity search with explicit casting
            cur.execute("""
                SELECT title, snippet, link, embedding <-> %s::vector AS distance
                FROM summaries
                ORDER BY distance
                LIMIT %s
            """, (Json(embedding), top_k))
        
        results = cur.fetchall()
        
        if len(results[0]) == 4:  # If we used pgvector
            return [{'title': r[0], 'snippet': r[1], 'link': r[2], 'similarity': 1 - r[3]} for r in results]
        else:  # If we used the fallback method
            return [{'title': r[0], 'snippet': r[1], 'link': r[2], 'similarity': 0} for r in results]
    except Exception as e:
        print(f"Error in get_similar_messages: {e}")
        # Return an empty list if there's an error
        return []

async def call_openai_api(user_message, user_role, context_messages=None):
    global recent_messages
    if len(user_message) > MAX_INPUT_LENGTH:
        user_message = user_message[:MAX_INPUT_LENGTH]

    messages = [
        {"role": "system", "content": system_prompt}
    ]
    if context_messages:
        context_texts = [f"Title: {msg['title']}\nSnippet: {msg['snippet']}\nLink: {msg['link']}" for msg in context_messages]
        context_combined = "\n\n".join(context_texts)
        messages.append({"role": "system", "content": f"Here are some related papers. Mention them in your response if they are relevant, and if you do then be sure to make the title of the paper a link to the paper:\n{context_combined}"})
    messages.extend(recent_messages)
    messages.append({'role': user_role, 'content': user_message})

    try:
        response = await asyncio.to_thread(
            lambda: openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
        )
    except Exception as e:
        if 'insufficient_quota' in str(e):
            print("Insufficient quota for gpt-4o-mini, retrying with gpt-4o")
            response = await asyncio.to_thread(
                lambda: openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages
                )
            )
        else:
            raise e

    response_content = response.choices[0].message.content
    if len(response_content) > MAX_MESSAGE_LENGTH:
        response_content = response_content[:MAX_MESSAGE_LENGTH]

    return response_content

async def get_recent_papers(limit=20):
    cur.execute("SELECT title, snippet, link FROM summaries ORDER BY id DESC LIMIT %s", (limit,))
    return cur.fetchall()

async def summarize_papers_in_batches(papers, batch_size=3):
    batches = [papers[i:i+batch_size] for i in range(0, len(papers), batch_size)]
    all_summaries = []

    for i, batch in enumerate(batches, 1):
        papers_text = "\n\n".join([f"Title: {paper[0]}\nLink: {paper[2]}" for paper in batch])
        prompt = f"Please provide a concise summary of the following {len(batch)} recent papers in one paragraph, focusing on their key findings and potential applications to low-resource language translation tasks. Remember to make the title of the paper a link to the paper:\n\n{papers_text}"

        response = await call_openai_api(prompt, "system")
        all_summaries.append(response)
        yield f"Batch {i}/{len(batches)}:\n{response}\n\n"

    overall_summary_prompt = f"Based on the following summaries of recent papers, provide a brief overall summary highlighting the most important trends and potential applications for low-resource language translation:\n\n{''.join(all_summaries)}"
    overall_summary = await call_openai_api(overall_summary_prompt, "system")
    yield f"Overall Summary:\n{overall_summary}"

# Add this new function after the existing imports
async def generate_audio(text):
    try:
        response = await asyncio.to_thread(
            lambda: openai_client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
        )
        return io.BytesIO(response.content)
    except Exception as e:
        print(f"Error generating audio: {e}")
        # Log the full traceback for debugging
        print(f"Full traceback: {traceback.format_exc()}")
        # Raise the exception to be handled by the caller
        raise

@discord_client.event
async def on_message(message):
    global recent_messages
    if message.author == discord_client.user:
        return

    user_message = message.content.lower()
    user_role = 'user' if message.author != discord_client.user else 'assistant'

    if discord_client.user in message.mentions:
        if any(keyword in user_message for keyword in ["summarize", "papers", "recent research"]):
            initial_response = await message.channel.send("Certainly! I'm checking the most recent papers. This might take a moment...")
            try:
                recent_papers = await get_recent_papers()
                if not recent_papers:
                    await message.channel.send("I couldn't find any recent papers to summarize.")
                    return

                # Create a thread for the summaries
                thread = await initial_response.create_thread(name="Paper Summaries with links", auto_archive_duration=60)

                overall_summary = None
                async for summary in summarize_papers_in_batches(recent_papers):
                    if summary.startswith("Overall Summary:"):
                        overall_summary = summary
                    else:
                        await thread.send(summary)

                # Send the overall summary as a new top-level message
                if overall_summary:
                    await thread.send(overall_summary)
                    await message.channel.send("Generating a voice message for the summary...")
                    
                    # Generate and send voice message
                    audio_file = await generate_audio(overall_summary)
                    if audio_file:
                        await message.channel.send(file=discord.File(audio_file, filename="summary.mp3"))
                    else:
                        await message.channel.send("I couldn't generate a voice message for the summary.")
                else:
                    await message.channel.send("Summary process completed, but no overall summary was generated.")

            except Exception as e:
                print(f"Error summarizing papers: {e}")
                await message.channel.send("Sorry, I encountered an error while summarizing the papers.")
        else:
            await message.channel.send("Checking my knowledge base... Apparently I don't know anything until it's in postgres.")
            try:
                context_messages = await get_similar_messages(user_message)
                if not context_messages:
                    await message.channel.send("I'm having trouble accessing my knowledge base. I'll do my best to answer without additional context.")
                response = await call_openai_api(user_message, user_role, context_messages)
                recent_messages.append({'role': user_role, 'content': user_message})
                recent_messages.append({'role': 'assistant', 'content': response})
                if len(recent_messages) > 20:
                    recent_messages = recent_messages[-20:]
                await message.channel.send(response)
            except Exception as e:
                print(f"Error: {e}")
                await message.channel.send("Sorry, I encountered an error while processing your message.")

@discord_client.event
async def on_ready():
    print(f'We have logged in as {discord_client.user}')

# Schedule the daily task
def run_daily_task():
    asyncio.run(daily_task())

schedule.every().day.at("00:00").do(run_daily_task)

# Run the bot
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the script with optional immediate indexing.")
    parser.add_argument('--index_now', action='store_true', help="Index immediately if this flag is set.")
    args = parser.parse_args()

    if args.index_now:
        print("Starting immediate indexing...")
        asyncio.run(daily_task())
    else:
        print("Starting daily task scheduler...")
    # Start the scheduler in a separate thread
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(1)
    import threading
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.start()

    # Run the Discord bot
    try:
        discord_client.run(os.getenv('DISCORD_BOT_TOKEN'))  # Load the bot token from the environment variable
    except Exception as e:
        print(f"Error: {e}")