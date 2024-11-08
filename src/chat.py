from langchain.agents import AgentType
import os
import re
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
embedding = OpenAIEmbeddings()

# Ensure your vector database is populated
vector_db = Chroma(
    persist_directory='data/vector_db',
    embedding_function=embedding
)

llm = OpenAI(temperature=0)

# Initialize memory to keep track of the conversation
memory = ConversationBufferMemory(memory_key="chat_history")


def suggest_song_by_mood(mood):
    # Search for songs matching the mood
    results = vector_db.similarity_search(mood, k=1)
    if results:
        song = results[0].metadata
        title = song.get('title', 'Unknown Title')
        artist = song.get('artist', 'Unknown Artist')
        genre = song.get('genre', 'Unknown Genre')

        prompt = f"""
        You are a creative music assistant. Based on the mood '{mood}', recommend the song '{title}'.
        Provide a brief and engaging explanation (2-3 sentences) on what song are you recommending and why this song matches the '{mood}' mood.
        Consider aspects like lyrics, melody, rhythm, and overall vibe.
        """
        explanation = llm.invoke(prompt)  # Use .invoke() as per deprecation warning

        return explanation.strip()
    else:
        prompt = f"Suggest a popular song that matches the {mood} mood."
        suggestion = llm.invoke(prompt)
        return f"🎶 **Song Suggestion**: Here's a song that matches your '{mood}' mood:\n{suggestion.strip()}"


def provide_song_summary(song_title):
    # Search for the song by title
    results = vector_db.similarity_search(song_title, k=1)
    if results:
        song = results[0].metadata
        title = song.get('title', song_title)
        artist = song.get('artist', 'Unknown Artist')
        mood = song.get('mood', 'Unknown mood')
        genre = song.get('genre', 'Unknown genre')

        prompt = f"Provide a brief summary of the song '{title}' without including any lyrics."
        summary = llm.invoke(prompt)  # Updated to use invoke

        return summary
    else:
        return "Sorry, I couldn't find more details about this song."


def suggest_similar_songs(song_title):
    result = vector_db.similarity_search(song_title, k=1)
    if not result:
        return f"Sorry, I couldn't find '{song_title}' in the database."

    song_metadata = result[0].metadata
    title = song_metadata.get('title', song_title)
    artist = song_metadata.get('artist', 'Unknown Artist')

    prompt = f"Recommend three songs similar to '{title}' in terms of theme, mood, or style."
    recommendations = llm.invoke(prompt)

    return f"If you liked '{title}', you might also enjoy:\n{recommendations}"


tools = [
    Tool(
        name="Mood Suggestion Tool",
        func=suggest_song_by_mood,
        description="Suggests a song based on mood.",
        return_direct=True
    ),
    Tool(
        name="Song Summary Tool",
        func=provide_song_summary,
        description="Provides a detailed summary of a song.",
        return_direct=True
    ),
    Tool(
        name="Similar Songs Tool",
        func=suggest_similar_songs,
        description="Recommends similar songs based on a given title.",
        return_direct=True
    )
]


agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

# Define a custom prompt template for the QA chain
prompt_template = PromptTemplate(
    template="""
You are an AI assistant that provides information about songs based on the context provided.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# Create the ConversationalRetrievalChain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_db.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={'prompt': prompt_template},
    verbose=True
)


def main():
    print("Chatbot: Hi! Feel free to ask me anything about songs in our collection.")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Chatbot: Goodbye!")
            break
        try:
            response = agent.run(user_input)
            print("Chatbot:", response)
        except Exception as e:
            print(f"Error: {e}")
            print("Chatbot: Sorry, I couldn't process your request.")


if __name__ == "__main__":
    main()
