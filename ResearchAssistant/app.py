from flask import Flask, render_template, request, jsonify
import ollama
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.llms import Ollama
from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
import wikipedia
from duckduckgo_search import DDGS
import requests
from typing import Optional, Dict, Any, List
import xml.etree.ElementTree as ET
import json
from typing import Union
import re

app = Flask(__name__)

# Initialize Llama model using Ollama
llama = Ollama(model="llama3.1")


def safe_wikipedia_search(query: str) -> str:
    """Safely search Wikipedia with error handling"""
    try:
        query = query.strip('"\'').replace('+', ' ')
        search_results = wikipedia.search(query, results=1)
        if not search_results:
            return f"No Wikipedia articles found for '{query}'"
        
        page_title = search_results[0]
        page = wikipedia.page(page_title, auto_suggest=False)
        return page.summary[0:500]
        
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            return page.summary[0:500]
        except:
            return f"Multiple Wikipedia articles found for '{query}'. Please be more specific."
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia article found for '{query}'"
    except Exception as e:
        return f"An error occurred while searching Wikipedia: {str(e)}"

def safe_duckduckgo_search(query: str) -> str:
    """Safely search DuckDuckGo with error handling"""
    try:
        query = query.strip('"\'').replace('+', ' ')
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        
        if not results:
            return "No results found on DuckDuckGo."
            
        formatted_results = []
        for r in results:
            link = r.get('link', 'No link available')
            body = r['body'][:200] + '...' if 'body' in r else ''
            formatted_results.append(f"- {r['title']}\n  {body}\n  Source: {link}")
        
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"An error occurred while searching DuckDuckGo: {str(e)}"

def safe_math_eval(expression: str) -> str:
    """Safely evaluate mathematical expressions"""
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Invalid mathematical expression. Only basic operations are allowed."
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating mathematical expression: {str(e)}"

def search_arxiv(query: str) -> str:
    """Search ARXiv with improved response parsing"""
    if not query:
        return "No query provided."
    try:
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=3"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.text)
        results = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
            link = entry.find('{http://www.w3.org/2005/Atom}id').text
            results.append(f"Title: {title}\nLink: {link}\nSummary: {summary[:200]}...\n")
        
        return "\n".join(results) if results else "No results found on arXiv."
    except Exception as e:
        return f"Error searching ArXiv: {str(e)}"

def search_pubmed(query: str) -> str:
    """Search PubMed with improved response handling"""
    if not query:
        return "No query provided."
    try:
        esearch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmax=3&format=json"
        response = requests.get(esearch_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        ids = data.get('esearchresult', {}).get('idlist', [])
        
        if not ids:
            return "No results found on PubMed."
            
        ids_string = ",".join(ids)
        esummary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={ids_string}&format=json"
        response = requests.get(esummary_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        for id in ids:
            paper = data['result'][id]
            title = paper.get('title', 'No title available')
            abstract = paper.get('abstract', 'No abstract available')
            results.append(f"Title: {title}\nPubMed ID: {id}\nAbstract: {abstract[:200]}...\n")
            
        return "\n".join(results)
    except Exception as e:
        return f"Error searching PubMed: {str(e)}"

# Define the tools
tools = [
    Tool(
        name="Wikipedia",
        func=safe_wikipedia_search,
        description="Get detailed explanations and summaries from Wikipedia. who, what, when, where, why, how, story.",
    ),
    Tool(
        name="DuckDuckGo",
        func=safe_duckduckgo_search,
        description="Search the web for current information. Input: search query. who, what, when, where, why, how.",
    ),
    Tool(
        name="BasicMath",
        func=safe_math_eval,
        description="Performs basic mathematical calculations. Input: mathematical expression using +, -, *, /, ().",
    ),
    Tool(
        name="ArXiv",
        func=search_arxiv,
        description="Searches arXiv for scientific papers. research paper topic or keywords.",
    ),
    Tool(
        name="PubMed",
        func=search_pubmed,
        description="Searches PubMed for medical research. research paper on medical topic or keywords.",
    ),
]

tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

CUSTOM_PROMPT = f"""You are a helpful research assistant that finds information using the available tools. Follow these guidelines strictly:

Available tools:

{tool_descriptions}

Instructions:
. Details from Internal Knowledge: Related information you already know
. Add line breaks between sections
. Use the exact tool name from the list above
. Keep queries simple and clear
. Use tools according to the type of information needed
. In case of research paper only use ArXiv and Pubmed
. Summarize information from multiple sources when relevant
. Stop as soon as you have a clear answer with reference links

Use this exact format:

Question: the input question you must answer
Thought: analyze the question and decide which tool to use
Action: use EXACTLY one of these tools: Wikipedia, DuckDuckGo, BasicMath, ArXiv, or PubMed
Action Input: just the plain search query or math expression
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat up to 3 times if needed)
Thought: I now know the final answer
Final Answer: provide a complete answer based on the information gathered with link (Details from Web Search: 
- First result title and brief description
- Second result title and brief description

References:
[1] link1
[2] link2
[etc])

Begin!

Question: {{input}}
Thought:"""

# Initialize the agent
agent = initialize_agent(
    tools,
    llama,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={
        "prefix": CUSTOM_PROMPT,
        "max_iterations": 3,
        "early_stopping_method": "generate",

    },
    handle_parsing_errors=True
)

@app.route('/')
def home():
    return render_template('index.html', tools=tools)

@app.route('/search', methods=['POST'])
def search():
    try:
        user_input = request.json.get('query')
        if not user_input:
            return jsonify({'error': 'No query provided'}), 400
        
        result = agent.run(user_input)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)