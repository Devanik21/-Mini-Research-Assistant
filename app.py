import streamlit as st
import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import re
from urllib.parse import quote_plus
import asyncio
import aiohttp
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional
import nltk
from collections import Counter
import yfinance as yf

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

@dataclass
class ResearchResult:
    title: str
    url: str
    snippet: str
    timestamp: datetime
    source: str
    sentiment: float
    relevance_score: float

class AdvancedResearchAssistant:
    def __init__(self):
        self.cache = {}
        self.session_history = []
        
    def get_cache_key(self, query: str, filters: dict) -> str:
        """Generate cache key for query and filters"""
        content = f"{query}_{json.dumps(filters, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using TextBlob"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keywords from text"""
        try:
            stop_words = set(stopwords.words('english'))
            word_tokens = word_tokenize(text.lower())
            filtered_words = [w for w in word_tokens if w.isalnum() and w not in stop_words and len(w) > 3]
            word_freq = Counter(filtered_words)
            return [word for word, _ in word_freq.most_common(top_n)]
        except:
            return []
    
    def calculate_relevance_score(self, query: str, title: str, snippet: str) -> float:
        """Calculate relevance score based on query match"""
        query_words = set(query.lower().split())
        content_words = set((title + " " + snippet).lower().split())
        
        if not query_words:
            return 0.0
            
        intersection = len(query_words.intersection(content_words))
        return intersection / len(query_words)
    
    async def search_duckduckgo(self, session: aiohttp.ClientSession, query: str, num_results: int = 10) -> List[Dict]:
        """Search using DuckDuckGo API"""
        try:
            url = "https://api.duckduckgo.com/"
            params = {'q': query, 'format': 'json', 'no_html': '1', 'skip_disambig': '1'}
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    # Get instant answer if available
                    if data.get('Abstract'):
                        results.append({
                            'title': data.get('AbstractText', 'DuckDuckGo Instant Answer'),
                            'url': data.get('AbstractURL', ''),
                            'snippet': data.get('Abstract', ''),
                            'source': 'DuckDuckGo Instant'
                        })
                    
                    # Get related topics
                    for topic in data.get('RelatedTopics', [])[:num_results]:
                        if isinstance(topic, dict) and 'Text' in topic:
                            results.append({
                                'title': topic.get('Text', '')[:100] + '...',
                                'url': topic.get('FirstURL', ''),
                                'snippet': topic.get('Text', ''),
                                'source': 'DuckDuckGo'
                            })
                            
                    return results[:num_results]
        except Exception as e:
            st.error(f"DuckDuckGo search error: {str(e)}")
        return []
    
    async def search_wikipedia(self, session: aiohttp.ClientSession, query: str, num_results: int = 5) -> List[Dict]:
        """Search Wikipedia"""
        try:
            search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
            encoded_query = quote_plus(query)
            
            async with session.get(f"{search_url}{encoded_query}", timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    return [{
                        'title': data.get('title', ''),
                        'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                        'snippet': data.get('extract', ''),
                        'source': 'Wikipedia'
                    }]
        except:
            pass
            
        # Fallback to search API
        try:
            search_api = "https://en.wikipedia.org/w/api.php"
            params = {'action': 'query', 'format': 'json', 'list': 'search', 'srsearch': query, 'srlimit': num_results}
            async with session.get(search_api, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    for item in data.get('query', {}).get('search', []):
                        title = item.get('title', '')
                        results.append({
                            'title': title,
                            'url': f"https://en.wikipedia.org/wiki/{quote_plus(title)}",
                            'snippet': re.sub(r'<[^>]+>', '', item.get('snippet', '')),
                            'source': 'Wikipedia'
                        })
                    return results
        except Exception as e:
            st.error(f"Wikipedia search error: {str(e)}")
        return []
    
    async def get_stock_data(self, symbol: str) -> Dict:
        """Get stock data using yfinance"""
        try:
            # yfinance is not natively async, so we run it in a thread pool
            loop = asyncio.get_event_loop()
            def sync_yf_call():
                stock = yf.Ticker(symbol)
                info = stock.info
                hist = stock.history(period="1mo")
                return {
                    'symbol': symbol,
                    'name': info.get('longName', symbol),
                    'price': info.get('currentPrice', 0),
                    'change': info.get('regularMarketChangePercent', 0),
                    'volume': info.get('volume', 0),
                    'market_cap': info.get('marketCap', 0),
                    'history': hist.to_dict('records') if not hist.empty else []
                }
            return await loop.run_in_executor(None, sync_yf_call)
        except Exception as e:
            st.error(f"Stock data error: {str(e)}")
            return {}
    
    async def comprehensive_search(self, query: str, filters: Dict, max_results: int = 20) -> List[ResearchResult]:
        """Perform comprehensive search across multiple sources"""
        cache_key = self.get_cache_key(query, filters)
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        tasks = []
        async with aiohttp.ClientSession() as session:
            # Search sources based on filters
            if filters.get('include_web', True):
                tasks.append(self.search_duckduckgo(session, query, max_results // 2))
            
            if filters.get('include_wikipedia', True):
                # Allocate a portion of max_results to Wikipedia, ensuring at least 1.
                tasks.append(self.search_wikipedia(session, query, max(1, max_results // 4)))
            
            # Gather results from all tasks
            results_from_sources = await asyncio.gather(*tasks)
        
        all_results = []
        for source_result in results_from_sources:
            all_results.extend(source_result)
        
        # Convert to ResearchResult objects
        research_results = []
        for result in all_results:
            if result and result.get('title') and result.get('snippet'):
                sentiment = self.analyze_sentiment(result['snippet'])
                relevance = self.calculate_relevance_score(query, result['title'], result['snippet'])
                
                research_results.append(ResearchResult(
                    title=result['title'],
                    url=result.get('url', ''),
                    snippet=result['snippet'],
                    timestamp=datetime.now(),
                    source=result.get('source', 'Unknown'),
                    sentiment=sentiment,
                    relevance_score=relevance
                ))
        
        # Filter by sentiment if specified
        if filters.get('sentiment_filter'):
            sentiment_map = {
                'positive': lambda s: s > 0.1,
                'negative': lambda s: s < -0.1,
                'neutral': lambda s: -0.1 <= s <= 0.1
            }
            if filters['sentiment_filter'] in sentiment_map:
                filter_func = sentiment_map[filters['sentiment_filter']]
                research_results = [r for r in research_results if filter_func(r.sentiment)]
        
        # Filter by date range if specified
        if filters.get('date_range') and filters['date_range'] != 'All Time':
            now = datetime.now()
            time_delta = None
            if filters['date_range'] == 'Last 24 Hours':
                time_delta = timedelta(hours=24)
            elif filters['date_range'] == 'Last Week':
                time_delta = timedelta(days=7)
            elif filters['date_range'] == 'Last Month':
                time_delta = timedelta(days=30)
            
            if time_delta:
                # NOTE: This filter relies on the timestamp assigned when the result is processed.
                # For sources that don't provide a publication date, this reflects when the search was run.
                research_results = [r for r in research_results if now - r.timestamp <= time_delta]

        # Sort by relevance score
        research_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Cache results
        self.cache[cache_key] = research_results[:max_results]
        
        return research_results[:max_results]

async def gemini_flash_response(prompt: str, api_key: str) -> str:
    """Get a response from Gemini 2.5 Flash for a given prompt."""
    import requests
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    # Add strict comprehensive instruction to the prompt
    comprehensive_instruction = (
        "INSTRUCTION: You must generate the most comprehensive, exhaustive, and detailed answer possible like the deep research. "
        "Use the full token allocation (up to 8192 tokens). Do not be brief. "
        "Your response must be long-form, deeply analytical, and cover every aspect of the topic. "
        "Be meticulous, thorough, and leave no relevant detail unexplored.\n\n"
    )
    full_prompt = comprehensive_instruction + prompt
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 8192,  # This is the maximum allowed
        }
    }
    params = {"key": api_key}
    try:
        # Use asyncio to run the synchronous requests.post in a non-blocking way
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: requests.post(endpoint, headers=headers, params=params, json=payload, timeout=90))
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
    except requests.exceptions.RequestException as e:
        return f"Gemini API request failed: {e}"
    except Exception as e:
        return f"An unexpected error occurred with the Gemini API: {e}"
    
    if "candidates" in data and data["candidates"]:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    return f"Gemini API returned an empty response. Response: {data}"

async def main():
    st.set_page_config(
        page_title="🔬 Mini Research Assistant Pro",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .result-card {
        border: 1px solid #23272f;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: #23272f;
        color: #f1f1f1;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .sentiment-positive { color: #28fa7a; }
    .sentiment-negative { color: #ff5c8a; }
    .sentiment-neutral { color: #b0b3b8; }
    .relevance-high { background-color: #22332a !important; }
    .relevance-medium { background-color: #2a2e22 !important; }
    .relevance-low { background-color: #332222 !important; }
    a { color: #7ecbff; }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize research assistant
    if 'research_assistant' not in st.session_state:
        st.session_state.research_assistant = AdvancedResearchAssistant()
    
    ra = st.session_state.research_assistant
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Mini Research Assistant Pro</h1>', unsafe_allow_html=True)
    st.markdown("**Professional-grade research tool with sentiment analysis, multi-source search, and advanced filtering**")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Research Controls")
        # --- AI Assistant API Key ---
        st.subheader("✨ AI Assistant")
        if 'gemini_api_key' not in st.session_state:
            st.session_state.gemini_api_key = ''
        st.session_state.gemini_api_key = st.text_input("Google Gemini API Key", type="password", value=st.session_state.gemini_api_key)
        
        # Search Configuration
        st.subheader("Search Settings")
        max_results = st.slider("Max Results", 5, 50, 20)
        
        search_sources = st.multiselect(
            "Search Sources",
            ["Web Search", "Wikipedia", "Stock Data"],
            default=["Web Search", "Wikipedia"]
        )
        
        # Advanced Filters
        st.subheader("Advanced Filters")
        sentiment_filter = st.selectbox(
            "Sentiment Filter",
            ["All", "Positive", "Negative", "Neutral"]
        )
        
        date_range = st.selectbox(
            "Recency Filter",
            ["All Time", "Last 24 Hours", "Last Week", "Last Month"]
        )
        
        # Research Templates
        st.subheader("📋 Research Templates")
        templates = {
            "Market Research": "latest trends market analysis consumer behavior",
            "Technology Research": "emerging technology innovation developments",
            "Academic Research": "peer reviewed studies research papers findings",
            "News & Current Events": "breaking news latest updates current events",
            "Company Analysis": "financial performance business strategy market position",
            "Health & Medical": "medical research health studies clinical trials"
        }
        
        selected_template = st.selectbox("Choose Template", ["Custom"] + list(templates.keys()))
        
        # --- Productivity Tools ---
        st.subheader("🛠️ Productivity Tools")
        
        with st.expander("Standalone Tools", expanded=False):
            # --- Word & Character Counter ---
            st.markdown("##### Word & Character Counter")
            counter_text = st.text_area("Paste text to count words, characters, and sentences:", height=100, key="counter_input")
            if st.button("🔢 Count", key="count_button"):
                if counter_text.strip():
                    num_words = len(counter_text.split())
                    num_chars = len(counter_text)
                    num_chars_no_space = len(counter_text.replace(" ", ""))
                    num_sentences = len(re.findall(r'[.!?]+', counter_text))
                    st.success(
                        f"**Words:** {num_words}\n\n"
                        f"**Characters (with spaces):** {num_chars}\n\n"
                        f"**Characters (no spaces):** {num_chars_no_space}\n\n"
                        f"**Sentences:** {num_sentences}"
                    )
                else:
                    st.info("Please paste some text to count.")

            st.markdown("---")
            # --- Text Case Converter ---
            st.markdown("##### Text Case Converter")
            case_text = st.text_area("Paste text to convert case:", height=100, key="case_input")
            case_option = st.selectbox("Choose case", ["UPPERCASE", "lowercase", "Title Case", "Sentence case"], key="case_option")
            if st.button("🔤 Convert Case", key="convert_case_button"):
                if case_text.strip():
                    if case_option == "UPPERCASE":
                        converted = case_text.upper()
                    elif case_option == "lowercase":
                        converted = case_text.lower()
                    elif case_option == "Title Case":
                        converted = case_text.title()
                    elif case_option == "Sentence case":
                        # Capitalize first letter of each sentence
                        sentences = re.split('([.!?] *)', case_text)
                        converted = ''.join([s.capitalize() for s in sentences])
                    st.success("**Converted Text:**")
                    st.code(converted, language="text")
                else:
                    st.info("Please paste some text to convert.")

            st.markdown("---")
            # --- Analyze Sentiment ---
            st.markdown("##### Analyze Sentiment")
            sentiment_text = st.text_area("Paste text to analyze its sentiment:", height=100, key="sentiment_input")
            if st.button("😊 Analyze Sentiment", key="analyze_sentiment_button"):
                if sentiment_text.strip():
                    sentiment = ra.analyze_sentiment(sentiment_text)
                    sentiment_label = "Positive" if sentiment > 0.1 else "Negative" if sentiment < -0.1 else "Neutral"
                    st.success(f"**Sentiment Score:** `{sentiment:.3f}`\n\n**Overall Sentiment:** {sentiment_label}")
                else:
                    st.warning("Please paste some text to analyze.")

            st.markdown("---")
            # --- Extract Keywords ---
            st.markdown("##### Extract Keywords")
            keyword_text = st.text_area("Paste text to extract keywords from:", height=100, key="keyword_input")
            if st.button("🏷️ Extract Keywords", key="extract_keywords_button"):
                if keyword_text.strip():
                    keywords = ra.extract_keywords(keyword_text, top_n=15)
                    if keywords:
                        st.success("**Extracted Keywords:**")
                        st.write(", ".join(f"`{k}`" for k in keywords))
                    else:
                        st.info("No significant keywords were found.")
                else:
                    st.warning("Please paste some text to extract keywords from.")

        with st.expander("AI-Powered Tools (Requires API Key)", expanded=False):
            if not st.session_state.gemini_api_key:
                st.info("Enter your Gemini API key above to use these tools.")
            else:
                # --- Content Summarizer (existing) ---
                st.markdown("##### Content Summarizer")
                summarizer_text = st.text_area("Paste text to summarize:", height=100, key="summarizer_input")
                if st.button("✍️ Summarize", key="summarize_button"):
                    if summarizer_text.strip():
                        prompt = f"Please summarize the following text concisely:\n\n---\n\n{summarizer_text}"
                        with st.spinner("AI is summarizing..."):
                            summary = await gemini_flash_response(prompt, st.session_state.gemini_api_key)
                        st.success("**Summary:**")
                        st.markdown(summary)
                    else:
                        st.warning("Please paste some text to summarize.")

                st.markdown("---")
                # --- Technical Text Simplifier ---
                st.markdown("##### Technical Text Simplifier")
                simplify_text = st.text_area("Paste technical text to simplify:", height=100, key="simplify_input")
                if st.button("🧑‍🎓 Simplify Text", key="simplify_button"):
                    if simplify_text.strip():
                        prompt = (
                            "Simplify the following technical text so that a non-expert can easily understand it. "
                            "Use clear, plain language and explain any jargon:\n\n"
                            f"{simplify_text}"
                        )
                        with st.spinner("AI is simplifying..."):
                            simplified = await gemini_flash_response(prompt, st.session_state.gemini_api_key)
                        st.success("**Simplified Text:**")
                        st.markdown(simplified)
                    else:
                        st.warning("Please paste technical text to simplify.")

                st.markdown("---")
                # --- Email Drafter ---
                st.markdown("##### Email Drafter")
                email_points = st.text_area("Enter key points for your email (one per line):", height=100, key="email_points_input")
                email_tone = st.selectbox("Email Tone", ["Professional", "Friendly", "Concise", "Persuasive"], key="email_tone")
                if st.button("📧 Draft Email", key="draft_email_button"):
                    if email_points.strip():
                        prompt = (
                            f"Write a {email_tone.lower()} email using these key points:\n"
                            f"{email_points}\n\n"
                            "Format as a complete email with greeting and closing."
                        )
                        with st.spinner("AI is drafting your email..."):
                            email = await gemini_flash_response(prompt, st.session_state.gemini_api_key)
                        st.success("**Drafted Email:**")
                        st.markdown(email)
                    else:
                        st.warning("Please enter key points for your email.")

                st.markdown("---")
                # --- Social Media Post Generator ---
                st.markdown("##### Social Media Post Generator")
                sm_topic = st.text_area("Describe your topic or announcement:", height=80, key="sm_topic_input")
                sm_platform = st.selectbox("Platform", ["Twitter/X", "LinkedIn", "Facebook", "Instagram"], key="sm_platform")
                if st.button("📱 Generate Post", key="generate_post_button"):
                    if sm_topic.strip():
                        prompt = (
                            f"Write an engaging {sm_platform} post about:\n"
                            f"{sm_topic}\n\n"
                            "Make it catchy and suitable for the platform."
                        )
                        with st.spinner("AI is generating your post..."):
                            post = await gemini_flash_response(prompt, st.session_state.gemini_api_key)
                        st.success("**Generated Post:**")
                        st.markdown(post)
                    else:
                        st.warning("Please describe your topic.")

                st.markdown("---")
                # --- Code Explainer ---
                st.markdown("##### Code Explainer")
                code_snippet = st.text_area("Paste code to explain:", height=100, key="code_explain_input")
                code_lang = st.text_input("Programming Language (optional)", key="code_lang_input")
                if st.button("💡 Explain Code", key="explain_code_button"):
                    if code_snippet.strip():
                        prompt = (
                            f"Explain the following code in plain English. "
                            f"{'The language is ' + code_lang + '.' if code_lang else ''}\n\n"
                            f"{code_snippet}"
                        )
                        with st.spinner("AI is explaining the code..."):
                            explanation = await gemini_flash_response(prompt, st.session_state.gemini_api_key)
                        st.success("**Code Explanation:**")
                        st.markdown(explanation)
                    else:
                        st.warning("Please paste code to explain.")

                st.markdown("---")
                # --- Translation Tool ---
                st.markdown("##### Translation Tool")
                translate_text = st.text_area("Paste text to translate:", height=80, key="translate_input")
                translate_lang = st.selectbox(
                    "Translate to",
                    ["Spanish", "French", "German", "Chinese", "Hindi", "Arabic", "Russian", "Portuguese", "Japanese", "Korean"],
                    key="translate_lang"
                )
                if st.button("🌐 Translate", key="translate_button"):
                    if translate_text.strip():
                        prompt = (
                            f"Translate the following text to {translate_lang}:\n\n"
                            f"{translate_text}"
                        )
                        with st.spinner("AI is translating..."):
                            translation = await gemini_flash_response(prompt, st.session_state.gemini_api_key)
                        st.success(f"**Translation ({translate_lang}):**")
                        st.markdown(translation)
                    else:
                        st.warning("Please paste text to translate.")

                st.markdown("---")
                # --- Idea Generator ---
                st.markdown("##### Idea Generator")
                idea_topic = st.text_area("Describe your topic or challenge:", height=80, key="idea_input")
                if st.button("💡 Generate Ideas", key="generate_ideas_button"):
                    if idea_topic.strip():
                        prompt = (
                            f"Brainstorm a list of creative, practical ideas for the following topic or challenge:\n\n"
                            f"{idea_topic}\n\n"
                            "Provide at least 5 ideas."
                        )
                        with st.spinner("AI is generating ideas..."):
                            ideas = await gemini_flash_response(prompt, st.session_state.gemini_api_key)
                        st.success("**Generated Ideas:**")
                        st.markdown(ideas)
                    else:
                        st.warning("Please describe your topic or challenge.")

                st.markdown("---")
                # --- Proofreader & Grammar Checker ---
                st.markdown("##### Proofreader & Grammar Checker")
                proof_text = st.text_area("Paste text to proofread:", height=100, key="proof_input")
                if st.button("📝 Proofread", key="proofread_button"):
                    if proof_text.strip():
                        prompt = (
                            "Proofread the following text for spelling, grammar, and punctuation errors. "
                            "Correct any mistakes and provide the improved version:\n\n"
                            f"{proof_text}"
                        )
                        with st.spinner("AI is proofreading..."):
                            proofed = await gemini_flash_response(prompt, st.session_state.gemini_api_key)
                        st.success("**Proofread Text:**")
                        st.markdown(proofed)
                    else:
                        st.warning("Please paste text to proofread.")

                st.markdown("---")
                # --- Pros and Cons Lister ---
                st.markdown("##### Pros and Cons Lister")
                proscons_topic = st.text_area("Enter a topic or decision to analyze:", height=80, key="proscons_input")
                if st.button("⚖️ List Pros & Cons", key="proscons_button"):
                    if proscons_topic.strip():
                        prompt = (
                            f"List the main pros and cons of the following topic or decision. "
                            f"Present them in a clear, balanced format:\n\n"
                            f"{proscons_topic}"
                        )
                        with st.spinner("AI is analyzing..."):
                            proscons = await gemini_flash_response(prompt, st.session_state.gemini_api_key)
                        st.success("**Pros & Cons:**")
                        st.markdown(proscons)
                    else:
                        st.warning("Please enter a topic or decision.")

        if st.button("📊 Analytics Dashboard"):
            st.session_state.show_analytics = True
    
    # --- Main Tabs: AI Assistant and Research Tool ---
    tab_ai, tab_research = st.tabs(["✨ AI Assistant", "🔬 Research Tool"])
    with tab_ai:
        st.markdown("## ✨ Gemini 2.5 Flash AI Assistant")
        st.markdown("Ask anything! This space is powered by Gemini 2.5 Flash and is independent of the research tool.")
        if not st.session_state.gemini_api_key:
            st.info("Please enter your Google Gemini API key in the sidebar to use the AI Assistant.")
        else:
            ai_prompt = st.text_area("Enter your prompt for Gemini 2.5 Flash", "", height=120)
            # Use session state to persist AI response
            if "ai_response" not in st.session_state:
                st.session_state["ai_response"] = ""
            if st.button("Generate AI Response", key="ai_generate"):
                if ai_prompt.strip():
                    with st.spinner("Gemini is thinking..."):
                        ai_response = await gemini_flash_response(ai_prompt, st.session_state.gemini_api_key)
                    st.session_state["ai_response"] = ai_response
                else:
                    st.warning("Please enter a prompt for Gemini.")
            # Display the response if available
            ai_response = st.session_state.get("ai_response", "")
            if ai_response and isinstance(ai_response, str) and ai_response.strip():
                st.markdown("**Gemini Response:**")
                st.write(ai_response)
                # --- Export options for AI response ---
                md_bytes = ai_response.encode("utf-8")
                st.download_button(
                    label="⬇️ Export as Markdown",
                    data=md_bytes,
                    file_name=f"gemini_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
                try:
                    from fpdf import FPDF
                    import markdown
                    from bs4 import BeautifulSoup

                    # Convert markdown to HTML
                    html = markdown.markdown(ai_response)
                    soup = BeautifulSoup(html, "html.parser")

                    class PDF(FPDF):
                        def header(self):
                            pass
                        def footer(self):
                            pass

                    pdf = PDF()
                    pdf.add_page()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.set_font("Arial", size=12)

                    def render_html_to_pdf(soup, pdf):
                        def sanitize(text):
                            """Encode and decode to replace unsupported chars for FPDF."""
                            return text.encode('latin-1', 'replace').decode('latin-1')

                        for elem in soup.children:
                            if elem.name == "h1":
                                pdf.set_font("Arial", "B", 20)
                                pdf.set_text_color(44, 62, 80)
                                pdf.cell(0, 12, sanitize(elem.get_text()), ln=1)
                                pdf.set_font("Arial", size=12)
                                pdf.set_text_color(0, 0, 0)
                            elif elem.name == "h2":
                                pdf.set_font("Arial", "B", 16)
                                pdf.set_text_color(52, 152, 219)
                                pdf.cell(0, 10, sanitize(elem.get_text()), ln=1)
                                pdf.set_font("Arial", size=12)
                                pdf.set_text_color(0, 0, 0)
                            elif elem.name == "h3":
                                pdf.set_font("Arial", "B", 14)
                                pdf.set_text_color(39, 174, 96)
                                pdf.cell(0, 9, sanitize(elem.get_text()), ln=1)
                                pdf.set_font("Arial", size=12)
                                pdf.set_text_color(0, 0, 0)
                            elif elem.name == "h4":
                                pdf.set_font("Arial", "B", 12)
                                pdf.set_text_color(142, 68, 173)
                                pdf.cell(0, 8, sanitize(elem.get_text()), ln=1)
                                pdf.set_font("Arial", size=12)
                                pdf.set_text_color(0, 0, 0)
                            elif elem.name == "ul":
                                for li in elem.find_all("li", recursive=False):
                                    pdf.cell(5)
                                    # Use a plain ASCII dash instead of Unicode bullet
                                    pdf.multi_cell(0, 8, sanitize("- " + li.get_text()))
                            elif elem.name == "ol":
                                for idx, li in enumerate(elem.find_all("li", recursive=False), 1):
                                    pdf.cell(5)
                                    pdf.multi_cell(0, 8, sanitize(f"{idx}. {li.get_text()}"))
                            elif elem.name == "strong" or elem.name == "b":
                                pdf.set_font("Arial", "B", 12)
                                pdf.multi_cell(0, 8, sanitize(elem.get_text()))
                                pdf.set_font("Arial", size=12)
                            elif elem.name == "em" or elem.name == "i":
                                pdf.set_font("Arial", "I", 12)
                                pdf.multi_cell(0, 8, sanitize(elem.get_text()))
                                pdf.set_font("Arial", size=12)
                            elif elem.name == "p":
                                pdf.multi_cell(0, 8, sanitize(elem.get_text()))
                            elif elem.name is None:
                                # Plain text node
                                text = elem.string
                                if text and text.strip():
                                    pdf.multi_cell(0, 8, sanitize(text))
                            else:
                                # Fallback for other tags
                                pdf.multi_cell(0, 8, sanitize(elem.get_text()))

                    render_html_to_pdf(soup, pdf)
                    pdf_output = pdf.output(dest='S').encode('latin1', errors='replace')
                    if pdf_output:
                        st.download_button(
                            label="⬇️ Export as PDF",
                            data=pdf_output,
                            file_name=f"gemini_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                except ImportError:
                    st.info("Install `fpdf`, `markdown`, and `beautifulsoup4` packages to enable styled PDF export: `pip install fpdf markdown beautifulsoup4`")
                # Clear the response after displaying
                if st.button("🔄 Regenerate Response", key="ai_regenerate"):
                    st.session_state["ai_response"] = ""
    
    with tab_research:
        st.subheader("🔬 Research Tool")
        
        # Only allow research after clicking "Start Research"
        col1, col2 = st.columns([3, 1])
        with col1:
            if 'selected_template' in locals() and selected_template != "Custom":
                query = st.text_area(
                    "🔍 Research Query",
                    value=templates[selected_template],
                    height=100,
                    help="Enter your research query. Use keywords and specific terms for best results."
                )
            else:
                query = st.text_area(
                    "🔍 Research Query",
                    placeholder="Enter your research question or topic...",
                    height=100,
                    help="Enter your research query. Use keywords and specific terms for best results."
                )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_button = st.button("🚀 Start Research", type="primary", use_container_width=True)
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.session_state.research_assistant.cache.clear()
                st.success("Cache cleared!")
        
        # Stock ticker input for financial research
        if "Stock Data" in search_sources:
            stock_symbol = st.text_input("📈 Stock Symbol (optional)", placeholder="AAPL, GOOGL, TSLA...")
        
        # --- Only run research if Start Research is clicked ---
        if search_button and query:
            filters = {
                'include_web': "Web Search" in search_sources,
                'include_wikipedia': "Wikipedia" in search_sources,
                'sentiment_filter': sentiment_filter.lower() if sentiment_filter != "All" else None,
                'date_range': date_range
            }
            with st.spinner("🔍 Conducting comprehensive research..."):
                results = await ra.comprehensive_search(query, filters, max_results)
                st.session_state.last_results = results
                st.session_state.last_query = query
                
                # Display metrics
                if results:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f'<div class="metric-card"><h3>{len(results)}</h3><p>Results Found</p></div>', unsafe_allow_html=True)
                    
                    with col2:
                        avg_sentiment = sum(r.sentiment for r in results) / len(results)
                        sentiment_emoji = "😊" if avg_sentiment > 0.1 else "😐" if avg_sentiment > -0.1 else "😔"
                        st.markdown(f'<div class="metric-card"><h3>{sentiment_emoji}</h3><p>Avg Sentiment: {avg_sentiment:.2f}</p></div>', unsafe_allow_html=True)
                    
                    with col3:
                        avg_relevance = sum(r.relevance_score for r in results) / len(results)
                        st.markdown(f'<div class="metric-card"><h3>{avg_relevance:.1%}</h3><p>Avg Relevance</p></div>', unsafe_allow_html=True)
                    
                    with col4:
                        sources = set(r.source for r in results)
                        st.markdown(f'<div class="metric-card"><h3>{len(sources)}</h3><p>Sources Used</p></div>', unsafe_allow_html=True)
        
        # Display stock data if requested
        if search_button and "Stock Data" in search_sources and 'stock_symbol' in locals() and stock_symbol:
            with st.spinner(f"📈 Fetching stock data for {stock_symbol}..."):
                stock_data = await ra.get_stock_data(stock_symbol.upper())
                
                if stock_data:
                    st.subheader(f"📊 Stock Analysis: {stock_data.get('name', stock_symbol)}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${stock_data.get('price', 0):.2f}")
                    with col2:
                        change = stock_data.get('change', 0)
                        st.metric("Change %", f"{change:.2f}%", delta=f"{change:.2f}%")
                    with col3:
                        volume = stock_data.get('volume', 0)
                        st.metric("Volume", f"{volume:,}")
        
        # Display results
        if 'last_results' in st.session_state and st.session_state.last_results:
            results = st.session_state.last_results
            
            # Tabs for different views
            tab1, tab2, tab3, tab4, tab_ai_analysis = st.tabs(["📋 Results", "📊 Analytics", "🏷️ Keywords", "📥 Export", "✨ AI Analysis"])
            
            with tab1:
                st.subheader(f"🔍 Research Results for: '{st.session_state.last_query}'")
                
                # Sort options
                sort_by = st.selectbox("Sort by", ["Relevance", "Sentiment", "Source", "Title"])
                
                if sort_by == "Relevance":
                    sorted_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)
                elif sort_by == "Sentiment":
                    sorted_results = sorted(results, key=lambda x: x.sentiment, reverse=True)
                elif sort_by == "Source":
                    sorted_results = sorted(results, key=lambda x: x.source)
                else:
                    sorted_results = sorted(results, key=lambda x: x.title)
                
                # Display results
                for i, result in enumerate(sorted_results, 1):
                    # Determine relevance class
                    if result.relevance_score > 0.7:
                        relevance_class = "relevance-high"
                    elif result.relevance_score > 0.3:
                        relevance_class = "relevance-medium"
                    else:
                        relevance_class = "relevance-low"
                    
                    # Sentiment styling
                    if result.sentiment > 0.1:
                        sentiment_class = "sentiment-positive"
                        sentiment_emoji = "😊"
                    elif result.sentiment < -0.1:
                        sentiment_class = "sentiment-negative"
                        sentiment_emoji = "😔"
                    else:
                        sentiment_class = "sentiment-neutral"
                        sentiment_emoji = "😐"
                    
                    st.markdown(f"""
                    <div class="result-card {relevance_class}">
                        <h4>{i}. {result.title}</h4>
                        <p><strong>Source:</strong> {result.source} | 
                        <strong>Relevance:</strong> {result.relevance_score:.1%} | 
                        <strong>Sentiment:</strong> <span class="{sentiment_class}">{sentiment_emoji} {result.sentiment:.2f}</span></p>
                        <p>{result.snippet}</p>
                        {f'<p><a href="{result.url}" target="_blank">🔗 Read More</a></p>' if result.url else ''}
                    </div>
                    """, unsafe_allow_html=True)
            
            with tab2:
                st.subheader("📊 Research Analytics")
                
                # Sentiment distribution
                sentiments = [r.sentiment for r in results]
                fig_sentiment = px.histogram(
                    x=sentiments,
                    nbins=20,
                    title="Sentiment Distribution",
                    labels={'x': 'Sentiment Score', 'y': 'Count'}
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
                
                # Source distribution
                source_counts = pd.DataFrame([r.source for r in results], columns=['Source']).value_counts().reset_index()
                source_counts.columns = ['Source', 'Count']
                
                fig_sources = px.pie(
                    source_counts,
                    values='Count',
                    names='Source',
                    title="Results by Source"
                )
                st.plotly_chart(fig_sources, use_container_width=True)
                
                # Relevance vs Sentiment scatter
                df_scatter = pd.DataFrame({
                    'Relevance': [r.relevance_score for r in results],
                    'Sentiment': [r.sentiment for r in results],
                    'Source': [r.source for r in results],
                    'Title': [r.title[:50] + '...' if len(r.title) > 50 else r.title for r in results]
                })
                
                fig_scatter = px.scatter(
                    df_scatter,
                    x='Relevance',
                    y='Sentiment',
                    color='Source',
                    hover_data=['Title'],
                    title="Relevance vs Sentiment Analysis"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with tab3:
                st.subheader("🏷️ Keyword Analysis")
                
                # Extract keywords from all snippets
                all_text = " ".join([r.snippet for r in results])
                keywords = ra.extract_keywords(all_text, 20)
                
                if keywords:
                    # Create word frequency chart
                    word_freq = Counter(all_text.lower().split())
                    common_words = [(word, count) for word, count in word_freq.most_common(15) if word in keywords]
                    
                    if common_words:
                        df_words = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
                        fig_words = px.bar(
                            df_words,
                            x='Frequency',
                            y='Word',
                            orientation='h',
                            title="Most Frequent Keywords"
                        )
                        st.plotly_chart(fig_words, use_container_width=True)
                    
                    # Display keyword cloud
                    st.write("**Top Keywords:**")
                    cols = st.columns(5)
                    for i, keyword in enumerate(keywords[:15]):
                        with cols[i % 5]:
                            st.button(keyword, key=f"keyword_{i}")
            
            with tab4:
                st.subheader("📥 Export Results")
                
                # Prepare export data
                export_data = []
                for result in results:
                    export_data.append({
                        'Title': result.title,
                        'URL': result.url,
                        'Snippet': result.snippet,
                        'Source': result.source,
                        'Sentiment': result.sentiment,
                        'Relevance': result.relevance_score,
                        'Timestamp': result.timestamp.isoformat()
                    })
                
                df_export = pd.DataFrame(export_data)
                
                # CSV download
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="📁 Download CSV",
                    data=csv,
                    file_name=f"research_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # JSON download
                json_data = json.dumps(export_data, indent=2, default=str)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_data,
                    file_name=f"research_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                # Display summary stats
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Results", len(results))
                    st.metric("Average Sentiment", f"{sum(r.sentiment for r in results) / len(results):.3f}")
                with col2:
                    st.metric("Average Relevance", f"{sum(r.relevance_score for r in results) / len(results):.3f}")
                    st.metric("Unique Sources", len(set(r.source for r in results)))
            
            with tab_ai_analysis:
                st.subheader("✨ AI-Powered Research Analysis")
                st.markdown("Use Gemini to generate a comprehensive summary and analysis of your research results.")

                if not st.session_state.gemini_api_key:
                    st.info("Please enter your Google Gemini API key in the sidebar to use this feature.")
                else:
                    if st.button("🤖 Generate Comprehensive Analysis", use_container_width=True, key="generate_ai_analysis"):
                        # 1. Format the results into a prompt
                        formatted_results = ""
                        for i, result in enumerate(results[:25], 1): # Limit to top 25 to avoid overly long prompts
                            formatted_results += f"Result {i}:\n"
                            formatted_results += f"Title: {result.title}\n"
                            formatted_results += f"Source: {result.source}\n"
                            formatted_results += f"Snippet: {result.snippet}\n"
                            formatted_results += f"Sentiment: {result.sentiment:.2f}\n"
                            formatted_results += f"Relevance: {result.relevance_score:.2%}\n---\n"
                        
                        query = st.session_state.get('last_query', 'the user query')

                        analysis_prompt = f"""**Role:** You are a Senior Research Analyst at a top-tier global consulting firm. Your work is known for its incredible depth, clarity, and actionable insights. You are tasked with creating a definitive, exhaustive analysis based on a curated set of research data.

**Objective:** Produce an extremely comprehensive and in-depth report analyzing the provided research findings. The report must be detailed, well-structured, and leave no stone unturned. The goal is to provide a complete and holistic understanding of the topic based *only* on the data provided.

**Original Research Query:** "{query}"

**Provided Research Data:**
---
{formatted_results}
---

**Mandatory Report Structure:**

You must generate a detailed report with the following sections. Be expansive and thorough in each one.

1.  **Executive Summary:** A concise yet powerful overview of the most critical findings. This should be a stand-alone summary that captures the essence of the entire report.

2.  **Deep Dive: Key Themes & Insights:**
    *   Identify 3-5 major recurring themes from the data.
    *   For each theme, provide a detailed explanation.
    *   Quote specific snippets or reference data points from the provided results to back up every insight.
    *   Analyze the implications of these themes. What do they mean in the broader context of the query?

3.  **Comprehensive Sentiment Analysis:**
    *   Provide an overall assessment of the sentiment (positive, negative, neutral, mixed).
    *   Go beyond a simple score. Discuss the nuances of the sentiment. Are there specific aspects that are viewed more positively or negatively?
    *   Analyze the sentiment distribution across different sources. Are some sources more biased than others?

4.  **Analysis of Contradictions, Inconsistencies, and Gaps:**
    *   Meticulously identify any conflicting information or contradictions between the different research results.
    *   Highlight what information is missing. What questions remain unanswered by this dataset? What are the clear gaps in the research provided?

5.  **Strategic Conclusion & Actionable Recommendations:**
    *   Summarize the most important conclusions drawn from the analysis.
    *   Based on the analysis, propose a set of clear, actionable next steps or areas for further, more targeted investigation. What should the reader do with this information?

**Final Instruction:** Your final output must be a long-form, detailed report. Do not be brief. Use the full token allocation if necessary to provide a truly comprehensive and valuable analysis. Your response should be written in professional, clear, and precise language.
"""
                        with st.spinner("🤖 The AI is analyzing your results... this may take a moment..."):
                            ai_analysis_response = await gemini_flash_response(analysis_prompt, st.session_state.gemini_api_key)
                        
                        st.markdown("### 🧠 AI Analysis Report")
                        st.markdown(ai_analysis_response)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h4>🔬 Mini Research Assistant Pro</h4>
        <p>Professional research tool with advanced analytics and multi-source intelligence</p>
        <p><strong>Features:</strong> Multi-source search • Sentiment analysis • Keyword extraction • Real-time stock data • Export capabilities</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    asyncio.run(main())
