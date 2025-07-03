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
import base64
from io import BytesIO
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time
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
    
    def search_duckduckgo(self, query: str, num_results: int = 10) -> List[Dict]:
        """Search using DuckDuckGo API"""
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
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
    
    def search_wikipedia(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search Wikipedia"""
        try:
            search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
            encoded_query = quote_plus(query)
            
            response = requests.get(f"{search_url}{encoded_query}", timeout=10)
            if response.status_code == 200:
                data = response.json()
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
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': num_results
            }
            
            response = requests.get(search_api, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
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
    
    def get_stock_data(self, symbol: str) -> Dict:
        """Get stock data using yfinance"""
        try:
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
        except Exception as e:
            st.error(f"Stock data error: {str(e)}")
            return {}
    
    def comprehensive_search(self, query: str, filters: Dict, max_results: int = 20) -> List[ResearchResult]:
        """Perform comprehensive search across multiple sources"""
        cache_key = self.get_cache_key(query, filters)
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        all_results = []
        
        # Search sources based on filters
        if filters.get('include_web', True):
            web_results = self.search_duckduckgo(query, max_results // 2)
            all_results.extend(web_results)
        
        if filters.get('include_wikipedia', True):
            # Allocate a portion of max_results to Wikipedia, ensuring at least 1.
            wiki_results = self.search_wikipedia(query, max(1, max_results // 4))
            all_results.extend(wiki_results)
        
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
            sentiment_type = filters['sentiment_filter']
            if sentiment_type == 'positive':
                research_results = [r for r in research_results if r.sentiment > 0.1]
            elif sentiment_type == 'negative':
                research_results = [r for r in research_results if r.sentiment < -0.1]
            elif sentiment_type == 'neutral':
                research_results = [r for r in research_results if -0.1 <= r.sentiment <= 0.1]
        
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

def gemini_flash_response(prompt: str, api_key: str) -> str:
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
        response = requests.post(endpoint, headers=headers, params=params, json=payload, timeout=90)
        if response.status_code == 200:
            data = response.json()
            if "candidates" in data and data["candidates"]:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            return f"Gemini API returned an empty response. Response: {data}"
        else:
            return f"Gemini API error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Gemini API request failed: {e}"

def main():
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
        gemini_api_key = st.text_input("Google Gemini API Key", type="password", key="gemini_api_key")
        # Do NOT assign to st.session_state["gemini_api_key"] here to avoid StreamlitAPIException
        # Use gemini_api_key directly everywhere below
        
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
        
        if st.button("📊 Analytics Dashboard"):
            st.session_state.show_analytics = True
    
    # --- Main Tabs: AI Assistant and Research Tool ---
    tab_ai, tab_research = st.tabs(["✨ AI Assistant", "🔬 Research Tool"])

    with tab_ai:
        st.markdown("## ✨ Gemini 2.5 Flash AI Assistant")
        st.markdown("Ask anything! This space is powered by Gemini 2.5 Flash and is independent of the research tool.")
        if not gemini_api_key:
            st.info("Please enter your Google Gemini API key in the sidebar to use the AI Assistant.")
        else:
            ai_prompt = st.text_area("Enter your prompt for Gemini 2.5 Flash", "", height=120)
            # Use session state to persist AI response
            if "ai_response" not in st.session_state:
                st.session_state["ai_response"] = ""
            if st.button("Generate AI Response", key="ai_generate"):
                if ai_prompt.strip():
                    with st.spinner("Gemini is thinking..."):
                        ai_response = gemini_flash_response(ai_prompt, gemini_api_key)
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
                        for elem in soup.children:
                            if elem.name == "h1":
                                pdf.set_font("Arial", "B", 20)
                                pdf.set_text_color(44, 62, 80)
                                pdf.cell(0, 12, elem.get_text(), ln=1)
                                pdf.set_font("Arial", size=12)
                                pdf.set_text_color(0, 0, 0)
                            elif elem.name == "h2":
                                pdf.set_font("Arial", "B", 16)
                                pdf.set_text_color(52, 152, 219)
                                pdf.cell(0, 10, elem.get_text(), ln=1)
                                pdf.set_font("Arial", size=12)
                                pdf.set_text_color(0, 0, 0)
                            elif elem.name == "h3":
                                pdf.set_font("Arial", "B", 14)
                                pdf.set_text_color(39, 174, 96)
                                pdf.cell(0, 9, elem.get_text(), ln=1)
                                pdf.set_font("Arial", size=12)
                                pdf.set_text_color(0, 0, 0)
                            elif elem.name == "h4":
                                pdf.set_font("Arial", "B", 12)
                                pdf.set_text_color(142, 68, 173)
                                pdf.cell(0, 8, elem.get_text(), ln=1)
                                pdf.set_font("Arial", size=12)
                                pdf.set_text_color(0, 0, 0)
                            elif elem.name == "ul":
                                for li in elem.find_all("li", recursive=False):
                                    pdf.cell(5)
                                    # Use a plain ASCII dash instead of Unicode bullet
                                    pdf.multi_cell(0, 8, "- " + li.get_text())
                            elif elem.name == "ol":
                                for idx, li in enumerate(elem.find_all("li", recursive=False), 1):
                                    pdf.cell(5)
                                    pdf.multi_cell(0, 8, f"{idx}. {li.get_text()}")
                            elif elem.name == "strong" or elem.name == "b":
                                pdf.set_font("Arial", "B", 12)
                                pdf.multi_cell(0, 8, elem.get_text())
                                pdf.set_font("Arial", size=12)
                            elif elem.name == "em" or elem.name == "i":
                                pdf.set_font("Arial", "I", 12)
                                pdf.multi_cell(0, 8, elem.get_text())
                                pdf.set_font("Arial", size=12)
                            elif elem.name == "p":
                                pdf.multi_cell(0, 8, elem.get_text())
                            elif elem.name is None:
                                # Plain text node
                                text = elem.string
                                if text and text.strip():
                                    pdf.multi_cell(0, 8, text)
                            else:
                                # Fallback for other tags
                                pdf.multi_cell(0, 8, elem.get_text())

                    render_html_to_pdf(soup, pdf)
                    pdf_output = pdf.output(dest='S').encode('latin1', errors='replace')
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
                results = ra.comprehensive_search(query, filters, max_results)
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
                stock_data = ra.get_stock_data(stock_symbol.upper())
                
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

                if not gemini_api_key:
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
                            ai_analysis_response = gemini_flash_response(analysis_prompt, gemini_api_key)
                        
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
    main()
