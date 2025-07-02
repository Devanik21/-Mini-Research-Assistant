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
            wiki_results = self.search_wikipedia(query, 5)
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
        
        # Sort by relevance score
        research_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Cache results
        self.cache[cache_key] = research_results[:max_results]
        
        return research_results[:max_results]

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
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: #f9f9f9;
    }
    .sentiment-positive { color: #28a745; }
    .sentiment-negative { color: #dc3545; }
    .sentiment-neutral { color: #6c757d; }
    .relevance-high { background-color: #d4edda; }
    .relevance-medium { background-color: #fff3cd; }
    .relevance-low { background-color: #f8d7da; }
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
    
    # Main search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if selected_template != "Custom":
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
    
    # Search execution
    if search_button and query:
        filters = {
            'include_web': "Web Search" in search_sources,
            'include_wikipedia': "Wikipedia" in search_sources,
            'sentiment_filter': sentiment_filter.lower() if sentiment_filter != "All" else None,
            'date_range': date_range
        }
        
        with st.spinner("🔍 Conducting comprehensive research..."):
            results = ra.comprehensive_search(query, filters, max_results)
            
            # Store results in session state
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
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Results", "📊 Analytics", "🏷️ Keywords", "📥 Export"])
        
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
