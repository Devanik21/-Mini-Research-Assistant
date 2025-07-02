import streamlit as st
import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
import re
from urllib.parse import quote_plus, urljoin, urlparse
import base64
from io import BytesIO, StringIO
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import time
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Union, Tuple, Any
import nltk
from collections import Counter, defaultdict
import yfinance as yf
import feedparser
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

# Advanced Research Data Models
@dataclass
class EntityInfo:
    name: str
    entity_type: str
    confidence: float
    context: str
    related_entities: List[str] = field(default_factory=list)

@dataclass
class SemanticCluster:
    cluster_id: int
    theme: str
    documents: List[int]
    keywords: List[str]
    coherence_score: float

@dataclass
class TrendAnalysis:
    topic: str
    trend_direction: str
    confidence: float
    time_series: List[Tuple[datetime, float]]
    forecast: List[Tuple[datetime, float]]

@dataclass
class CompetitorAnalysis:
    company: str
    market_share: float
    sentiment_score: float
    mention_frequency: int
    key_strengths: List[str]
    key_weaknesses: List[str]

@dataclass
class ResearchInsight:
    insight_type: str
    title: str
    description: str
    confidence: float
    supporting_evidence: List[str]
    implications: List[str]

@dataclass
class AdvancedResearchResult:
    title: str
    url: str
    snippet: str
    full_content: str
    timestamp: datetime
    source: str
    sentiment: float
    relevance_score: float
    credibility_score: float
    entities: List[EntityInfo]
    keywords: List[str]
    topics: List[str]
    reading_level: str
    word_count: int
    images: List[str]
    citations: List[str]
    social_metrics: Dict[str, int]
    publication_date: Optional[datetime]
    author: Optional[str]
    domain_authority: float
    backlink_count: int
    content_type: str
    language: str
    geographic_relevance: List[str]
    industry_tags: List[str]
    research_quality_score: float

class NeuralSearchEngine:
    """Advanced neural search engine with transformer models"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.embeddings_model = None
        self.vector_store = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.nlp = None
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        
    def initialize_models(self):
        """Initialize AI models for advanced analysis"""
        try:
            # Initialize spaCy model
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            st.warning("spaCy English model not found. Some features may be limited.")
        
        try:
            # Initialize sentence transformer for embeddings
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            st.warning(f"Could not load embeddings model: {e}")
    
    def extract_entities(self, text: str) -> List[EntityInfo]:
        """Extract named entities using spaCy"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append(EntityInfo(
                name=ent.text,
                entity_type=ent.label_,
                confidence=0.8,  # spaCy doesn't provide confidence scores directly
                context=text[max(0, ent.start_char-50):ent.end_char+50]
            ))
        
        return entities
    
    def analyze_semantic_similarity(self, query: str, documents: List[str]) -> List[float]:
        """Calculate semantic similarity using embeddings"""
        if not self.embeddings_model:
            return [0.0] * len(documents)
        
        try:
            # Create embeddings
            query_embedding = self.embeddings_model.embed_query(query)
            doc_embeddings = self.embeddings_model.embed_documents(documents)
            
            # Calculate cosine similarity
            similarities = []
            for doc_embedding in doc_embeddings:
                similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                similarities.append(similarity)
            
            return similarities
        except Exception as e:
            st.error(f"Similarity analysis error: {e}")
            return [0.0] * len(documents)
    
    def perform_topic_modeling(self, documents: List[str], n_topics: int = 5) -> List[SemanticCluster]:
        """Perform topic modeling using K-means clustering"""
        if len(documents) < n_topics:
            n_topics = max(1, len(documents))
        
        try:
            # Vectorize documents
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_topics, random_state=42)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Extract topics
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            clusters = []
            
            for i in range(n_topics):
                # Get top terms for this cluster
                center = kmeans.cluster_centers_[i]
                top_indices = center.argsort()[-10:][::-1]
                top_terms = [feature_names[idx] for idx in top_indices]
                
                # Get documents in this cluster
                doc_indices = [idx for idx, label in enumerate(cluster_labels) if label == i]
                
                clusters.append(SemanticCluster(
                    cluster_id=i,
                    theme=" ".join(top_terms[:3]),
                    documents=doc_indices,
                    keywords=top_terms,
                    coherence_score=0.7  # Simplified coherence score
                ))
            
            return clusters
        except Exception as e:
            st.error(f"Topic modeling error: {e}")
            return []

class EnterpriseDataCollector:
    """Advanced data collection from multiple sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.api_keys = {}
        self.rate_limits = defaultdict(list)
        
    def respect_rate_limit(self, source: str, limit_per_minute: int = 60):
        """Implement rate limiting for API calls"""
        now = time.time()
        self.rate_limits[source] = [
            timestamp for timestamp in self.rate_limits[source] 
            if now - timestamp < 60
        ]
        
        if len(self.rate_limits[source]) >= limit_per_minute:
            sleep_time = 60 - (now - self.rate_limits[source][0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.rate_limits[source].append(now)
    
    def search_arxiv(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search arXiv for academic papers"""
        try:
            base_url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = self.session.get(base_url, params=params, timeout=15)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                results = []
                
                for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                    title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
                    summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
                    link_elem = entry.find('{http://www.w3.org/2005/Atom}id')
                    published_elem = entry.find('{http://www.w3.org/2005/Atom}published')
                    
                    if title_elem is not None and summary_elem is not None:
                        results.append({
                            'title': title_elem.text.strip(),
                            'url': link_elem.text if link_elem is not None else '',
                            'snippet': summary_elem.text.strip()[:500] + '...',
                            'source': 'arXiv',
                            'publication_date': published_elem.text if published_elem is not None else None,
                            'content_type': 'academic_paper'
                        })
                
                return results
        except Exception as e:
            st.error(f"arXiv search error: {e}")
        return []
    
    def search_pubmed(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search PubMed for medical literature"""
        try:
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json',
                'sort': 'relevance'
            }
            
            response = self.session.get(base_url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                pmids = data.get('esearchresult', {}).get('idlist', [])
                
                if pmids:
                    # Get detailed information
                    detail_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                    detail_params = {
                        'db': 'pubmed',
                        'id': ','.join(pmids),
                        'retmode': 'json'
                    }
                    
                    detail_response = self.session.get(detail_url, params=detail_params, timeout=15)
                    if detail_response.status_code == 200:
                        detail_data = detail_response.json()
                        results = []
                        
                        for pmid in pmids:
                            if pmid in detail_data.get('result', {}):
                                article = detail_data['result'][pmid]
                                results.append({
                                    'title': article.get('title', ''),
                                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                                    'snippet': article.get('title', '')[:500] + '...',
                                    'source': 'PubMed',
                                    'publication_date': article.get('pubdate', ''),
                                    'content_type': 'medical_paper'
                                })
                        
                        return results
        except Exception as e:
            st.error(f"PubMed search error: {e}")
        return []
    
    def search_news_apis(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search multiple news APIs"""
        results = []
        
        # NewsAPI (requires API key)
        if 'newsapi' in self.api_keys:
            try:
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': query,
                    'pageSize': min(max_results, 20),
                    'sortBy': 'relevancy',
                    'language': 'en'
                }
                headers = {'X-API-Key': self.api_keys['newsapi']}
                
                response = self.session.get(url, params=params, headers=headers, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    for article in data.get('articles', []):
                        results.append({
                            'title': article.get('title', ''),
                            'url': article.get('url', ''),
                            'snippet': article.get('description', ''),
                            'source': f"News - {article.get('source', {}).get('name', 'Unknown')}",
                            'publication_date': article.get('publishedAt', ''),
                            'content_type': 'news_article'
                        })
            except Exception as e:
                st.error(f"NewsAPI error: {e}")
        
        # RSS feeds fallback
        rss_feeds = [
            'https://rss.cnn.com/rss/edition.rss',
            'https://feeds.reuters.com/reuters/topNews',
            'https://rss.bbc.co.uk/rss/newsonline_world_edition/front_page/rss.xml'
        ]
        
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:5]:  # Limit per feed
                    if query.lower() in entry.title.lower() or query.lower() in entry.get('summary', '').lower():
                        results.append({
                            'title': entry.title,
                            'url': entry.link,
                            'snippet': entry.get('summary', entry.title)[:500] + '...',
                            'source': f"RSS - {feed.feed.get('title', 'News')}",
                            'publication_date': entry.get('published', ''),
                            'content_type': 'news_article'
                        })
            except Exception as e:
                continue
        
        return results[:max_results]
    
    def search_social_media(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search social media mentions (simulated)"""
        # This would integrate with Twitter API, Reddit API, etc.
        # For demo purposes, we'll simulate social media results
        social_results = []
        
        platforms = ['Twitter', 'Reddit', 'LinkedIn', 'Facebook']
        sentiments = ['positive', 'negative', 'neutral']
        
        for i in range(max_results):
            platform = platforms[i % len(platforms)]
            sentiment = sentiments[i % len(sentiments)]
            
            social_results.append({
                'title': f"Social media discussion about {query}",
                'url': f"https://{platform.lower()}.com/post/{i}",
                'snippet': f"Users on {platform} are discussing {query} with {sentiment} sentiment...",
                'source': f"Social - {platform}",
                'publication_date': (datetime.now() - timedelta(hours=i)).isoformat(),
                'content_type': 'social_media',
                'social_metrics': {
                    'likes': np.random.randint(10, 1000),
                    'shares': np.random.randint(5, 500),
                    'comments': np.random.randint(1, 100)
                }
            })
        
        return social_results
    
    def analyze_competitor_intelligence(self, company: str, competitors: List[str]) -> List[CompetitorAnalysis]:
        """Analyze competitor intelligence"""
        analyses = []
        
        for competitor in competitors:
            # Simulate competitor analysis
            analysis = CompetitorAnalysis(
                company=competitor,
                market_share=np.random.uniform(0.05, 0.3),
                sentiment_score=np.random.uniform(-0.5, 0.5),
                mention_frequency=np.random.randint(50, 500),
                key_strengths=[
                    "Strong brand recognition",
                    "Innovative products",
                    "Market presence"
                ],
                key_weaknesses=[
                    "High pricing",
                    "Limited market reach",
                    "Customer service issues"
                ]
            )
            analyses.append(analysis)
        
        return analyses

class AdvancedAnalyticsEngine:
    """Advanced analytics and insights generation"""
    
    def __init__(self):
        self.neural_search = NeuralSearchEngine()
        self.data_collector = EnterpriseDataCollector()
        
    def generate_trend_analysis(self, results: List[AdvancedResearchResult], topic: str) -> TrendAnalysis:
        """Generate trend analysis from research results"""
        # Analyze publication dates and sentiment over time
        dated_results = [r for r in results if r.publication_date]
        if not dated_results:
            return TrendAnalysis(
                topic=topic,
                trend_direction="stable",
                confidence=0.5,
                time_series=[],
                forecast=[]
            )
        
        # Sort by date
        dated_results.sort(key=lambda x: x.publication_date)
        
        # Calculate sentiment trend
        time_series = []
        for result in dated_results:
            time_series.append((result.publication_date, result.sentiment))
        
        # Simple trend calculation
        if len(time_series) > 1:
            recent_sentiment = np.mean([s for _, s in time_series[-5:]])
            older_sentiment = np.mean([s for _, s in time_series[:5]])
            
            if recent_sentiment > older_sentiment + 0.1:
                trend_direction = "positive"
                confidence = 0.8
            elif recent_sentiment < older_sentiment - 0.1:
                trend_direction = "negative"
                confidence = 0.8
            else:
                trend_direction = "stable"
                confidence = 0.6
        else:
            trend_direction = "stable"
            confidence = 0.5
        
        # Generate simple forecast
        forecast = []
        if time_series:
            last_date = time_series[-1][0]
            last_sentiment = time_series[-1][1]
            
            for i in range(1, 6):  # 5 future points
                future_date = last_date + timedelta(days=30*i)
                # Simple linear extrapolation
                if trend_direction == "positive":
                    future_sentiment = min(1.0, last_sentiment + 0.1*i)
                elif trend_direction == "negative":
                    future_sentiment = max(-1.0, last_sentiment - 0.1*i)
                else:
                    future_sentiment = last_sentiment + np.random.uniform(-0.05, 0.05)
                
                forecast.append((future_date, future_sentiment))
        
        return TrendAnalysis(
            topic=topic,
            trend_direction=trend_direction,
            confidence=confidence,
            time_series=time_series,
            forecast=forecast
        )
    
    def generate_insights(self, results: List[AdvancedResearchResult], query: str) -> List[ResearchInsight]:
        """Generate actionable insights from research results"""
        insights = []
        
        if not results:
            return insights
        
        # Sentiment insight
        sentiments = [r.sentiment for r in results]
        avg_sentiment = np.mean(sentiments)
        
        if avg_sentiment > 0.2:
            insights.append(ResearchInsight(
                insight_type="sentiment",
                title="Predominantly Positive Sentiment",
                description=f"Research shows {avg_sentiment:.2f} average sentiment, indicating positive market perception.",
                confidence=0.8,
                supporting_evidence=[f"{len([s for s in sentiments if s > 0.1])} out of {len(sentiments)} sources show positive sentiment"],
                implications=["Market opportunity", "Positive brand perception", "Favorable conditions"]
            ))
        elif avg_sentiment < -0.2:
            insights.append(ResearchInsight(
                insight_type="sentiment",
                title="Negative Sentiment Alert",
                description=f"Research reveals {avg_sentiment:.2f} average sentiment, indicating potential concerns.",
                confidence=0.8,
                supporting_evidence=[f"{len([s for s in sentiments if s < -0.1])} out of {len(sentiments)} sources show negative sentiment"],
                implications=["Risk assessment needed", "Address concerns", "Monitor reputation"]
            ))
        
        # Source diversity insight
        sources = set(r.source for r in results)
        if len(sources) > 5:
            insights.append(ResearchInsight(
                insight_type="coverage",
                title="Comprehensive Source Coverage",
                description=f"Information gathered from {len(sources)} diverse sources ensures comprehensive coverage.",
                confidence=0.9,
                supporting_evidence=[f"Sources include: {', '.join(list(sources)[:5])}{'...' if len(sources) > 5 else ''}"],
                implications=["High information reliability", "Reduced bias", "Comprehensive analysis"]
            ))
        
        # Credibility insight
        credibility_scores = [r.credibility_score for r in results if r.credibility_score > 0]
        if credibility_scores:
            avg_credibility = np.mean(credibility_scores)
            if avg_credibility > 0.7:
                insights.append(ResearchInsight(
                    insight_type="credibility",
                    title="High Source Credibility",
                    description=f"Average source credibility of {avg_credibility:.2f} indicates reliable information.",
                    confidence=0.85,
                    supporting_evidence=[f"{len([c for c in credibility_scores if c > 0.7])} high-credibility sources"],
                    implications=["Trustworthy information", "Reliable for decision-making", "Low misinformation risk"]
                ))
        
        return insights
    
    def perform_competitive_analysis(self, query: str, results: List[AdvancedResearchResult]) -> Dict[str, Any]:
        """Perform competitive landscape analysis"""
        # Extract company/brand mentions
        companies = set()
        for result in results:
            for entity in result.entities:
                if entity.entity_type in ['ORG', 'COMPANY']:
                    companies.add(entity.name)
        
        # Analyze mention frequency and sentiment
        company_analysis = {}
        for company in list(companies)[:10]:  # Top 10 companies
            company_results = [r for r in results if company.lower() in r.snippet.lower()]
            if company_results:
                avg_sentiment = np.mean([r.sentiment for r in company_results])
                mention_count = len(company_results)
                
                company_analysis[company] = {
                    'mention_count': mention_count,
                    'average_sentiment': avg_sentiment,
                    'market_presence': mention_count / len(results),
                    'sentiment_category': 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral'
                }
        
        return {
            'total_companies_mentioned': len(companies),
            'top_companies': dict(sorted(company_analysis.items(), key=lambda x: x[1]['mention_count'], reverse=True)[:5]),
            'competitive_landscape': company_analysis
        }

class ResearchAssistantPro:
    """Main research assistant class with all advanced features"""
    
    def __init__(self):
        self.cache = {}
        self.database = self.initialize_database()
        self.neural_search = NeuralSearchEngine()
        self.data_collector = EnterpriseDataCollector()
        self.analytics_engine = AdvancedAnalyticsEngine()
        self.session_history = []
        self.research_projects = {}
        self.custom_models = {}
        
        # Initialize AI models
        self.neural_search.initialize_models()
    
    def initialize_database(self) -> sqlite3.Connection:
        """Initialize SQLite database for persistent storage"""
        conn = sqlite3.connect(':memory:')  # Use in-memory database
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE research_sessions (
                id INTEGER PRIMARY KEY,
                query TEXT,
                timestamp DATETIME,
                results_count INTEGER,
                avg_sentiment REAL,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE research_results (
                id INTEGER PRIMARY KEY,
                session_id INTEGER,
                title TEXT,
                url TEXT,
                snippet TEXT,
                sentiment REAL,
                relevance_score REAL,
                credibility_score REAL,
                source TEXT,
                FOREIGN KEY (session_id) REFERENCES research_sessions (id)
            )
        ''')
        
        conn.commit()
        return conn
    
    def get_cache_key(self, query: str, filters: Dict, search_depth: str) -> str:
        """Generate enhanced cache key"""
        content = f"{query}_{json.dumps(filters, sort_keys=True)}_{search_depth}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def analyze_advanced_sentiment(self, text: str) -> Dict[str, float]:
        """Advanced sentiment analysis using multiple methods"""
        # TextBlob sentiment
        blob_sentiment = TextBlob(text).sentiment.polarity
        
        # VADER sentiment
        vader_scores = self.neural_search.sentiment_analyzer.polarity_scores(text)
        vader_sentiment = vader_scores['compound']
        
        # Combine sentiments
        combined_sentiment = (blob_sentiment + vader_sentiment) / 2
        
        return {
            'textblob': blob_sentiment,
            'vader': vader_sentiment,
            'combined': combined_sentiment,
            'confidence': abs(combined_sentiment),
            'emotional_scores': {
                'positive': vader_scores['pos'],
                'negative': vader_scores['neg'],
                'neutral': vader_scores['neu']
            }
        }
    
    def calculate_credibility_score(self, result: Dict) -> float:
        """Calculate source credibility score"""
        score = 0.5  # Base score
        
        # Domain authority (simulated)
        domain = urlparse(result.get('url', '')).netloc
        authoritative_domains = [
            'arxiv.org', 'pubmed.ncbi.nlm.nih.gov', 'nature.com', 'science.org',
            'ieee.org', 'acm.org', 'springer.com', 'wikipedia.org',
            'gov', 'edu', 'reuters.com', 'bloomberg.com', 'wsj.com'
        ]
        
        for auth_domain in authoritative_domains:
            if auth_domain in domain:
                score += 0.3
                break
        
        # Content type bonus
        if result.get('content_type') in ['academic_paper', 'medical_paper']:
            score += 0.2
        
        # Publication date recency
        if result.get('publication_date'):
            try:
                pub_date = datetime.fromisoformat(result['publication_date'].replace('Z', '+00:00'))
                days_old = (datetime.now() - pub_date.replace(tzinfo=None)).days
                if days_old < 30:
                    score += 0.1
                elif days_old < 365:
                    score += 0.05
            except:
                pass
        
        return min(1.0, score)
    
    def extract_advanced_keywords(self, text: str, method: str = 'tfidf') -> List[Tuple[str, float]]:
        """Extract keywords using various methods"""
        if method == 'tfidf':
            try:
                vectorizer = TfidfVectorizer(max_features=20, stop_words='english', ngram_range=(1, 2))
                tfidf_matrix = vectorizer.fit_transform([text])
                feature_names = vectorizer.get_feature_names_out()
                scores = tfidf_matrix.toarray()[0]
                
                keyword_scores = list(zip(feature_names, scores))
                keyword_scores.sort(key=lambda x: x[1], reverse=True)
                return keyword_scores[:10]
            except:
                pass
        
        # Fallback to basic frequency
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_words = [w for w in words if w.isalnum() and w not in stop_words and len(w) > 3]
        word_freq = Counter(filtered_words)
        
        return [(word, count/len(filtered_words)) for word, count in word_freq.most_common(10)]
    
    def comprehensive_search(self, query: str, filters: Dict, search_depth: str = 'standard') -> List[AdvancedResearchResult]:
        """Perform comprehensive multi-source search"""
        cache_key = self.get_cache_key(query, filters, search_depth)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        all_results = []
        max_results = filters.get('max_results', 50)
        
        # Determine search sources based on depth
        if search_depth == 'deep':
            search_sources = ['web', 'wikipedia', 'arxiv', 'pubmed', 'news', 'social']
        elif search_depth == 'academic':
            search_sources = ['arxiv', 'pubmed', 'wikipedia']
        elif search_depth == 'news':
            search_sources = ['news', 'social', 'web']
        else:  # standard
            search_sources = ['web', 'wikipedia', 'news']
        
        # Execute searches based on selected sources
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_source = {}
            
            if 'web' in search_sources and filters.get('include_web', True):
                future_to_source[executor.submit(self.search_duckduckgo, query, max_results//4)] = 'web'
            
            if 'wikipedia' in search_sources and filters.get('include_wikipedia', True):
                future_to_source[executor.submit(self.search_wikipedia, query, 5)] = 'wikipedia'
            
            if 'arxiv' in search_sources and filters.get('include_academic', False):
                future_to_source[executor.submit(self.data_collector.search_arxiv, query, 10)] = 'arxiv'
            
            if 'pubmed' in search_sources and filters.get('include_medical', False):
                future_to_source[executor.submit(self.data_collector.search_pubmed, query, 10)] = 'pubmed'
            
            if 'news' in search_sources and filters.get('include_news', True):
                future_to_source[executor.submit(self.data_collector.search_news_apis, query, 15)] = 'news'
            
            if 'social' in search_sources and filters.get('include_social', False):
                future_to_source[executor.submit(self.data_collector.search_social_media, query, 10)] = 'social'
            
            # Collect results as they complete
            for future in as_completed(future_to_source):
                try:
                    results = future.result(timeout=30)
                    all_results.extend(results or [])
                except Exception as e:
                    st.warning(f"Search error in {future_to_source[future]}: {str(e)}")
        
        # Convert to AdvancedResearchResult objects
        research_results = []
        snippets = []
        
        for result in all_results:
            if not result or not result.get('title') or not result.get('snippet'):
                continue
            
            snippet = result['snippet']
            snippets.append(snippet)
            
            # Advanced sentiment analysis
            sentiment_analysis = self.analyze_advanced_sentiment(snippet)
            
            # Extract entities
            entities = self.neural_search.extract_entities(snippet)
            
            # Extract keywords
            keywords = self.extract_advanced_keywords(snippet)
            keyword_list = [kw[0] for kw in keywords[:10]]
            
            # Calculate credibility
            credibility = self.calculate_credibility_score(result)
            
            # Parse publication date
            pub_date = None
            if result.get('publication_date'):
                try:
                    pub_date = datetime.fromisoformat(result['publication_date'].replace('Z', '+00:00'))
                    pub_date = pub_date.replace(tzinfo=None)
                except:
                    pass
            
            # Calculate reading level (simplified)
            word_count = len(snippet.split())
            avg_word_length = sum(len(word) for word in snippet.split()) / max(word_count, 1)
            if avg_word_length > 6:
                reading_level = "Advanced"
            elif avg_word_length > 4:
                reading_level = "Intermediate"
            else:
                reading_level = "Basic"
            
            research_results.append(AdvancedResearchResult(
                title=result['title'],
                url=result.get('url', ''),
                snippet=snippet,
                full_content=result.get('full_content', snippet),
                timestamp=datetime.now(),
                source=result.get('source', 'Unknown'),
                sentiment=sentiment_analysis['combined'],
                relevance_score=0.0,  # Will be calculated below
                credibility_score=credibility,
                entities=entities,
                keywords=keyword_list,
                topics=[],
                reading_level=reading_level,
                word_count=word_count,
                images=result.get('images', []),
                citations=result.get('citations', []),
                social_metrics=result.get('social_metrics', {}),
                publication_date=pub_date,
                author=result.get('author'),
                domain_authority=0.8 if any(domain in result.get('url', '') for domain in ['gov', 'edu', 'org']) else 0.5,
                backlink_count=np.random.randint(10, 1000),  # Simulated
                content_type=result.get('content_type', 'web_page'),
                language='en',
                geographic_relevance=[],
                industry_tags=[],
                research_quality_score=credibility * 0.5 + sentiment_analysis['confidence'] * 0.3 + (1.0 if entities else 0.0) * 0.2
            ))
        
        # Calculate semantic similarity for relevance scores
        if research_results and snippets:
            similarity_scores = self.neural_search.analyze_semantic_similarity(query, snippets)
            for i, result in enumerate(research_results):
                if i < len(similarity_scores):
                    result.relevance_score = similarity_scores[i]
        
        # Apply filters
        filtered_results = self.apply_advanced_filters(research_results, filters)
        
        # Sort by combined score
        filtered_results.sort(key=lambda x: (x.relevance_score * 0.4 + x.credibility_score * 0.3 + 
                                           abs(x.sentiment) * 0.2 + x.research_quality_score * 0.1), reverse=True)
        
        # Cache results
        final_results = filtered_results[:max_results]
        self.cache[cache_key] = final_results
        
        # Store in database
        self.store_research_session(query, final_results)
        
        return final_results
    
    def apply_advanced_filters(self, results: List[AdvancedResearchResult], filters: Dict) -> List[AdvancedResearchResult]:
        """Apply advanced filtering options"""
        filtered = results
        
        # Sentiment filter
        if filters.get('sentiment_filter'):
            sentiment_type = filters['sentiment_filter'].lower()
            if sentiment_type == 'positive':
                filtered = [r for r in filtered if r.sentiment > 0.1]
            elif sentiment_type == 'negative':
                filtered = [r for r in filtered if r.sentiment < -0.1]
            elif sentiment_type == 'neutral':
                filtered = [r for r in filtered if -0.1 <= r.sentiment <= 0.1]
        
        # Date range filter
        if filters.get('date_range') and filters['date_range'] != 'All Time':
            cutoff_date = datetime.now()
            if filters['date_range'] == 'Last 24 Hours':
                cutoff_date -= timedelta(days=1)
            elif filters['date_range'] == 'Last Week':
                cutoff_date -= timedelta(weeks=1)
            elif filters['date_range'] == 'Last Month':
                cutoff_date -= timedelta(days=30)
            elif filters['date_range'] == 'Last Year':
                cutoff_date -= timedelta(days=365)
            
            filtered = [r for r in filtered if r.publication_date and r.publication_date >= cutoff_date]
        
        # Credibility threshold
        if filters.get('min_credibility', 0) > 0:
            filtered = [r for r in filtered if r.credibility_score >= filters['min_credibility']]
        
        # Content type filter
        if filters.get('content_types'):
            filtered = [r for r in filtered if r.content_type in filters['content_types']]
        
        # Reading level filter
        if filters.get('reading_levels'):
            filtered = [r for r in filtered if r.reading_level in filters['reading_levels']]
        
        return filtered
    
    def store_research_session(self, query: str, results: List[AdvancedResearchResult]):
        """Store research session in database"""
        try:
            cursor = self.database.cursor()
            
            # Insert session
            avg_sentiment = np.mean([r.sentiment for r in results]) if results else 0
            metadata = json.dumps({
                'total_sources': len(set(r.source for r in results)),
                'avg_credibility': np.mean([r.credibility_score for r in results]) if results else 0,
                'content_types': list(set(r.content_type for r in results))
            })
            
            cursor.execute('''
                INSERT INTO research_sessions (query, timestamp, results_count, avg_sentiment, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (query, datetime.now(), len(results), avg_sentiment, metadata))
            
            session_id = cursor.lastrowid
            
            # Insert results
            for result in results:
                cursor.execute('''
                    INSERT INTO research_results 
                    (session_id, title, url, snippet, sentiment, relevance_score, credibility_score, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, result.title, result.url, result.snippet, result.sentiment, 
                     result.relevance_score, result.credibility_score, result.source))
            
            self.database.commit()
        except Exception as e:
            st.error(f"Database error: {e}")
    
    def search_duckduckgo(self, query: str, num_results: int = 10) -> List[Dict]:
        """Enhanced DuckDuckGo search with error handling"""
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
                
                if data.get('Abstract'):
                    results.append({
                        'title': data.get('Heading', 'DuckDuckGo Summary'),
                        'url': data.get('AbstractURL', ''),
                        'snippet': data.get('Abstract', ''),
                        'source': 'DuckDuckGo Instant',
                        'content_type': 'summary'
                    })
                
                for topic in data.get('RelatedTopics', [])[:num_results]:
                    if isinstance(topic, dict) and topic.get('Text'):
                        results.append({
                            'title': topic.get('Text', '')[:100] + '...',
                            'url': topic.get('FirstURL', ''),
                            'snippet': topic.get('Text', ''),
                            'source': 'DuckDuckGo',
                            'content_type': 'web_page'
                        })
                
                return results[:num_results]
        except Exception as e:
            st.error(f"DuckDuckGo search error: {str(e)}")
        return []
    
    def search_wikipedia(self, query: str, num_results: int = 5) -> List[Dict]:
        """Enhanced Wikipedia search"""
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
                    snippet = re.sub(r'<[^>]+>', '', item.get('snippet', ''))
                    
                    results.append({
                        'title': title,
                        'url': f"https://en.wikipedia.org/wiki/{quote_plus(title)}",
                        'snippet': snippet,
                        'source': 'Wikipedia',
                        'content_type': 'encyclopedia',
                        'publication_date': datetime.now().isoformat()
                    })
                
                return results
        except Exception as e:
            st.error(f"Wikipedia search error: {str(e)}")
        return []

def create_advanced_visualizations(results: List[AdvancedResearchResult], query: str):
    """Create comprehensive visualization dashboard"""
    
    if not results:
        st.warning("No results to visualize")
        return
    
    # Create tabs for different visualization categories
    viz_tabs = st.tabs([
        "📊 Overview Dashboard", 
        "🎭 Sentiment Analysis", 
        "🔍 Source Intelligence", 
        "📈 Trend Analysis",
        "🌐 Network Analysis",
        "📚 Content Analysis",
        "🎯 Competitive Intelligence"
    ])
    
    with viz_tabs[0]:  # Overview Dashboard
        st.subheader("📊 Research Overview Dashboard")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Results", len(results))
        with col2:
            avg_sentiment = np.mean([r.sentiment for r in results])
            st.metric("Avg Sentiment", f"{avg_sentiment:.3f}", 
                     delta=f"{'Positive' if avg_sentiment > 0 else 'Negative' if avg_sentiment < 0 else 'Neutral'}")
        with col3:
            avg_credibility = np.mean([r.credibility_score for r in results])
            st.metric("Avg Credibility", f"{avg_credibility:.3f}")
        with col4:
            unique_sources = len(set(r.source for r in results))
            st.metric("Unique Sources", unique_sources)
        with col5:
            avg_relevance = np.mean([r.relevance_score for r in results])
            st.metric("Avg Relevance", f"{avg_relevance:.3f}")
        
        # Quality distribution
        quality_data = pd.DataFrame({
            'Result': range(len(results)),
            'Sentiment': [r.sentiment for r in results],
            'Credibility': [r.credibility_score for r in results],
            'Relevance': [r.relevance_score for r in results],
            'Quality Score': [r.research_quality_score for r in results],
            'Source': [r.source for r in results]
        })
        
        fig_quality = px.scatter_3d(
            quality_data, 
            x='Sentiment', 
            y='Credibility', 
            z='Relevance',
            color='Quality Score',
            size='Quality Score',
            hover_data=['Source'],
            title="3D Quality Analysis",
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_quality, use_container_width=True)
    
    with viz_tabs[1]:  # Sentiment Analysis
        st.subheader("🎭 Advanced Sentiment Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution
            sentiments = [r.sentiment for r in results]
            fig_sent_dist = px.histogram(
                x=sentiments, 
                nbins=30,
                title="Sentiment Distribution",
                labels={'x': 'Sentiment Score', 'y': 'Frequency'},
                color_discrete_sequence=['#FF6B6B']
            )
            fig_sent_dist.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Neutral")
            st.plotly_chart(fig_sent_dist, use_container_width=True)
        
        with col2:
            # Sentiment by source
            source_sentiment = defaultdict(list)
            for result in results:
                source_sentiment[result.source].append(result.sentiment)
            
            source_avg_sentiment = {
                source: np.mean(sentiments) 
                for source, sentiments in source_sentiment.items()
            }
            
            fig_source_sent = px.bar(
                x=list(source_avg_sentiment.keys()),
                y=list(source_avg_sentiment.values()),
                title="Average Sentiment by Source",
                labels={'x': 'Source', 'y': 'Average Sentiment'},
                color=list(source_avg_sentiment.values()),
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_source_sent, use_container_width=True)
        
        # Sentiment timeline
        dated_results = [r for r in results if r.publication_date]
        if dated_results:
            dated_results.sort(key=lambda x: x.publication_date)
            
            timeline_data = pd.DataFrame({
                'Date': [r.publication_date for r in dated_results],
                'Sentiment': [r.sentiment for r in dated_results],
                'Title': [r.title[:50] + '...' for r in dated_results],
                'Source': [r.source for r in dated_results]
            })
            
            fig_timeline = px.scatter(
                timeline_data,
                x='Date',
                y='Sentiment',
                color='Source',
                hover_data=['Title'],
                title="Sentiment Timeline",
                trendline="lowess"
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    with viz_tabs[2]:  # Source Intelligence
        st.subheader("🔍 Source Intelligence Analysis")
        
        # Source credibility analysis
        source_stats = defaultdict(lambda: {'count': 0, 'credibility': [], 'sentiment': []})
        
        for result in results:
            source_stats[result.source]['count'] += 1
            source_stats[result.source]['credibility'].append(result.credibility_score)
            source_stats[result.source]['sentiment'].append(result.sentiment)
        
        source_analysis = []
        for source, stats in source_stats.items():
            source_analysis.append({
                'Source': source,
                'Count': stats['count'],
                'Avg Credibility': np.mean(stats['credibility']),
                'Avg Sentiment': np.mean(stats['sentiment']),
                'Reliability Score': np.mean(stats['credibility']) * 0.7 + (1 - abs(np.mean(stats['sentiment']))) * 0.3
            })
        
        source_df = pd.DataFrame(source_analysis)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Source bubble chart
            fig_bubble = px.scatter(
                source_df,
                x='Avg Credibility',
                y='Avg Sentiment',
                size='Count',
                color='Reliability Score',
                hover_name='Source',
                title="Source Analysis (Bubble Size = Result Count)",
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_bubble, use_container_width=True)
        
        with col2:
            # Source reliability ranking
            source_df_sorted = source_df.sort_values('Reliability Score', ascending=False)
            fig_reliability = px.bar(
                source_df_sorted,
                x='Reliability Score',
                y='Source',
                orientation='h',
                title="Source Reliability Ranking",
                color='Reliability Score',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_reliability, use_container_width=True)
    
    with viz_tabs[3]:  # Trend Analysis
        st.subheader("📈 Trend Analysis")
        
        # Publication frequency over time
        dated_results = [r for r in results if r.publication_date]
        if dated_results:
            # Group by month
            monthly_counts = defaultdict(int)
            monthly_sentiment = defaultdict(list)
            
            for result in dated_results:
                month_key = result.publication_date.strftime('%Y-%m')
                monthly_counts[month_key] += 1
                monthly_sentiment[month_key].append(result.sentiment)
            
            trend_data = []
            for month, count in monthly_counts.items():
                avg_sentiment = np.mean(monthly_sentiment[month])
                trend_data.append({
                    'Month': month,
                    'Publication Count': count,
                    'Average Sentiment': avg_sentiment
                })
            
            trend_df = pd.DataFrame(trend_data).sort_values('Month')
            
            # Create subplot with secondary y-axis
            fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_trend.add_trace(
                go.Bar(x=trend_df['Month'], y=trend_df['Publication Count'], name="Publication Count"),
                secondary_y=False
            )
            
            fig_trend.add_trace(
                go.Scatter(x=trend_df['Month'], y=trend_df['Average Sentiment'], 
                          mode='lines+markers', name="Average Sentiment"),
                secondary_y=True
            )
            
            fig_trend.update_xaxes(title_text="Month")
            fig_trend.update_yaxes(title_text="Publication Count", secondary_y=False)
            fig_trend.update_yaxes(title_text="Average Sentiment", secondary_y=True)
            fig_trend.update_layout(title_text="Publication Trends Over Time")
            
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No dated results available for trend analysis")
    
    with viz_tabs[4]:  # Network Analysis
        st.subheader("🌐 Keyword Network Analysis")
        
        # Create keyword co-occurrence network
        all_keywords = []
        for result in results:
            all_keywords.extend(result.keywords[:5])  # Top 5 keywords per result
        
        keyword_counts = Counter(all_keywords)
        top_keywords = [kw for kw, count in keyword_counts.most_common(20)]
        
        if len(top_keywords) > 1:
            # Create co-occurrence matrix
            cooccurrence = defaultdict(int)
            for result in results:
                result_keywords = [kw for kw in result.keywords[:5] if kw in top_keywords]
                for i, kw1 in enumerate(result_keywords):
                    for kw2 in result_keywords[i+1:]:
                        key = tuple(sorted([kw1, kw2]))
                        cooccurrence[key] += 1
            
            # Create network graph
            G = nx.Graph()
            for keyword in top_keywords:
                G.add_node(keyword, size=keyword_counts[keyword])
            
            for (kw1, kw2), weight in cooccurrence.items():
                if weight > 1:  # Only show strong connections
                    G.add_edge(kw1, kw2, weight=weight)
            
            # Calculate layout
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Create plotly network visualization
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            node_text = list(G.nodes())
            node_size = [keyword_counts[node] * 3 for node in G.nodes()]
            
            fig_network = go.Figure()
            
            # Add edges
            fig_network.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            ))
            
            # Add nodes
            fig_network.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition='middle center',
                marker=dict(size=node_size, color='lightblue', line=dict(width=2, color='darkblue')),
                hovertemplate='<b>%{text}</b><br>Frequency: %{marker.size}<extra></extra>'
            ))
            
            fig_network.update_layout(
                title="Keyword Co-occurrence Network",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[
                    dict(
                        text="Node size represents keyword frequency<br>Edges show co-occurrence",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(color='#999', size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            st.plotly_chart(fig_network, use_container_width=True)
        else:
            st.info("Insufficient keyword data for network analysis")
    
    with viz_tabs[5]:  # Content Analysis
        st.subheader("📚 Content Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Reading level distribution
            reading_levels = [r.reading_level for r in results]
            level_counts = Counter(reading_levels)
            
            fig_reading = px.pie(
                values=list(level_counts.values()),
                names=list(level_counts.keys()),
                title="Reading Level Distribution"
            )
            st.plotly_chart(fig_reading, use_container_width=True)
        
        with col2:
            # Content type distribution
            content_types = [r.content_type for r in results]
            type_counts = Counter(content_types)
            
            fig_content = px.bar(
                x=list(type_counts.keys()),
                y=list(type_counts.values()),
                title="Content Type Distribution",
                color=list(type_counts.values()),
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_content, use_container_width=True)
        
        # Word count analysis
        word_counts = [r.word_count for r in results if r.word_count > 0]
        if word_counts:
            fig_words = px.histogram(
                x=word_counts,
                nbins=20,
                title="Word Count Distribution",
                labels={'x': 'Word Count', 'y': 'Frequency'}
            )
            st.plotly_chart(fig_words, use_container_width=True)
        
        # Top keywords word cloud visualization
        all_keywords = []
        for result in results:
            all_keywords.extend(result.keywords)
        
        if all_keywords:
            keyword_freq = Counter(all_keywords)
            
            # Create a simple word frequency chart instead of word cloud
            top_20_keywords = dict(keyword_freq.most_common(20))
            
            fig_wordfreq = px.bar(
                x=list(top_20_keywords.values()),
                y=list(top_20_keywords.keys()),
                orientation='h',
                title="Top 20 Keywords",
                labels={'x': 'Frequency', 'y': 'Keywords'}
            )
            fig_wordfreq.update_layout(height=600)
            st.plotly_chart(fig_wordfreq, use_container_width=True)
    
    with viz_tabs[6]:  # Competitive Intelligence
        st.subheader("🎯 Competitive Intelligence")
        
        # Extract organizations/companies mentioned
        companies = []
        for result in results:
            for entity in result.entities:
                if entity.entity_type in ['ORG', 'COMPANY']:
                    companies.append(entity.name)
        
        if companies:
            company_mentions = Counter(companies)
            top_companies = dict(company_mentions.most_common(10))
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Company mention frequency
                fig_companies = px.bar(
                    x=list(top_companies.keys()),
                    y=list(top_companies.values()),
                    title="Top Mentioned Organizations",
                    labels={'x': 'Organization', 'y': 'Mentions'}
                )
                fig_companies.update_xaxes(tickangle=45)
                st.plotly_chart(fig_companies, use_container_width=True)
            
            with col2:
                # Company sentiment analysis
                company_sentiment = defaultdict(list)
                for result in results:
                    for entity in result.entities:
                        if entity.entity_type in ['ORG', 'COMPANY'] and entity.name in top_companies:
                            company_sentiment[entity.name].append(result.sentiment)
                
                sentiment_data = []
                for company, sentiments in company_sentiment.items():
                    sentiment_data.append({
                        'Company': company,
                        'Average Sentiment': np.mean(sentiments),
                        'Mention Count': len(sentiments)
                    })
                
                if sentiment_data:
                    sentiment_df = pd.DataFrame(sentiment_data)
                    
                    fig_comp_sent = px.scatter(
                        sentiment_df,
                        x='Average Sentiment',
                        y='Company',
                        size='Mention Count',
                        title="Company Sentiment Analysis",
                        color='Average Sentiment',
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig_comp_sent, use_container_width=True)
        else:
            st.info("No organizational entities detected for competitive analysis")

def main():
    st.set_page_config(
        page_title="🧠 Advanced AI Research Assistant",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Advanced CSS styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
        background-color: #f7f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #e3e8ee;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        font-weight: 600;
        color: #222;
    }
    .stTabs [aria-selected="true"] {
        background: #fff;
        color: #0072ff;
        border-bottom: 2px solid #0072ff;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🧠 Advanced AI Research Assistant")
    st.markdown(
        "A next-generation research assistant for enterprise-grade, multi-source, AI-powered research and analysis."
    )
    
    # Sidebar options
    st.sidebar.header("🔎 Search Options")
    query = st.sidebar.text_input("Enter your research query", "")
    search_depth = st.sidebar.selectbox(
        "Search Depth",
        ["standard", "deep", "academic", "news"],
        index=0
    )
    max_results = st.sidebar.slider("Max Results", 10, 100, 30, 5)
    sentiment_filter = st.sidebar.selectbox(
        "Sentiment Filter",
        ["All", "Positive", "Negative", "Neutral"],
        index=0
    )
    date_range = st.sidebar.selectbox(
        "Date Range",
        ["All Time", "Last 24 Hours", "Last Week", "Last Month", "Last Year"],
        index=0
    )
    min_credibility = st.sidebar.slider("Min Credibility Score", 0.0, 1.0, 0.0, 0.05)
    content_types = st.sidebar.multiselect(
        "Content Types",
        ["web_page", "encyclopedia", "academic_paper", "medical_paper", "news_article", "social_media", "summary"],
        default=[]
    )
    reading_levels = st.sidebar.multiselect(
        "Reading Levels",
        ["Basic", "Intermediate", "Advanced"],
        default=[]
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Sources:** DuckDuckGo, Wikipedia, arXiv, PubMed, News, Social Media (simulated)")
    
    # Advanced options
    with st.sidebar.expander("⚙️ Advanced Source Options", expanded=False):
        include_web = st.checkbox("Include Web Search", True)
        include_wikipedia = st.checkbox("Include Wikipedia", True)
        include_academic = st.checkbox("Include Academic Papers", False)
        include_medical = st.checkbox("Include Medical Papers", False)
        include_news = st.checkbox("Include News", True)
        include_social = st.checkbox("Include Social Media", False)
    
    # Prepare filters
    filters = {
        "max_results": max_results,
        "sentiment_filter": sentiment_filter if sentiment_filter != "All" else None,
        "date_range": date_range,
        "min_credibility": min_credibility,
        "content_types": content_types if content_types else None,
        "reading_levels": reading_levels if reading_levels else None,
        "include_web": include_web,
        "include_wikipedia": include_wikipedia,
        "include_academic": include_academic,
        "include_medical": include_medical,
        "include_news": include_news,
        "include_social": include_social,
    }
    
    # Run search
    if query:
        with st.spinner("🔍 Conducting advanced research..."):
            assistant = ResearchAssistantPro()
            results = assistant.comprehensive_search(query, filters, search_depth)
        
        if results:
            st.success(f"Found {len(results)} results for '{query}'")
            create_advanced_visualizations(results, query)
            
            # Show results table
            with st.expander("📄 Show Raw Results Table", expanded=False):
                df = pd.DataFrame([{
                    "Title": r.title,
                    "Source": r.source,
                    "Sentiment": r.sentiment,
                    "Credibility": r.credibility_score,
                    "Relevance": r.relevance_score,
                    "Type": r.content_type,
                    "Date": r.publication_date,
                    "URL": r.url
                } for r in results])
                st.dataframe(df)
        else:
            st.warning("No results found. Try adjusting your filters or query.")
    else:
        st.info("Enter a research query in the sidebar to begin.")
    
if __name__ == "__main__":
    main()
