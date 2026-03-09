"""
Claude-powered sentiment analysis for stocks.
Fetches recent news via yfinance and uses the Anthropic API
to produce structured, analyst-style sentiment reports.
"""

import os
from datetime import datetime
from typing import Literal

import anthropic
import yfinance as yf
from pydantic import BaseModel


class ArticleSentiment(BaseModel):
    headline: str
    sentiment: Literal['bullish', 'bearish', 'neutral']
    impact: Literal['high', 'medium', 'low']
    key_point: str


class SentimentAnalysis(BaseModel):
    overall_sentiment: Literal['strongly_bullish', 'bullish', 'neutral', 'bearish', 'strongly_bearish']
    confidence: int          # 0–100
    sentiment_score: int     # -100 (very bearish) to +100 (very bullish)
    articles: list[ArticleSentiment]
    key_themes: list[str]
    bull_case: list[str]
    bear_case: list[str]
    analyst_take: str


def fetch_news(ticker_str: str) -> list[dict]:
    """Pull recent news articles from yfinance."""
    ticker = yf.Ticker(ticker_str)
    news = ticker.news or []
    articles = []
    for item in news[:12]:  # cap at 12 articles
        content = item.get('content', {})
        title = content.get('title') or item.get('title', '')
        summary = content.get('summary') or ''
        provider = content.get('provider', {})
        publisher = provider.get('displayName') if isinstance(provider, dict) else str(provider)
        pub_date = content.get('pubDate') or ''
        if title:
            articles.append({
                'title': title,
                'summary': summary[:400] if summary else '',
                'publisher': publisher or 'Unknown',
                'date': pub_date[:10] if pub_date else '',
            })
    return articles


def analyze_sentiment(ticker_str: str, company_name: str = '') -> dict | None:
    """
    Use Claude to analyze news sentiment for a stock.
    Returns a structured sentiment report or None if unavailable.
    """
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        return None

    articles = fetch_news(ticker_str)
    if not articles:
        return None

    news_block = '\n\n'.join(
        f"[{i+1}] {a['date']} | {a['publisher']}\n"
        f"Headline: {a['title']}\n"
        f"Summary: {a['summary'] or 'No summary available.'}"
        for i, a in enumerate(articles)
    )

    name_hint = f" ({company_name})" if company_name else ""
    today = datetime.now().strftime('%B %d, %Y')

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.parse(
        model='claude-opus-4-6',
        max_tokens=2048,
        system=(
            "You are a sharp, data-driven equity research analyst. "
            "Analyze news headlines and summaries to produce concise, structured sentiment reports. "
            "Be specific about what matters for investors. Do not give financial advice — "
            "focus on synthesizing information objectively."
        ),
        messages=[{
            'role': 'user',
            'content': (
                f"Today is {today}. Analyze the following recent news for {ticker_str}{name_hint} "
                f"and produce a structured sentiment report.\n\n"
                f"NEWS ARTICLES:\n{news_block}\n\n"
                f"For each article, classify sentiment and identify the key investment-relevant point. "
                f"Then synthesize an overall picture: what are the dominant themes, bull case, bear case, "
                f"and your concise analyst take (2-3 sentences max)."
            ),
        }],
        output_format=SentimentAnalysis,
    )

    result = response.parsed_output
    return {
        'overall_sentiment': result.overall_sentiment,
        'confidence': result.confidence,
        'sentiment_score': result.sentiment_score,
        'articles': [a.model_dump() for a in result.articles],
        'key_themes': result.key_themes,
        'bull_case': result.bull_case,
        'bear_case': result.bear_case,
        'analyst_take': result.analyst_take,
        'article_count': len(result.articles),
    }
