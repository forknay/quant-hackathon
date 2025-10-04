"""
Stock Sentiment Ranking Processor

Processes FinBERT sentiment results from real stock TextData
and generates final sentiment-based rankings.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

def process_sentiment_rankings():
    """Process FinBERT results and create sentiment rankings"""
    print("🚀 PROCESSING REAL STOCK SENTIMENT RANKINGS")
    print("=" * 60)
    
    # Load FinBERT results
    results_file = "finbert_results_real_stocks_20251004_185131.csv"
    metadata_file = "metadata_real_stocks_20251004_185131.json"
    
    try:
        df_results = pd.read_csv(results_file)
        print(f"✅ Loaded {len(df_results)} sentiment results")
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        print("Make sure you've run the FinBERT processing first!")
        return
    
    # Load metadata
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"✅ Loaded metadata for {len(metadata.get('input_stocks', []))} input stocks")
    except FileNotFoundError:
        print(f"⚠️  Metadata file not found: {metadata_file}")
        metadata = {}
    
    # Convert FinBERT labels to numerical scores
    print("\n📊 Converting sentiment labels to scores...")
    
    df_results['positive_score'] = 0.0
    df_results['negative_score'] = 0.0
    df_results['neutral_score'] = 0.0
    
    for idx, row in df_results.iterrows():
        label = row['sentiment_label'].lower()
        score = row['sentiment_score']
        
        if label == 'positive':
            df_results.at[idx, 'positive_score'] = score
            df_results.at[idx, 'neutral_score'] = (1 - score) * 0.6
            df_results.at[idx, 'negative_score'] = (1 - score) * 0.4
        elif label == 'negative':
            df_results.at[idx, 'negative_score'] = score
            df_results.at[idx, 'neutral_score'] = (1 - score) * 0.6
            df_results.at[idx, 'positive_score'] = (1 - score) * 0.4
        else:  # neutral
            df_results.at[idx, 'neutral_score'] = score
            df_results.at[idx, 'positive_score'] = (1 - score) * 0.5
            df_results.at[idx, 'negative_score'] = (1 - score) * 0.5
    
    # Calculate net sentiment
    df_results['net_sentiment'] = df_results['positive_score'] - df_results['negative_score']
    
    print(f"Sentiment distribution:")
    print(f"  Positive: {(df_results['sentiment_label'] == 'Positive').sum()}")
    print(f"  Negative: {(df_results['sentiment_label'] == 'Negative').sum()}")
    print(f"  Neutral: {(df_results['sentiment_label'] == 'Neutral').sum()}")
    
    # Aggregate by stock
    print("\n📈 Aggregating sentiment by stock...")
    
    stock_sentiment = df_results.groupby('stock_identifier').agg({
        'gvkey': 'first',
        'iid': 'first',
        'positive_score': 'mean',
        'negative_score': 'mean', 
        'neutral_score': 'mean',
        'net_sentiment': 'mean',
        'sentiment_score': 'mean',
        'text_id': 'count',
        'text_type': lambda x: list(x.unique()),
        'year': lambda x: list(sorted(x.unique()))
    }).round(4)
    
    stock_sentiment.columns = ['gvkey', 'iid', 'avg_positive', 'avg_negative', 'avg_neutral',
                              'net_sentiment', 'avg_confidence', 'text_count', 'text_types', 'years']
    stock_sentiment = stock_sentiment.reset_index()
    
    # Calculate weighted sentiment score
    # Weight by confidence and log of text count
    stock_sentiment['weighted_sentiment'] = (
        stock_sentiment['net_sentiment'] * 
        stock_sentiment['avg_confidence'] * 
        np.log1p(stock_sentiment['text_count'])
    )
    
    # Add sentiment strength category
    def categorize_sentiment(net_sent):
        if net_sent > 0.2:
            return 'Strong Positive'
        elif net_sent > 0.05:
            return 'Positive'
        elif net_sent > -0.05:
            return 'Neutral'
        elif net_sent > -0.2:
            return 'Negative'
        else:
            return 'Strong Negative'
    
    stock_sentiment['sentiment_category'] = stock_sentiment['net_sentiment'].apply(categorize_sentiment)
    
    # Rank by weighted sentiment (best to worst)
    stock_sentiment = stock_sentiment.sort_values('weighted_sentiment', ascending=False)
    stock_sentiment['sentiment_rank'] = range(1, len(stock_sentiment) + 1)
    
    # Save detailed results
    output_file = f"stock_sentiment_rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    stock_sentiment.to_csv(output_file, index=False)
    
    print(f"\n🏆 FINAL SENTIMENT RANKINGS")
    print("=" * 60)
    display_cols = ['sentiment_rank', 'stock_identifier', 'net_sentiment', 
                   'sentiment_category', 'weighted_sentiment', 'text_count']
    print(stock_sentiment[display_cols].to_string(index=False))
    
    # Detailed analysis by text type
    print(f"\n📋 SENTIMENT BY TEXT TYPE")
    print("=" * 60)
    
    text_type_analysis = df_results.groupby(['stock_identifier', 'text_type']).agg({
        'net_sentiment': 'mean',
        'sentiment_score': 'mean',
        'text_id': 'count'
    }).round(4)
    
    print("Stock-wise sentiment by text type:")
    for stock_identifier in stock_sentiment['stock_identifier']:
        print(f"\n{stock_identifier}:")
        stock_texts = text_type_analysis.loc[stock_identifier] if stock_identifier in text_type_analysis.index else pd.DataFrame()
        if not stock_texts.empty:
            for text_type, row in stock_texts.iterrows():
                print(f"  {text_type}: {row['net_sentiment']:.4f} (confidence: {row['sentiment_score']:.4f}, texts: {row['text_id']})")
        else:
            print("  No text type breakdown available")
    
    # Summary statistics
    print(f"\n📊 SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total stocks analyzed: {len(stock_sentiment)}")
    print(f"Average net sentiment: {stock_sentiment['net_sentiment'].mean():.4f}")
    print(f"Sentiment range: {stock_sentiment['net_sentiment'].min():.4f} to {stock_sentiment['net_sentiment'].max():.4f}")
    print(f"Most positive stock: {stock_sentiment.iloc[0]['stock_identifier']} ({stock_sentiment.iloc[0]['net_sentiment']:.4f})")
    print(f"Most negative stock: {stock_sentiment.iloc[-1]['stock_identifier']} ({stock_sentiment.iloc[-1]['net_sentiment']:.4f})")
    
    # Sentiment distribution
    sentiment_dist = stock_sentiment['sentiment_category'].value_counts()
    print(f"\nSentiment distribution:")
    for category, count in sentiment_dist.items():
        print(f"  {category}: {count} stocks")
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Create summary report
    summary = {
        'processing_timestamp': datetime.now().isoformat(),
        'input_stocks': metadata.get('input_stocks', []),
        'total_stocks_analyzed': len(stock_sentiment),
        'total_texts_processed': len(df_results),
        'sentiment_rankings': stock_sentiment[['sentiment_rank', 'stock_identifier', 'net_sentiment', 'sentiment_category']].to_dict('records'),
        'summary_statistics': {
            'avg_sentiment': float(stock_sentiment['net_sentiment'].mean()),
            'sentiment_range': [float(stock_sentiment['net_sentiment'].min()), float(stock_sentiment['net_sentiment'].max())],
            'most_positive': stock_sentiment.iloc[0]['stock_identifier'],
            'most_negative': stock_sentiment.iloc[-1]['stock_identifier']
        },
        'sentiment_distribution': sentiment_dist.to_dict(),
        'output_file': output_file
    }
    
    summary_file = f"sentiment_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Summary saved to: {summary_file}")
    print("\n🎯 Stock sentiment analysis complete!")
    
    return output_file, summary_file

if __name__ == "__main__":
    process_sentiment_rankings()
