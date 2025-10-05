"""
Create mock FinBERT results to test the enhanced sentiment normalization system.
This simulates what would be returned from Lightning.ai after processing the real stock texts.
"""

import pandas as pd
import numpy as np
import json
import random

def create_mock_finbert_results():
    """Create realistic mock FinBERT results based on the real stock input data"""
    
    # Load the real input data to get the structure
    input_file = r"C:\Users\positive\Documents\GitHub\quant-hackathon\sentiment_analysis\lightning_ai\finbert_real_stocks_20251004_185131.csv"
    df_input = pd.read_csv(input_file)
    
    print(f"Creating mock FinBERT results for {len(df_input)} real stock texts...")
    
    # Create results based on the input texts
    results = []
    
    # Set random seed for reproducible results
    np.random.seed(42)
    random.seed(42)
    
    # Define realistic sentiment distributions based on typical financial text
    # Financial texts tend to be neutral or negative (discussing risks, uncertainties)
    sentiment_labels = ['positive', 'negative', 'neutral']
    
    for idx, row in df_input.iterrows():
        # Generate realistic sentiment based on text type and content
        text_type = row['text_type']
        stock_id = row['stock_identifier']
        
        # Risk factor texts (rf) tend to be more negative
        # Management discussion (mgmt) tends to be more neutral/mixed
        if text_type == 'rf':
            # Risk factors are usually negative or neutral
            sentiment_probs = [0.05, 0.65, 0.30]  # 5% pos, 65% neg, 30% neutral
        else:  # mgmt
            # Management discussion is more balanced but still conservative
            sentiment_probs = [0.15, 0.35, 0.50]  # 15% pos, 35% neg, 50% neutral
        
        # Add some stock-specific bias
        if stock_id == "1004:01":  # Slightly more positive for this stock
            sentiment_probs[0] += 0.10  # More positive
            sentiment_probs[1] -= 0.05  # Less negative
            sentiment_probs[2] -= 0.05  # Less neutral
        elif stock_id == "1050:01":  # Slightly more negative for this stock
            sentiment_probs[0] -= 0.05  # Less positive
            sentiment_probs[1] += 0.10  # More negative
            sentiment_probs[2] -= 0.05  # Less neutral
        
        # Normalize probabilities
        sentiment_probs = np.array(sentiment_probs)
        sentiment_probs = sentiment_probs / sentiment_probs.sum()
        
        # Sample sentiment
        sentiment = np.random.choice(sentiment_labels, p=sentiment_probs)
        
        # Generate confidence score - higher for clearer sentiments
        if sentiment == 'neutral':
            confidence = np.random.uniform(0.40, 0.75)  # Lower confidence for neutral
        else:
            confidence = np.random.uniform(0.65, 0.95)  # Higher confidence for pos/neg
        
        # Create result record
        result = {
            'text_id': row['text_id'],
            'stock_identifier': row['stock_identifier'],
            'gvkey': row['gvkey'],
            'iid': row['iid'],
            'text_type': row['text_type'],
            'year': row['year'],
            'sentiment_label': sentiment,
            'confidence_score': round(confidence, 4),
            'text_length': row['text_length']
        }
        
        results.append(result)
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Show distribution
    print(f"\nMock FinBERT Results Summary:")
    print(f"Total texts processed: {len(df_results)}")
    
    sentiment_dist = df_results['sentiment_label'].value_counts()
    for label, count in sentiment_dist.items():
        pct = (count / len(df_results)) * 100
        print(f"  {label.capitalize()}: {count} ({pct:.1f}%)")
    
    print(f"\nBy stock:")
    for stock in df_results['stock_identifier'].unique():
        stock_data = df_results[df_results['stock_identifier'] == stock]
        print(f"  {stock}: {len(stock_data)} texts")
        for label in ['positive', 'negative', 'neutral']:
            count = (stock_data['sentiment_label'] == label).sum()
            print(f"    {label}: {count}")
    
    # Save results
    output_file = "mock_finbert_results_real_stocks.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nMock FinBERT results saved to: {output_file}")
    
    return df_results

if __name__ == "__main__":
    mock_results = create_mock_finbert_results()