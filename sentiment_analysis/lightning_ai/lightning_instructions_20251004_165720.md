# Lightning.ai FinBERT Processing Instructions

## Input Data
- **Main file**: `finbert_input_20251004_165720.csv`
- **Total texts**: 30
- **Model**: yiyanghkust/finbert-tone

## Setup Code for Lightning.ai

```python
# Install requirements
!pip install transformers torch pandas

# Import libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np

# Load FinBERT model
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
```

## Processing Code

```python
# Load your data
df = pd.read_csv('finbert_input_20251004_165720.csv')
texts = df['text'].tolist()
text_ids = df['text_id'].tolist()

# Process in batches
batch_size = 16
results = []

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_ids = text_ids[i:i+batch_size]
    
    # Tokenize
    inputs = tokenizer(
        batch_texts, 
        truncation=True, 
        padding=True, 
        max_length=512, 
        return_tensors='pt'
    ).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Store results
    for j, pred in enumerate(predictions.cpu().numpy()):
        results.append({
            'text_id': batch_ids[j],
            'negative': float(pred[0]),
            'neutral': float(pred[1]),
            'positive': float(pred[2])
        })
    
    if (i // batch_size + 1) % 10 == 0:
        print(f"Processed {i+batch_size}/{len(texts)} texts")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('finbert_sentiment_results.csv', index=False)
print(f"Results saved! Processed {len(results)} texts.")
```

## Expected Output
CSV file with sentiment scores for each text_id.

## Next Steps
1. Download the results file
2. Use the process_results.py script to analyze final rankings
