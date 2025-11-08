# BPE Tokenizer Experiments

This branch contains implementations of Byte Pair Encoding (BPE) tokenizers for Tamil language and hybrid Tamil-Stock market data.

## 🚀 Live Demo

Try the tokenizer interactively: [HuggingFace Space - Indic Language Stock Tokenizer](https://huggingface.co/spaces/malarsaravanan/indic_language_stock_tokenizer)

---

## Experiment 1: Indic Language Tokenizer (Tamil)

### Objective
Train a BPE tokenizer for Tamil, an Indic language with complex morphology and agglutinative properties.

### Dataset
- **Source**: HuggingFace Wikipedia corpus (`wikimedia/wikipedia`, 20231101.ta)
- **Corpus Size**: 50,000 documents
- **Language**: Tamil (தமிழ்)

### Results
| Metric | Value | Status |
|--------|-------|--------|
| **Vocabulary Size** | 8,000 tokens | ✅ Pass |
| **Compression Ratio** | 4.67x | ✅ Pass |
| **Dataset Source** | HuggingFace | ✅ Verified |

### Target Requirements
- ✅ Vocabulary: ≥ 5,000 tokens
- ✅ Compression: ≥ 3.0x
- ✅ Indic Language: Tamil

### Analysis

**Tamil Language Characteristics in BPE:**

The Tamil tokenizer demonstrates effective handling of Indic script through ByteLevel encoding:

1. **High Compression Efficiency (4.67x)**
   - Tamil's UTF-8 characters are 3-4 bytes each
   - BPE successfully learns multi-byte patterns
   - Common Tamil morphemes are captured as single tokens

2. **Vocabulary Distribution**
   - BPE learns frequent Tamil word parts (stems, suffixes)
   - Agglutinative nature results in productive token reuse
   - 8,000 tokens provide good coverage for Tamil text

3. **ByteLevel Encoding Benefits**
   - Handles Tamil Unicode (U+0B80 to U+0BFF) natively
   - No special preprocessing required
   - Maintains script integrity during tokenization

4. **Key Insights**
   - Tamil's complex orthography benefits from subword tokenization
   - BPE effectively handles consonant clusters and vowel markers
   - Compression ratio exceeds target, indicating efficient token learning

---

## Experiment 2: Hybrid Tokenizer (Tamil + Stock Market Data)

### Objective
Train a BPE tokenizer that handles both Tamil language text and English financial terminology, creating a dual-domain tokenizer for financial NLP applications.

### Dataset
- **Tamil Source**: HuggingFace Wikipedia corpus (`wikimedia/wikipedia`, 20231101.ta)
- **Stock Source**: Financial news (`zeroshot/twitter-financial-news-sentiment`)
- **Corpus Size**: 30,000 documents (10% Tamil, 90% Stock)
- **Stock Format**: Natural language (e.g., "$AAPL surged +2.5% on strong earnings")
- **Live Demo**: [Try the hybrid tokenizer on HuggingFace Spaces](https://huggingface.co/spaces/malarsaravanan/indic_language_stock_tokenizer)

### Combined Results
| Metric | Value | Status |
|--------|-------|--------|
| **Total Vocabulary** | 40,000 tokens | ✅ Pass |
| **Overall Compression** | 5.78x | ✅ Pass |
| **Tamil Tokens** | 35,991 (90.0%) | ✅ Pass |
| **Stock Tokens** | 5,572 (13.9%) | ✅ Pass |
| **Dataset Composition** | 10% Tamil / 90% Stock | ✅ Balanced |

### Individual Domain Performance
| Domain | Vocabulary | Compression | Characteristics |
|--------|-----------|-------------|-----------------|
| **Tamil** | 35,991 tokens | 5.12x | Byte-encoded characters |
| **Stock** | 5,572 tokens | 4.90x | English vocabulary |

### Target Requirements
- ✅ Tamil Vocabulary: ≥ 5,000 tokens (achieved 35,991)
- ✅ Stock Vocabulary: ≥ 5,000 tokens (achieved 5,572)
- ✅ Compression: ≥ 4.0x (achieved 5.78x)
- ✅ Dual Domain: Tamil + Stock Market Data

### Analysis

**Impact of Combining Tamil and Stock Data in BPE:**

1. **Vocabulary Distribution Asymmetry**
   - Tamil dominates vocabulary (90%) despite only 10% of training data
   - **Cause**: ByteLevel encoding creates more tokens per Tamil character (3-4 bytes)
   - **Cause**: English stock terms are single-byte, requiring fewer unique tokens
   - **Result**: 10/90 data split → 90/14 vocabulary split

2. **ByteLevel Encoding Impact**
   - **Tamil**: Each character → 3-4 byte tokens → More vocabulary needed
   - **Stock**: Each word → 1-7 byte tokens → Less vocabulary needed
   - Example: Tamil "தமிழ்" (5 chars) → ~15 bytes → Multiple tokens
   - Example: Stock "Apple" (5 chars) → 5 bytes → 1-2 tokens

3. **Domain-Specific Token Learning**
   - **Tamil tokens**: Morphological patterns (stems, suffixes, compound words)
   - **Stock tokens**: Company names (Apple, TCS, Reliance), financial terms (surge, bullish, revenue)
   - Both domains maintain high compression (5.1x Tamil, 4.9x Stock)

4. **Hybrid Tokenizer Advantages**
   - Can process mixed Tamil-English financial text
   - Handles code-switching naturally (e.g., "ரிலையன்ஸ் stock rose to 2480")
   - Suitable for Tamil financial news, trading platforms, and market analysis

5. **Key Insights**
   - ByteLevel encoding naturally biases vocabulary toward high-byte scripts
   - 90% stock training data compensates for English's byte efficiency
   - Final tokenizer successfully learns both domains (35K Tamil + 5.5K Stock)
   - Compression ratios remain high for both domains (~5x)

6. **Real-World Application**
   - Tamil financial news tokenization
   - Bilingual trading platform interfaces
   - Tamil stock market sentiment analysis
   - Mixed-language financial document processing

---

## Repository Structure

```
assignment_11/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
│
├── tamil_bpe_tokenizer/               # Experiment 1
│   ├── tamil_bpe_training.ipynb       # Training notebook
│   ├── tamil_bpe_tokenizer.json       # Trained tokenizer
│   └── tokenizer_summary.json         # Results summary
│
└── hybrid_tamil_stock_tokenizer/     # Experiment 2
    ├── hybrid_tamil_stock_bpe_training.ipynb  # Training notebook
    ├── hybrid_tamil_stock_tokenizer.json      # Trained tokenizer
    └── hybrid_tokenizer_summary.json          # Results summary
```

---

## Setup and Installation

### Prerequisites
- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended)
- Internet connection for dataset downloads

### Installation

```bash
# Clone repository
cd assignment_11

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# Or: venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Experiments

**Experiment 1: Tamil Tokenizer**
```bash
jupyter notebook tamil_bpe_tokenizer/tamil_bpe_training.ipynb
```
Training time: ~10-15 minutes

**Experiment 2: Hybrid Tokenizer**
```bash
jupyter notebook hybrid_tamil_stock_tokenizer/hybrid_tamil_stock_bpe_training.ipynb
```
Training time: ~45-60 minutes

---

## Usage Examples

### Tamil Tokenizer
```python
from tokenizers import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_file("tamil_bpe_tokenizer/tamil_bpe_tokenizer.json")

# Tokenize Tamil text
text = "தமிழ் மொழி இந்தியாவின் பழமையான மொழிகளில் ஒன்று"
encoding = tokenizer.encode(text)

print(f"Text: {text}")
print(f"Tokens: {encoding.tokens}")
print(f"Token count: {len(encoding.tokens)}")
```

### Hybrid Tokenizer
```python
from tokenizers import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_file("hybrid_tamil_stock_tokenizer/hybrid_tamil_stock_tokenizer.json")

# Tokenize hybrid Tamil + Stock text
text = "ரிலையன்ஸ் பங்கு $Reliance rose to 2480 +1.2% இன்று"
encoding = tokenizer.encode(text)

print(f"Text: {text}")
print(f"Tokens: {encoding.tokens}")
print(f"Token count: {len(encoding.tokens)}")
```

---

## Technical Details

### Algorithm
- **Method**: Byte Pair Encoding (BPE)
- **Pre-tokenizer**: ByteLevel
- **Normalizer**: NFD + StripAccents
- **Decoder**: ByteLevel

### Special Tokens
Both tokenizers include standard special tokens:
- `[UNK]` - Unknown token
- `[PAD]` - Padding token
- `[CLS]` - Classification token
- `[SEP]` - Separator token
- `[MASK]` - Mask token

---

## Dependencies

Key libraries (see `requirements.txt` for full list):
- `tokenizers==0.22.1` - HuggingFace tokenizers
- `datasets==4.3.0` - Dataset loading and processing
- `huggingface-hub==1.0.1` - HuggingFace Hub integration
- `numpy==2.3.4` - Numerical computing
- `tqdm==4.67.1` - Progress bars

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

Created as part of NLP coursework exploring BPE tokenization for Indic languages and hybrid multi-domain scenarios.

---

## Acknowledgments

- HuggingFace for tokenizers library and datasets
- Tamil Wikipedia contributors
- Financial news dataset providers
