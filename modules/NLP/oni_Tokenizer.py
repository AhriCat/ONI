import re
from collections import defaultdict, Counter
import torch
from IPython.display import display, Image
from tokenizers import pre_tokenizers

class MultitokenBPETokenizer:
    def __init__(self, vocab_size=1000000, max_merges=30000, n_future_tokens=4, return_tensors=None, padding=None, truncation=None):
        self.vocab_size = vocab_size
        self.max_merges = max_merges
        self.n_future_tokens = n_future_tokens
        self.token_to_id = {
                "[PAD]": 0,
                "[UNK]": 1,
                "[CLS]": 2,
                "[SEP]": 3,
                "[MASK]": 4,
                "[EOS]": 5,
                "[BOS]": 6,
                "[USER]": 7,
                "[ASSISTANT]": 8,
                "[SYSTEM]": 9,
                "[CONVERSATION]": 10,
                "[TEXT]": 11,
                "[NAME]": 76,
                "[VISION]": 12,
                "[AUDIO]": 13,
                "[AGENTIC]": 14,
                "[EXPLORATORY]": 15,
                "[TASK-FOCUSED]": 16,
                "[URL]": 17,
                "[EMAIL]": 18,
                "[PHONE]": 19,
                "[DATE]": 20,
                "[TIME]": 21,
                "[NUMBER]": 22,
                "[PERCENT]": 23,
                "[MONEY]": 24,
                "[ORDINAL]": 25,
                "[QUANTITY]": 26,
                "[PRODUCT]": 27,
                "[LOCATION]": 28,
                "[PERSON]": 29,
                "[ORGANIZATION]": 30,
                "[ENTITY]": 31,
                "[CODE]": 32,
                "[JAVA]": 33,
                "[PYTHON]": 34,
                "[C++]": 35,
                "[HTML]": 36,
                "[CSS]": 37,
                "[JS]": 38,
                "[START]": 39,
                "[END]": 40,
                "[NEWLINE]": 41,
                "[TAB]": 42,
                "[SLOT]": 43,
                "[DOMAIN]": 44,
                "[INTENT]": 45,
                "[PARAM]": 46,
                "[VALUE]": 47,
                "[SUBJECT]": 48,
                "[OBJECT]": 49,
                "[RELATION]": 50,
                "[BEGIN]": 51,
                "[STOP]": 52,
                "[CLS1]": 53,
                "[CLS2]": 54,
                "[SEP1]": 55,
                "[SEP2]": 56,
                "[ANS]": 57,
                "[QUESTION]": 58,
                "[QUOTE]": 59,
                "[CITATION]": 60,
                "[EMOJI]": 61,
                "[HASHTAG]": 62,
                "[MENTION]": 63,
                "[LINK]": 64,
                "[FOOTER]": 65,
                "[HEADER]": 66,
                "[TITLE]": 67,
                "[SUBTITLE]": 68,
                "[FIGURE]": 69,
                "[TABLE]": 70,
                "[LIST]": 71,
                "[ITEM]": 72,
                "[BULLET]": 73,
                "[NUMBERED]": 74,
                "[END_OF_TEXT]": 75,
                # Future tokens for various contexts
                "[SIMULATION]": 77,
                "[SCENARIO-PLANNING]": 78,
                "[OPTIMIZATION]": 79,
                "[PRECISION]": 80,
                "[ITERATIVE-LEARNING]": 81,
                "[MULTI-AGENT]": 82,
                "[PERSONALITY]": 83,
                "[EMOTION]": 84,
                "[ADAPTATION]": 85,
                "[HAPTICS]": 86,
                "[DRONE]": 87,
                "[ROBOTIC]": 88,
                "[AR/VR]": 89,
                "[GEOSPATIAL]": 90,
                "[MACHINE]": 91,
                "[DECISION-ASSIST]": 92,
                "[NEUROINTERFACE]": 93,
                "[DIGITAL_TWIN]": 94,
                "[GENETIC]": 95,
                "[CLIMATE]": 96,
                "[QUANTUM]": 97,
                "[TELECOMM]": 98,
                "[ASTRO]": 99,
                "[REMOTE]": 100,
                "[BIOLOGY]": 101,
                "[3D_MODEL]": 102,
                "[AFFECT]": 103,
                "[ETHICS]": 104,
                "[EXPERIMENT]": 105,
                "[TRANSLATION]": 106,
                "[SENTIMENT]": 107,
                "[METADATA]": 108,
                "[SIMULATED_ENV]": 109,
                "[CRITICAL_TASK]": 110,
                "[AUTOMATED]": 111,
                "[REINFORCEMENT]": 112,
                "[HEALTH]": 113,
                "[HISTORICAL]": 114,
                "[FORECAST]": 115,
                "[RESOURCE]": 116,
                "[CONTEXTUAL]": 117,
                "[CUSTOMER]": 118,
                "[SENSOR]": 119,
                "[BIOMETRIC]": 120,
                "[MOBILITY]": 121,
                "[ADVISORY]": 122,
                "[CONSTRUCT]": 123,
                "[PERCEPTION]": 124,
                "[PLANNING]": 125,
                "[COMPLIANCE]": 126,
                "[MONITORING]": 127,
                "[IDENTITY]": 128,
                "[PRIVACY]": 129,
                "[IMMERSIVE]": 130,
                "[SUSTAINABILITY]": 131,
                "[FEEDBACK]": 132,
                "[MACHINE_LEARNING]": 133,
                "[ENVIRONMENTAL]": 134,
                "[ASSIST]": 135,
                "[PHYSICAL]": 136,
                "[EMBODIED]": 137,
                "[REPLICATION]": 138,
                "[ANALOG]": 139,
                "[IMMUNOLOGY]": 140,
                "[EDUCATIONAL]": 141,
                "[DEEP_LEARNING]": 142,
                "[POLICY]": 143,
                "[SECURE]": 144,
                "[SYNTHETIC]": 145,
            }
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab = {}
        self.merges = {}
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+""")
        self.pad_token = "[PAD]"
        self.pad_token_id = 0
        self.eos_token = "[EOS]"
        self.eos_token_id = 5
        self.return_tensors = return_tensors
        self.padding = padding
        self.truncation = truncation
        self.pre_tokenizer = pre_tokenizers.Whitespace()

    def __call__(self, text, role=None, mode=None, modality=None, 
                return_tensors=None, padding=None, truncation=None):
        if isinstance(text, list):
            tokens = [self.tokenize(t, role, mode, modality) for t in text]
        else:
            tokens = self.tokenize(text, role, mode, modality)
        
        if isinstance(tokens[0], list):
            token_ids = [[self.token_to_id.get(token, self.token_to_id['[UNK]']) for token in t] for t in tokens]
        else:
            token_ids = [self.token_to_id.get(token, self.token_to_id['[UNK]']) for token in tokens]
        
        token_ids = [ids + [self.token_to_id['[EOS]']] for ids in token_ids]

        # Pad sequences
        max_len = padding if padding else max(len(ids) for ids in token_ids)
        padded_token_ids = self.pad_sequences(token_ids, max_len)

        if return_tensors == 'pt':
            padded_token_ids = torch.tensor(padded_token_ids).to(torch.bfloat16)

        # Create attention mask
        attention_mask = [[1 if id != self.token_to_id['[PAD]'] else 0 for id in seq] for seq in padded_token_ids]

        return {
            'input_ids': torch.tensor(padded_token_ids).to(torch.bfloat16) if return_tensors == 'pt' else padded_token_ids,
            'attention_mask': torch.tensor(attention_mask) if return_tensors == 'pt' else attention_mask
        }
    def build_vocab(self, token_counter, min_frequency=1):
        vocab = {char: freq for char, freq in token_counter.items() if freq >= min_frequency}
        for char in vocab:
            if char not in self.token_to_id:
                self.token_to_id[char] = len(self.token_to_id)
                self.id_to_token[len(self.id_to_token)] = char
        return vocab

    def train(self, texts, min_frequency=1):
        token_counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                token_counter[token] += 1

        vocab = self.build_vocab(token_counter, min_frequency)

        merge_count = 0
        while len(self.token_to_id) < self.vocab_size and merge_count < self.max_merges:
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_token = "".join(best)
            if new_token not in self.token_to_id:
                self.token_to_id[new_token] = len(self.token_to_id)
                self.id_to_token[len(self.id_to_token)] = new_token
            self.merges[best] = new_token
            vocab = self.merge_vocab(best, vocab)
            merge_count += 1

        print(f"Vocabulary size: {len(self.token_to_id)}")
        print("Vocabulary:", self.token_to_id)

    def get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def tokenize(self, text, role=None, mode=None, modality=None):
        role = role if role else "user"
        mode = mode if mode else "assistant"
        modality = modality if modality else "text"

        if isinstance(text, list):
            # Process each text separately and collect tokens
            all_tokens = []
            for t in text:
                tokens = [
                    f"[{role.upper()}]",
                    f"[{mode.upper()}]",
                    f"[{modality.upper()}]",
                ]
                tokens.extend([token for token in self.pattern.findall(t.lower()) if token.strip()])
                all_tokens.append(tokens)
            return all_tokens
        else:
            tokens = [
                f"[{role.upper()}]",
                f"[{mode.upper()}]",
                f"[{modality.upper()}]",
            ]
            tokens.extend([token for token in self.pattern.findall(text.lower()) if token.strip()])
            return tokens

    def encode(self, text, role=None, mode=None, modality=None):
        text = str(text)  # Convert to string
    # rest of the method
        tokens = self.tokenize(text, role, mode, modality)
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                for char in token:
                    ids.append(self.token_to_id.get(char, self.token_to_id['[UNK]']))
        ids.append(self.token_to_id['[EOS]'])
        return torch.tensor(ids)

    def encode_plus(self, texts, roles=None, modes=None, modalities=None):
        encoded_texts = []
        for i, text in enumerate(texts):
            role = roles[i] if roles else None
            mode = modes[i] if modes else None
            modality = modalities[i] if modalities else None
            encoded_texts.append(self.encode(text, role, mode, modality))
        return encoded_texts

    def pad_sequences(self, sequences, max_length, pad_value=0):
        padded_sequences = []
        for seq in sequences:
            # Convert the sequence to a list if it's a tensor
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()
            
            # Pad or truncate each sequence as needed
            if len(seq) < max_length:
                seq = seq + [pad_value] * (max_length - len(seq))
            else:
                seq = seq[:max_length]
            
            # Convert each padded sequence to a tensor
            padded_sequences.append(torch.tensor(seq))
        
        # Stack the list of tensors into a single tensor
        return torch.stack(padded_sequences)


    def decode(self, ids, skip_special_tokens=True, pad_to_max_length=None):
        # Ensure ids is a tensor or convert to tensor
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids)
        
        # Handle padding if specified
        if pad_to_max_length is not None:
            # Pad or truncate the tensor to the specified length
            if ids.numel() < pad_to_max_length:
                padding = torch.full((pad_to_max_length - ids.numel(),), 
                                     self.token_to_id['[PAD]'], 
                                     dtype=ids.dtype)
                ids = torch.cat([ids, padding])
            else:
                ids = ids[:pad_to_max_length]
        
        # Convert to list of integers
        ids_list = ids.tolist()

        # List of special tokens to potentially remove
        special_tokens_to_remove = [
            '[CLS]', '[SEP]', '[MASK]', '[EOS]', '[USER]', '[ASSISTANT]',
            '[SYSTEM]', '[CONVERSATION]', '[TEXT]', '[VISION]', '[AUDIO]',
            '[AGENTIC]', '[EXPLORATORY]', '[TASK-FOCUSED]', '[URL]', '[EMAIL]',
            '[PHONE]', '[DATE]', '[TIME]', '[NUMBER]', '[PERCENT]', '[MONEY]',
            '[ORDINAL]', '[QUANTITY]', '[PRODUCT]', '[LOCATION]', '[PERSON]',
            '[ORGANIZATION]', '[ENTITY]', '[CODE]', '[JAVA]', '[PYTHON]',
            '[C++]', '[HTML]', '[CSS]', '[JS]', '[START]', '[END]',
            '[NEWLINE]', '[TAB]', '[SLOT]', '[DOMAIN]', '[INTENT]', '[PARAM]',
            '[VALUE]', '[SUBJECT]', '[OBJECT]', '[RELATION]', '[BEGIN]',
            '[STOP]', '[CLS1]', '[CLS2]', '[SEP1]', '[SEP2]', '[ANS]',
            '[QUESTION]', '[QUOTE]', '[CITATION]', '[EMOJI]', '[HASHTAG]',
            '[MENTION]', '[LINK]', '[FOOTER]', '[HEADER]', '[TITLE]',
            '[SUBTITLE]', '[FIGURE]', '[TABLE]', '[LIST]', '[ITEM]',
            '[BULLET]', '[NUMBERED]', '[END_OF_TEXT]', '[NAME]'
        ]

        # Create tokens list
        tokens = [
            self.id_to_token[id]
            for id in ids_list
            if id in self.id_to_token and id != self.token_to_id['[PAD]']
        ]

        # Conditionally remove special tokens
        if skip_special_tokens:
            tokens = [
                token for token in tokens if token not in special_tokens_to_remove
            ]

        # Join tokens into a string
        text = ''.join(tokens).replace('▁', ' ').strip()

        return text

    def decode_plus(self, batch_ids, skip_special_tokens=True, pad_to_max_length=None):
        return [
            self.decode(ids, skip_special_tokens, pad_to_max_length) 
            for ids in batch_ids
        ]
 
    def batch_encode(self, texts, roles=None, modes=None, modalities=None, max_length=6400):
        encoded_texts = self.encode_plus(texts, roles, modes, modalities)
        return self.pad_sequences(encoded_texts, max_length)

    def batch_decode(self, batch_ids):
        return self.decode_plus(batch_ids)

    def predict_future_tokens(self, text, model, role=None, mode=None, modality=None):
        encoded_text = self.encode(text, role, mode, modality)
        input_ids = encoded_text.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
        
        # Get the last token's logits
        last_token_logits = logits[0, -1, :]
        
        # Get the top n_future_tokens predictions
        top_k_values, top_k_indices = torch.topk(last_token_logits, self.n_future_tokens)
        
        predicted_tokens = []
        for idx in top_k_indices:
            predicted_tokens.append(self.id_to_token[idx.item()])
        
        return predicted_tokens

    def generate_text(self, text, model, role=None, mode=None, modality=None, max_length=100):
        input_ids = self.encode(text, role, mode, modality)
        
        for _ in range(max_length):
            with torch.no_grad():
                outputs = model(input_ids.unsqueeze(0))
                next_token_logits = outputs.logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).unsqueeze(0)
                
                if next_token.item() == self.token_to_id['[EOS]']:
                    break
                
                input_ids = torch.cat([input_ids, next_token])
        
        return self.decode(input_ids)

# Example usage
tokenizer = MultitokenBPETokenizer(vocab_size=10000, max_merges=1000, n_future_tokens=4)

# Example texts with roles, modes, modalities, and agentic modes
texts = [
    "Developing a quantum computing algorithm.",  # Scientist - exploratory
    "Analyzing stock market trends using predictive models.",  # Trader - task-focused
    "Creating a visual representation of the data.",  # Assistant - creative
    "Scraping data from the web for analysis.",  # Crawler - data-gathering
    "Compiling code for a new software release.",  # Dev - execution
    "Setting up a new machine learning experiment.",  # Scientist - experimental
    "Generating a 3D model based on input parameters.",  # Assistant - creative
    "Translating text to multiple languages.",  # Assistant - linguistic
    "Running simulations to test hypotheses.",  # Scientist - analytical
    "Optimizing a portfolio for maximum returns.",  # Trader - optimization
    "Parsing natural language inputs to structured data.",  # NLP Engine - processing
    "Detecting objects in images using a trained model.",  # Vision System - recognition
    "Summarizing a research paper for a quick overview.",  # Assistant - summarization
    "Calculating the trajectory of a spacecraft.",  # Calculator - precision
    "Monitoring network traffic for anomalies.",  # System - security
    "Generating audio responses based on text input.",  # Voice Assistant - interaction
    "Simulating economic scenarios for risk assessment.",  # Economist - scenario-planning
    "Designing a user interface for an application.",  # Dev - UI/UX
    "Running an experiment to test user engagement.",  # System - experimental
    "Training a neural network for image classification.",  # Scientist - iterative-learning
    "Executing a high-frequency trading strategy.",  # Trader - execution
    "Processing real-time data for trend analysis.",  # Crawler - real-time-processing
    "Developing a chatbot with multiple personalities.",  # Dev - multi-agent
    "Composing music using AI-generated melodies.",  # Assistant - creative
]
roles = [
    "scientist", "trader", "assistant", "crawler", "dev", "scientist", 
    "assistant", "assistant", "scientist", "trader", "nlp engine", 
    "vision system", "assistant", "calculator", "system", "voice assistant", 
    "economist", "dev", "system", "scientist", "trader", "crawler", 
    "dev", "assistant"
]
modes = [
    "exploratory", "task-focused", "creative", "data-gathering", "execution", "experimental",
    "creative", "linguistic", "analytical", "optimization", "processing", 
    "recognition", "summarization", "precision", "security", "interaction", 
    "scenario-planning", "UI/UX", "experimental", "iterative-learning", 
    "execution", "real-time-processing", "multi-agent", "creative"
]
modalities = [
    "text", "financial data", "visual", "web", "code", "lab equipment", 
    "3D model", "translation", "simulation", "portfolio", "text", 
    "image", "text", "math", "network", "audio", 
    "economic model", "design", "experiment", "image", 
    "market data", "web", "interface", "music"
]



# Train the tokenizer
tokenizer.train(texts)

# Encode texts with roles, modes, modalities, and agentic modes
encoded_texts = tokenizer.batch_encode(texts, roles, modes, modalities, max_length=6400)
decoded_texts = tokenizer.batch_decode(encoded_texts)

print("Encoded Texts:", encoded_texts)
print("Decoded Texts:", decoded_texts)
