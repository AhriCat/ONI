working_data = []
vocab = []
#5 most recent items in vocab
semantic_data = vocab[0:10]



class Memory():
    def __init__(self, working_memory_capacity=5, stm=[], ltm=[], ltm_capacity=1000000000000000000000, corpus=[],episodic_memory=[], semantic_memory=[]):
        self.load_long_term_memory(ltm)
        self.working_memory = []  # Short-term working memory as a list
        self.semantic_memory = semantic_memory  # Store generalized knowledge
        self.ltm_path = os.path.join('Downloads/ONI/ltm_path/', "ltm_data.json")
        self.corpus = corpus  # Initial semantic knowledge
        self.working_memory_capacity = working_memory_capacity
        self.ltm_capacity = ltm_capacity
        self.stm = []
        self.models = {}
        self.ltm = ltm
        self.episodic_memory = []


    def handle_media(self, data):
        """Handle different types of media files for episodic memory."""
        media_extensions = {
            '.mov': 'video',
            '.mp4': 'video',
            '.avi': 'video',
            '.wav': 'audio',
            '.mp3': 'audio',
            '.txt': 'plaintext',
            '.pdf': 'PDF',
            '.doc': 'Document File',
            'ODT': 'Open Document TXT',
            '.py': 'python file',
            '.html': 'website',
            '.js': 'javascript',
            '.css': 'styles'
        }
        file_extension = os.path.splitext(data)[1]
        media_type = media_extensions.get(file_extension, 'unknown')

        if media_type in ['video', 'audio']:
            return {'type': media_type, 'path': data}
        else:
            return None
    def load_long_term_memory(self, ltm):
        """Load LTM from a file."""
        ltm_path = os.path.join('Documents/PATH/ONI/ltm_path', 'ltm_path.json')
        if os.path.exists(ltm_path):
            with open(os.path.join(ltm_path, 'ltm_path.json'), 'r') as file:
                loaded_data = json.load(file)
                self.loaded_memory = loaded_data
                for tokens in loaded_data.get('ltm_path.txt', []):
                    if tokens not in self.semantic_memory:
                        self.semantic_memory += tokens

    @classmethod
    def update_episodic_memory(cls, data, data_type, key):
        if not hasattr(cls, 'episodic_memory'):
            cls.episodic_memory = {}
        cls.episodic_memory[key] = {'data': data, 'type': data_type}

    @classmethod
    def retrieve_from_episodic(cls, key):
        if not hasattr(cls, 'episodic_memory'):
            cls.episodic_memory = {}
        return cls.episodic_memory.get(key, {}).get('data', None)


    def update_semantic_memory(self, text, media_type=None):
        """
        Add text data to semantic memory.

        Args:
            text (str): The text data to add to the semantic memory.
            media_type (str, optional): The type of media from which the text is extracted.
                                        Expected values are '.txt', '.pdf', '.doc', '.py', '.js', '.css', '.html'.
                                        Default is None.
        """
        if media_type:
            if media_type not in ['.txt', '.pdf', '.doc', '.py', '.js', '.css', '.html']:
                raise ValueError("Unsupported media type")

        def current_index(tokens, text):
                for tokens in text:
                    token = current_index
                    tokens = text.split()
                    for token in tokens:
                        if token not in self.semantic_memory:
                            self.semantic_memory[token] = current_index
                            current_index += 1
                            return current_index

        # Ensure we respect the ltm_capacity
        if len(self.semantic_memory) > self.ltm_capacity:
            # Here you could implement some strategy to remove old entries
            # For simplicity, we are not implementing it in this example
            raise MemoryError("Semantic memory has exceeded its capacity")


    def lookup_token(self, token):
        """
        Lookup the index of a token in the semantic memory.

        Args:
            token (str): The token to lookup.

        Returns:
            int: The index of the token in the semantic memory.
        """
        return self.semantic_memory.get(token, -1)

# Example usage:
# Assuming self.semantic_memory is a list and self.ltm_capacity is an integer.


#memory consilidation and clarity (mindfulness of own memory)
    def meditate(self):
        """ Compress and refine semantic and working memory. """
        unique_data = set(self.semantic_memory + self.working_memory)
        self.ltm = list(unique_data)[:self.ltm_capacity]

#normal sleep (self learning while wait)
    def dream(self, model_name='nlp_model.h5'):
        """ Analyze and consolidate episodic memory using a neural network model. """
        if model_name in self.models:
            model = self.models[model_name]
        else:
            model = load_model(model_name)
            self.models[model_name] = model

        for memory in self.episodic_memory:
# Assuming each memory item is path to media, and model can process this directly
            data = self.handle_media(memory['path'])
            if data:
                processed_data = model.predict(data)
                memory['processed'] = processed_data
#deep sleep
    def sleep(self, model_name='nlp_model.h5'):
        """ Main sleep function to stop all processes and consolidate memories. """
        print("AI is going to sleep...")
        self.meditate()
        self.dream(model_name=model_name)
        self.save_long_term_memory()
        print("AI has woken up with refreshed memories.")

    def save_long_term_memory(self):
            """Save compressed long-term memory."""
            ltm = self.ltm
            with open("C:/users/jonny/Documents/PATH/ONI/ltm_path/ltm_data.json", 'w') as file:

              json.dump(ltm, file)
              print("Long-term memory saved.")

    def handle_pdf(self, pdf_path):
        """Extract text from PDF and add to semantic memory."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    self.update_semantic_memory(text)

            print(f"PDF '{os.path.basename(pdf_path)}' added to semantic memory.")
        except Exception as e:
            print(f"Error processing PDF: {e}")

    def update_memory(self, stm_data, episodic_data=None, semantic_data=None):
        """Update all memory stores with new data."""
        self.update_working_memory(working_data)
        if working_data not in list(stm):
            stm.append(working_data)
        if episodic_data:
            self.update_episodic_memory(episodic_data)
        if semantic_data:
            self.update_semantic_memory(semantic_data)
        self.save_long_term_memory()

    def categorize_and_store(self, db_path='Documents/PATH/ltm_path/personality.db'):
        """Categorize memory items and store them in a SQLite database."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS personalities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                race TEXT,
                origin TEXT,
                age INTEGER,
                type TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                personality_id INTEGER,
                timestamp TEXT,
                input_text TEXT,
                response_text TEXT,
                FOREIGN KEY (personality_id) REFERENCES personalities(id)
            )
        """)
        conn.commit()
        conn.close()


# Usage
mem = Memory()
ltm = Memory().ltm
#mem.update_memory("New STM data")
#memory.update_memory(episodic_data="example.mp3")  # Example for audio
#mem.update_memory(semantic_data=("General knowledge about AI":'machine learning'))  # Example for semantic memory
#mem.update_semantic_memory( 'Documents/PATH/ONI/IdentityPersonalANDSocialinpressreformatted.pdf', media_type='.pdf')
save = mem.save_long_term_memory()
#load = mem.load_long_term_memory(ltm)
vocab = []
while mem == Memory():
    mem.working_memory += input_data
    update_memory
    update_semantic_memory(text = vocab, mediatype=None)
    update_long_term_memory
print("Memory system upgraded with episodic and semantic capabilities.")
print(mem.ltm)