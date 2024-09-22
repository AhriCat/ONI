class FilePreprocessor:
    def __init__(self, path_to_file):
        self.path_to_file = path_to_file

        # List of rules/action key pair.
        # Add your new rule and how to process the text (function) here
        self.rules = [
            (self._is_python_file, self._process_if_python),
            (self._is_docx_file, self._process_if_docx),
            (self._is_pdf_file, self._process_if_pdf),
            (self._is_image_file, self._process_if_image),
            (self._is_text_file, self._process_if_text),
            # Add more rules for other file types as needed
        ]

    def process_file(self) -> str:
        """
        Process the file based on the internal rules.
        """
        for condition, action in self.rules:
            if condition():
                return action()
        return "Unsupported file format"

    def _is_python_file(self) -> bool:
        """
        Rule to check if the file is a Python file.
        """
        return self.path_to_file.endswith(".py")

    def _process_if_python(self) -> str:
        """
        Action to process Python files by checking for class definitions and indenting if found.
        """
        if self._contains_class_definition():
            with open(self.path_to_file, "r") as file:
                content = file.read()
            return textwrap.indent(content, "    ")
        return ""

    def _is_docx_file(self) -> bool:
        """
        Rule to check if the file is a DOCX file.
        """
        return self.path_to_file.endswith(".docx")

    def _process_if_docx(self) -> str:
        """
        Action to extract text from a DOCX file.
        """
        doc = docx.Document(self.path_to_file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text

    def _is_pdf_file(self) -> bool:
        """
        Rule to check if the file is a PDF file.
        """
        return self.path_to_file.endswith(".pdf")

    def _process_if_pdf(self) -> str:
        """
        Action to extract text from a PDF file.
        """
        with open(self.path_to_file, "rb") as file:
            pdf_reader = PyPDF2.PdfFileReader(file)
            text = ""
            for page_num in range(pdf_reader.numPages):
                page = pdf_reader.getPage(page_num)
                text += page.extract_text()
        return text

    def _is_image_file(self) -> bool:
        """
        Rule to check if the file is an image file (PNG or JPG).
        """
        return self.path_to_file.endswith(".png") or self.path_to_file.endswith(".jpg")

    def _process_if_image(self) -> str:
        """
        Action to extract text from an image file.
        """
        with Image.open(self.path_to_file) as img:
            text = pytesseract.image_to_string(img)
        return text

    def _is_text_file(self) -> bool:
        """
        Rule to check if the file is a plain text file.
        """
        return self.path_to_file.endswith(".txt")

    def _process_if_text(self) -> str:
        """
        Action to read text from a plain text file.
        """
        with open(self.path_to_file, "r") as file:
            text = file.read()
        return text

    def _contains_class_definition(self) -> bool:
        """
        Check if the file contains a Python class definition using the ast module.
        """
        try:
            with open(self.path_to_file, "r") as file:
                content = file.read()
            parsed_ast = ast.parse(content)
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.ClassDef):
                    return True
        except SyntaxError as e:
            print(f"Syntax error when parsing the file: {e}")
        return False
