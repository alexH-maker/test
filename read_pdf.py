import PyPDF2
from pathlib import Path
path = Path('Multi Agent Path Finding using Evolutionary Game Theory.pdf')
reader = PyPDF2.PdfReader(str(path))
text = '\n'.join(page.extract_text() or '' for page in reader.pages)
print(text[:4000])
