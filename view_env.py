from pathlib import Path
text = Path('mapf_egt/env.py').read_text().splitlines()
for i in range(50, 90):
    if i < len(text):
        print(f"{i+1}: {text[i]}")
