from pathlib import Path
text = Path('mapf_egt/egt.py').read_text().splitlines()
for i,line in enumerate(text,1):
    if 50 <= i <= 120 or 150 <= i <= 220:
        print(f"{i}: {line}")
