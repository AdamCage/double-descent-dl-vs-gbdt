import os
import re
from pathlib import Path

text = Path(os.environ["TEMP"], "torch-index.html").read_text(encoding="utf-8", errors="ignore")
hits = sorted(set(re.findall(r"torch-2\.[^\"']+cp31[12]-cp31[12]-win_amd64\.whl", text)))
print("count", len(hits))
for name in hits[-40:]:
    print(name)
