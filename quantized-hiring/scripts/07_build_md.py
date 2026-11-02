"""Build paper/main.md from paper/main_template.md by splicing in the generated markdown tables."""
import re
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "paper"
t = (P / "main_template.md").read_text(encoding="utf-8")
t = re.sub(r"\{\{(\w+)\}\}", lambda m: (P / "tables" / f"{m.group(1)}.md").read_text(encoding="utf-8").strip(), t)
(P / "main.md").write_text(t, encoding="utf-8")
print("main.md written")
