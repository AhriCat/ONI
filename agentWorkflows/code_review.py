from pathlib import Path


def code_review_agent(oni: OniMicro, root: str = "."):
    """
    – Walk through a directory and gather Python files
    – Ask Oni to suggest improvements for each file
    – Return a consolidated report
    """
    py_files = Path(root).rglob("*.py")
    suggestions = []
    for fpath in py_files:
        source = fpath.read_text()
        suggestion = oni.nlp_module.generate(
            f"Review this code and suggest improvements:\n{source[:2000]}"
        )
        suggestions.append(f"File: {fpath}\n{suggestion}\n")
    report = "\n".join(suggestions)
    oni.save_report(report)
    return report


oni = OniMicro(
    tokenizer=tokenizer,
    input_dim=896,
    hidden_dim=896,
    output_dim=896,
    nhead=8,
    num_layers=20,
    exec_func=exec_func,
    state_size=256,
    action_size=20,
)

print(code_review_agent(oni))
