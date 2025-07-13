def research_and_summarize(oni: OniMicro, query: str, depth: int = 3) -> str:
    """
    – Opens the browser           – Performs a recursive Google/Bing search
    – Clicks top <depth> links    – Screenshots each page
    – OCRs the screenshot         – Feeds text through NLP → summary
    – Runs sentiment + compassion – Returns a polished synopsis
    """
    oni.open_browser()
    oni.perform_search(query)

    raw_text, synopses = [], []
    for k in range(1, depth + 1):
        oni.select_link(k)
        page_txt = oni.screenshot_and_read()
        raw_text.append(page_txt)
        # quick compress w/ Oni NLP
        synopsis = oni.nlp_module.generate(f"Summarize:\n{page_txt[:2000]}")
        synopses.append(synopsis)
        oni.browser.back()

    big_summary = oni.nlp_module.generate(
        "Combine the following synopses into a neutral, fact‑checked abstract:\n"
        + "\n---\n".join(synopses)
    )

    # guard‑rail: compassion + emotion check
    if not oni.validate_response(big_summary):
        big_summary = oni.fallback_with_coder(big_summary)

    oni.close_browser()
    return big_summary

###if not wanting to use a fresh "blank slate" import weights or memory into the script using the other functionalities. 
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

print(research_and_summarize(oni, "emergent AGI regulatory frameworks"))
