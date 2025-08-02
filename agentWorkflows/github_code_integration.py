from pathlib import Path
import subprocess


def github_code_integrator(oni: OniMicro, repo_url: str, commit_message: str = "Add note from Oni agent"):
    """
    – Clone a GitHub repository
    – Generate a short summary of the README
    – Create a note file and commit it back to the repo
    """
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    if Path(repo_name).exists():
        subprocess.run(["git", "-C", repo_name, "pull"], check=True)
    else:
        subprocess.run(["git", "clone", repo_url], check=True)

    readme_path = Path(repo_name) / "README.md"
    if readme_path.exists():
        readme_text = readme_path.read_text()
        summary = oni.nlp_module.generate(f"Summarize this project:\n{readme_text[:2000]}")
    else:
        summary = "README not found"

    note_file = Path(repo_name) / "ONI_NOTE.txt"
    note_file.write_text("Edited by Oni agent\n")
    subprocess.run(["git", "add", note_file.name], cwd=repo_name, check=True)
    subprocess.run(["git", "commit", "-m", commit_message], cwd=repo_name, check=True)
    return summary


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

print(github_code_integrator(oni, "https://github.com/example/repo.git"))
