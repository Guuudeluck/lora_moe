"""
CLI entry point for llama-finetune package.
"""

from llama_finetune.workflows.finetune_pipeline import main as workflow_main


def main():
    """Main CLI entry point."""
    workflow_main()


if __name__ == "__main__":
    main()
