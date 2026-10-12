# Agent instructions

## Architecture

This repository contains independent implementations of foundational AI architectures. Each subdirectory (such as `RAG/` or `ADK/`) is a self-contained module with its own code, documentation, and tests.

## Environment isolation

- Never initialize or use a global virtual environment in the repository root.
- Initialize virtual environments locally inside each module directory (`<module>/.venv`).
- Use `uv` or `python -m venv` within the target module directory.
- Run commands and test runners through the module-local virtual environment, for example `cd <module> && uv run <command>` or `<module>/.venv/bin/python <command>`.
- Do not install dependencies across module boundaries.

## Development workflow

- Read the target module's `FIELD_GUIDE.md` before making changes. It defines architectural invariants, state transitions, and milestone specifications.
- Rely on the standard library where the module specification requires it.
- Run the module test suite before completing any task.
