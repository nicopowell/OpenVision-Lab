# OpenVision Lab

## Project Context

OpenVision Lab is a desktop playground for learning and experimenting with
Computer Vision. The initial workflow is:

1. Load an image.
2. Apply image-processing operations.
3. Display the result.

The initial product is intentionally small: load one image, convert it to
grayscale, and display both the original and processed images.

The current project state is documented in `ROADMAP.md`. The project is
currently at milestone M0.1: preparing the Python development environment.

## Stack

- Python
- PySide6 for the desktop interface
- OpenCV and NumPy for image processing
- pytest for processing logic tests when appropriate

Linux is the primary development platform. Keep the code portable to other
desktop platforms when doing so does not add unnecessary complexity.

## Development Workflow

- Inspect the relevant project files before making changes.
- Before a non-trivial change, explain the proposed plan and wait for approval
  when the scope or design has not already been agreed.
- Develop in small, functional, and reviewable increments.
- Keep the implementation aligned with the current milestone in `ROADMAP.md`.
- Do not implement future roadmap items prematurely.

## Development Principles

- Prefer the simplest solution that solves the current problem.
- Keep the application functional and understandable before making it extensible.
- Let the architecture evolve when the project creates a concrete need.
- Do not introduce plugins, factories, registries, dependency injection, or
  similar abstractions without a concrete current need.
- Explain important design decisions before implementing non-trivial changes.
- Keep image-processing logic separate from user-interface code.
- Introduce tests progressively, starting with deterministic processing logic
  when appropriate.
- Use Context7 when current or library-specific documentation is needed.
- Avoid premature optimization and speculative features.
- Preserve cross-platform compatibility when it does not complicate the design.

## Language

The project uses English for source code, comments, documentation, user-facing
text, identifiers, and commit messages.

Communicate with the project owner in Spanish unless requested otherwise.
