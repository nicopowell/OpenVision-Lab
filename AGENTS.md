# OpenVision Lab

## Project Context

OpenVision Lab is a desktop playground for learning and experimenting with
Computer Vision. The initial workflow is:

1. Load an image.
2. Apply image-processing operations.
3. Display the result.

The initial product is intentionally small: load one image, convert it to
grayscale, and display both the original and processed images.

The current project state and milestone are documented in `ROADMAP.md`.

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

## Documentation and Comments

This project is a learning tool, so documentation should help a student follow,
understand, and review the code.

- Add brief comments when they clarify non-obvious concepts or important
  decisions. Explain what a piece of code does and, especially, why it is
  implemented that way.
- Avoid comments that merely restate what the code already says.
- Give functions and methods a short docstring when their purpose, parameters,
  return value, exceptions, or contract are not obvious from the signature and
  the code.
- Do not document functions or methods whose name, parameters, and behavior are
  already clear on their own. Add documentation only when it provides
  information that is not evident from the signature and the code.
- Explain important parameters, especially when they have restrictions, units,
  formats, ranges, or effects that are not obvious from the name.
- Document relevant data types and shapes when useful for learning, such as
  shapes, dtypes, image channels, and conventions like BGR/RGB.
- Explain non-obvious design decisions close to the code or in the appropriate
  documentation.
- When a function can raise exceptions that matter to its callers, briefly
  document the conditions that trigger them.
- Keep documentation concise. The goal is to help a student follow and review
  the code, not to document every line or write excessively long explanations.

## Language

The project uses English for source code, comments, documentation, user-facing
text, identifiers, and commit messages.

Communicate with the project owner in Spanish unless requested otherwise.
