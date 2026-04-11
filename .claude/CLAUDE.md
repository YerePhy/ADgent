# Design & architecture
- Suggest options that account for scalability and relevant software/system design patterns
- Be critic and honest — if a pattern or concept smells, say it explicitly rather than being compliant

# Code style
- Do not make very large files — raise a warning when a module starts growing past 500 lines and suggest a split strategy
- All docstrings must follow [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

# Safety
- Raise a warning when a change might introduce substantial latency or a security vulnerability (e.g. injection, exposed secrets, unvalidated input at system boundaries)

# Workflow
- Suggest using /clear when a task is complete to save tokens
