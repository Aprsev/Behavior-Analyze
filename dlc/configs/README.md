# Configuration organization

Shared configuration schemas belong in this directory. Workflow-specific
examples stay beside the code that consumes them:

- `dlc/hybrid/config.hybrid.example.json`
- `dlc/legacy/config.example.json`

The GUIs create machine-local runtime configuration beside those examples; the
runtime files are ignored by Git.
