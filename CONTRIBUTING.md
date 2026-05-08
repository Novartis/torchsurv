# Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

## Getting Started

Before contributing, please ensure you can set up the development environment by following the instructions in our [developer notes](/docs/devnotes.md).

The quickest way to get started is with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Novartis/torchsurv.git
cd torchsurv
uv sync --group dev       # creates .venv and installs everything
uv run pre-commit install # set up pre-commit hooks
uv run pytest             # run the test suite
```

See the [developer notes](/docs/devnotes.md) for more details.

## How to Contribute

### Reporting Issues

If you encounter bugs, have feature requests, or want to suggest improvements:

- Create an [issue](https://github.com/Novartis/torchsurv/issues) in the GitHub repository
- Provide detailed information about the problem
- Include reproducible code examples when applicable

### Contributing Code

1. **Fork** the repository
2. **Create** a new branch for your changes
3. **Make** your changes following our coding conventions
4. **Add** tests for any new functionality
5. **Submit** a pull request with a clear description of your changes

## License

By contributing to this project, you agree to license your contributions under the same license as this project.

## Questions?

If you have questions about contributing, feel free to open an issue for discussion.
