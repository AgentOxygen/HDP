# Contributing to the HDP

Thank you for your interest in improving the Heatwave Diagnostics Package! Contributions of all kinds are welcome, including bug reports, feature suggestions, documentation improvements, and code.

## Reporting bugs and asking questions

Please report bugs, ask questions, and make suggestions through the [GitHub Issues form](https://github.com/AgentOxygen/HDP/issues). When reporting a bug, it helps to include:

- A short description of what you expected to happen and what actually happened.
- A minimal code snippet that reproduces the problem.
- Your operating system, Python version, and HDP version (`pip show hdp-python`).

## Setting up a development environment

The full instructions live in the [Developer's Guide](https://hdp.readthedocs.io/en/latest/dev_guide.html). In short:

```bash
git clone git@github.com:AgentOxygen/HDP.git
cd HDP
pip install -e .
```

HDP requires Python >= 3.12.3. Core dependencies are declared in [pyproject.toml](pyproject.toml).

## Running the tests

All tests are written with [pytest](https://docs.pytest.org/en/stable/) and live in the `hdp/tests/` directory. The recommended way to run them is inside the project's Docker container, which mirrors the environment used in continuous integration:

```bash
docker build --rm -t hdp .
docker run -v .:/project -it hdp
```

Tests also run automatically via [GitHub Actions](https://github.com/AgentOxygen/HDP/actions) on every push. Please make sure the suite passes locally before opening a pull request, and add tests covering any new behavior.

## Submitting changes

1. Fork the repository and create a branch for your change.
2. Make your changes, keeping them focused and consistent with the surrounding code style.
3. Add or update tests and documentation as needed.
4. Ensure the test suite passes.
5. Open a pull request against the `main` branch with a clear description of the change and the motivation behind it.

## License

By contributing to the HDP, you agree that your contributions will be licensed under the [MIT License](LICENSE) that covers the project.
