# Contribution Guide

First off, thank you for considering contributing to openDVP! It's people like you that make open source so great. We welcome contributions of all kinds, from bug reports to new features.

This guide will walk you through the process of setting up your development environment and submitting your first contribution.

## Development workflow summary

1. Fork the opendvp repository to your own GitHub account
2. Create a development environment
3. Create a new branch for your PR
4. Add your feature or bugfix to the codebase
5. Make sure all tests are passing
6. Ensure code style, and/or ensure documentation looks good.
7. Open a PR back to the main repository

## Step 1: Fork the opendvp repository to your own GitHub account

First, [fork the repository](https://github.com/CosciaLab/openDVP/fork) to your own GitHub account.  
For more context about what forking means check: [Github Manual: Forking](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)

## Step 2: Create a development environment

Clone your fork to your local machine:

```bash
git clone https://github.com/YOUR_USERNAME/openDVP.git
cd openDVP

# Add the main repository as a remote
git remote add upstream https://github.com/CosciaLab/opendvp.git
# this allows you to easily update your opendvp copy if new changes come to the main opendvp
```

Now you must setup a tool that will check your code when you commit it. The tool is called `pre-commit`, briefly, `pre-commit` will run a check based on rules found in `.pre-commit-config.yaml`.

```bash
# run to install pre-commit
uv run pre-commit install
```

## Step 3: Create a new branch for your PR

All development should occur in branches of your forked opendvp,  each branch dedicated to a particular purpose: feature, bug, etc.  
You can create a branch with:

```bash
$ git checkout main                 # Starting from the main branch
$ git pull                          # Syncing with the repo
$ git switch -c {your-branch-name}  # Making and changing to the new branch
```

<details>
<summary>Detailed command explanation</summary>

The provided commands guide you through the process of creating a new branch to work on your feature or bug fix.  

`$ git checkout main`  
This command changes your current location in the repository to the main branch. You're making sure you start from the most up-to-date, stable version of the code.

`$ git pull`  
This command downloads the latest changes from the remote repository to your local copy. This is a crucial step to ensure your local main branch is synchronized with the project's main branch before you start your work. This prevents merge conflicts later.

`$ git switch -c {your-branch-name}`  
This is a shorthand command that does two things:
switch -c: creates a new branch with the name you provide (in this case, {your-branch-name}).
switch: immediately changes to that new branch.
</details>

## Step 4: Add your feature or bugfix to the codebase

Code away!

## Step 5: Make sure all tests are passing

Testing is very important. To ensure that all the functions are working, and our users won't be surprised by a nasty bug we program these tests. Look inside `opendvp/tests/` to see how it looks. We use [pytest](https://docs.pytest.org/en/stable/) to test opendvp.

After you have added your magnificient code, you should also add some tests in their respective directories. If you have never written tests before, don't worry. I suggest you look at the other tests, look at the ideas in the dropdown, and perhaps ask your trusty LLM how to get started.

<details>
<summary> What to test in a new feature </summary>

- If you’re not sure what to tests about your function, some ideas include:
- Are there arguments which conflict with each other? Check that if they are both passed, the function throws an error (see pytest.raises docs).
- Are there input values which should cause your function to error?
- Did you add a helpful error message that recommends better outputs? Check that that error message is actually thrown.
- Can you place bounds on the values returned by your function?
- Are there different input values which should generate equivalent output (e.g. if an array is sparse or dense)?
- Do you have arguments which should have orthogonal effects on the output? Check that they are independent. For example, if there is a flag for extended output, the base output should remain the same either way.
- Are you optimizing a method? Check that it’s results are the same as a gold standard implementation.

</details>

Before pushing your efforts online you should run all tests, for quick function feedback you can run pytest for a single function:

```bash
# this runs all tests
uv run pytest 

# this runs tests for a particular function
uv run pytest tests/io/test_DIANN_to_adata.py
```

## Step 6: Code style check, documentation check

Since you already installed `pre-commit` every time you have committed it should have ran `ruff`.
We use `ruff` for code formatting and linting to ensure a consistent code style throughout the project.

### Manual checks

You can also run the formatter and linter manually:

To format your code:
```bash
uv run ruff format .
```

To check for linting errors:
```bash
uv run ruff check .
```

The settings for these checks live inside `pyproject.toml`.

### Building the Documentation

Our documentation is built using [Sphinx](https://www.sphinx-doc.org/en/master/) and is located in the `docs/` directory.

To build the documentation locally run:

```bash
uv run --group docs sphinx-build -b html docs/ docs/_build/html/
```

The generated HTML files will be in the `docs/_build/html` directory. You can open `index.html` in your browser to view the documentation.

## Step 7: Open a PR back to the main repository

Once you've made your changes and are happy with them, you're ready to submit a pull request.

As of now, Jose will perform all the version bumping and releasing, do not worry about this.

Go to your github branch website and click the big green button `Compare & Pull Request`

Ensure that the **pull request**  is from your fork to the `main` branch of the `CosciaLab/openDVP` repository.

In your pull request description, please explain the changes you've made and why you've made them. If your pull request addresses an open issue, please link to it. Once you've submitted your pull request, our continuous integration (CI) system will automatically run the tests to make sure everything is working as expected. We will then review your contribution and provide feedback.

Thank you for contributing to openDVP!!
