# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Resolve a GitLab token ONCE for a whole react_api_web batch.

run_agent.sh calls this in its preamble and exports the result as GITLAB_TOKEN,
so every per-task run_react_api_web_agent.py inherits it and hits the env
fast-path in _resolve_gitlab_token — instead of each of ~48 tasks re-minting a
fresh PAT via Playwright (get_glpat has no caching and leaves a token behind on
every call).

Reuses run_react_api_web_agent._resolve_gitlab_token so the env-check +
/api/v4/user verify + get_glpat mint logic lives in exactly one place. Prints
ONLY the token to stdout (login/warning noise is redirected to stderr) so the
caller can capture it with $(...). Exits non-zero with nothing on stdout when no
token can be resolved, which drives run_agent.sh's up-front abort.
"""

import contextlib
import os
import sys

import click


@click.command()
@click.option("--gitlab-url", required=True, help="Base URL of the GitLab instance")
@click.option("--pte-dir", required=True, help="Path to the PTE project root")
def main(gitlab_url, pte_dir):
    sys.path.insert(0, str(pte_dir))  # so _resolve_gitlab_token can import get_glpat
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_react_api_web_agent import _resolve_gitlab_token

    # get_glpat prints "Log in successful" (and requests emits a urllib3 warning)
    # to stdout — keep stdout clean so $(...) captures only the token.
    with contextlib.redirect_stdout(sys.stderr):
        token = _resolve_gitlab_token(gitlab_url)

    sys.stdout.write(token)


if __name__ == "__main__":
    main()
