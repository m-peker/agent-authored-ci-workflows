"""Per-file metadata used to check whether templates, copies or non-workflow files drive the agent-human comparison.

active    : the file is a workflow GitHub actually runs: `.github/workflows/<name>.yml|yaml` at the repository root
            (files in sub-directories, composite `action.yml` definitions and vendored copies never run).
template  : the file installs or configures a coding-agent product itself (e.g. Copilot's copilot-setup-steps.yml,
            the Claude Code / Codex / Gemini / Cursor GitHub integrations). Such files are mostly product templates.
purpose   : coarse workflow purpose from the file name, job names and used actions (first matching rule wins).
nhash     : hash of the normalised content (comments, blank lines and surrounding whitespace removed);
            identical nhash across repositories = copy / boilerplate.
"""
import hashlib
import re

ACTIVE = re.compile(r"^\.github/workflows/[^/]+\.ya?ml$", re.I)

TEMPLATE = re.compile(
    r"copilot-setup-steps|anthropics/claude-code-action|anthropics/claude-code-base-action|openai/codex-action|"
    r"google-github-actions/run-gemini-cli|cursor-agent|devin-ai|@claude\b|claude-code-review", re.I)

PURPOSE_RULES = [
    ("agent_integration", TEMPLATE),
    ("security_scan", re.compile(r"codeql|scorecard|zizmor|trivy|snyk|semgrep|gitleaks|dependency-review|"
                                 r"osv-scanner|bandit|security", re.I)),
    ("release_publish", re.compile(r"release|publish|npm publish|pypi|twine|goreleaser|changeset|semantic-release|"
                                   r"docker/build-push-action|upload-release|tag", re.I)),
    ("deploy", re.compile(r"deploy|pages|vercel|netlify|firebase|aws-actions|azure/|gcloud|kubectl|terraform|helm", re.I)),
    ("maintenance", re.compile(r"stale|label|welcome|greet|triage|auto-assign|lock|close|issue|dependabot|renovate|"
                               r"sync|cron|schedule", re.I)),
    ("docs", re.compile(r"\bdocs?\b|documentation|mkdocs|sphinx|docusaurus|typedoc|storybook", re.I)),
    ("lint_format", re.compile(r"lint|format|prettier|eslint|ruff|black|flake8|clippy|gofmt|pre-commit|typecheck|mypy",
                               re.I)),
    ("build_test", re.compile(r"test|build|\bci\b|check|pytest|jest|vitest|cargo|go test|npm (run )?test|e2e|coverage|"
                              r"matrix|compile", re.I)),
]
_COMMENT = re.compile(r"(^|\s)#.*$")


def active(filename):
    return bool(ACTIVE.match(filename))


def purpose(text, filename):
    hay = filename + "\n" + text
    if TEMPLATE.search(hay):
        return "agent_integration"
    head = filename + "\n" + "\n".join(l for l in text.splitlines()
                                       if re.match(r"\s*(name|uses|run|on|cron)\b", l))
    for name, rx in PURPOSE_RULES[1:]:
        if rx.search(head):
            return name
    return "other"


def nhash(text):
    lines = [_COMMENT.sub("", l).strip() for l in text.splitlines()]
    norm = "\n".join(l for l in lines if l)
    return hashlib.sha1(norm.encode("utf-8", "replace")).hexdigest()[:16]
