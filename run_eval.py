#!/usr/bin/env python3
"""
mcp-solver test automation script

Usage:
    python run_tests.py --solver mzn --folder tests/problems/mzn
    python run_tests.py --solver pysat --folder tests/problems/pysat
    python run_tests.py --solver z3 --folder tests/problems/z3 --runner llm
    python run_tests.py --solver maxsat --folder tests/problems/maxsat --runner both
    python run_tests.py --solver asp --folder tests/problems/asp --timeout 600

Arguments:
    --solver    Solver mode to use. Required.
                Choices: mzn, pysat, maxsat, z3, asp

    --folder    Path to folder containing .md problem files. Required.
                Can be absolute or relative to the project root.

    --runner    Which test runner to use (default: both).
                Choices:
                  mcp  — uv run run-test <solver> --problem <file>
                  llm  — uv run python tests/run_test_llm.py <solver> --problem <file>
                  both — runs mcp first, then llm for every problem

    --timeout   Seconds to wait before killing a single test (default: 600).

    --stop-on-fail
                Stop the whole run as soon as one test fails.

    --verbose   Show full subprocess output even for passing tests.
                By default, output is shown only on failure.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_SOLVERS = {"mzn", "pysat", "maxsat", "z3", "asp"}
VALID_RUNNERS = {"mcp", "llm", "both"}

# Colour codes (disabled automatically if not a TTY)
if sys.stdout.isatty():
    C_RED    = "\033[0;31m"
    C_GREEN  = "\033[0;32m"
    C_YELLOW = "\033[1;33m"
    C_BLUE   = "\033[0;34m"
    C_CYAN   = "\033[0;36m"
    C_BOLD   = "\033[1m"
    C_RESET  = "\033[0m"
else:
    C_RED = C_GREEN = C_YELLOW = C_BLUE = C_CYAN = C_BOLD = C_RESET = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def info(msg: str)    -> None: print(f"{C_BLUE}[INFO]{C_RESET}  {msg}")
def ok(msg: str)      -> None: print(f"{C_GREEN}[PASS]{C_RESET}  {msg}")
def warn(msg: str)    -> None: print(f"{C_YELLOW}[WARN]{C_RESET}  {msg}")
def fail(msg: str)    -> None: print(f"{C_RED}[FAIL]{C_RESET}  {msg}")
def header(msg: str)  -> None: print(f"\n{C_BOLD}{C_CYAN}{msg}{C_RESET}")


# ---------------------------------------------------------------------------
# Token / verdict data structures
# ---------------------------------------------------------------------------
@dataclass
class TokenStats:
    """Holds token usage parsed from subprocess output."""
    # LLM-runner style (per-agent rows)
    react_input:    Optional[int] = None
    react_output:   Optional[int] = None
    react_total:    Optional[int] = None
    reviewer_input: Optional[int] = None
    reviewer_output: Optional[int] = None
    reviewer_total: Optional[int] = None
    combined_total: Optional[int] = None
    # MCP-runner style (single combined figure)
    mcp_combined_total: Optional[int] = None

    def has_data(self) -> bool:
        return any(
            v is not None for v in [
                self.react_total, self.reviewer_total,
                self.combined_total, self.mcp_combined_total,
            ]
        )


@dataclass
class TestResult:
    label: str
    problem: str        # filename stem
    runner_tag: str     # "MCP" or "LLM"
    solver: str
    passed: bool        # exit-code == 0
    elapsed: float
    verdict: Optional[str] = None    # "correct" / "incorrect" / None
    tokens: TokenStats = field(default_factory=TokenStats)
    timed_out: bool = False


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _parse_k(value: str) -> Optional[int]:
    """Convert strings like '138k', '2.4k', '893', '1.1k' to ints."""
    value = value.strip()
    if not value or value in ("-", ""):
        return None
    try:
        if value.lower().endswith("k"):
            return int(float(value[:-1]) * 1000)
        return int(value.replace(",", ""))
    except ValueError:
        return None


def parse_verdict(output: str) -> Optional[str]:
    """
    Extract verdict from subprocess output.
    Looks for:
      <verdict>correct</verdict>   or   Verdict: correct/incorrect
      or the reviewer line: "Review complete: verdict is 'correct'"
    """
    # XML tag style (LLM runner)
    m = re.search(r"<verdict>(correct|incorrect)</verdict>", output, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # Plain "Verdict: ..." line
    m = re.search(r"(?i)^verdict:\s*(correct|incorrect)", output, re.MULTILINE)
    if m:
        return m.group(1).lower()
    # MCP runner reviewer log line
    m = re.search(r"verdict is '(correct|incorrect)'", output, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # Reviewer summary line: "Correctness: ✅ correct"
    m = re.search(r"(?i)correctness:\s*[✅❌]?\s*(correct|incorrect)", output)
    if m:
        return m.group(1).lower()
    return None


def parse_token_stats(output: str) -> TokenStats:
    """
    Parse token usage tables from subprocess output.

    
    """
    ts = TokenStats()

    # ---- LLM-runner table (4 data columns) ---------------------------------
    # Match table rows like: │ ReAct Agent │  859 │  1.1k │  2.0k │ Exact │
    # Must be line-anchored so newlines don't bleed into the groups.
    llm_row = re.compile(
        r"^│\s*(?P<agent>[^│\n]+?)\s*│\s*(?P<inp>[^│\n]+?)\s*│\s*(?P<out>[^│\n]+?)\s*│\s*(?P<tot>[^│\n]+?)\s*│",
        re.MULTILINE,
    )
    for m in llm_row.finditer(output):
        agent = m.group("agent").strip().lower()
        inp   = _parse_k(m.group("inp"))
        out   = _parse_k(m.group("out"))
        tot   = _parse_k(m.group("tot"))
        if "react" in agent:
            ts.react_input, ts.react_output, ts.react_total = inp, out, tot
        elif "reviewer" in agent:
            ts.reviewer_input, ts.reviewer_output, ts.reviewer_total = inp, out, tot
        elif "combined" in agent:
            ts.combined_total = tot

    # ---- MCP-runner table (3 data columns) ---------------------------------
    # Match rows like: │ Token │ ReAct Agent Input (exact) │ 138k │
    mcp_row = re.compile(
        r"│\s*Token\s*│\s*(?P<item>[^│]+?)\s*│\s*(?P<val>[^│]+?)\s*│"
    )
    for m in mcp_row.finditer(output):
        item = m.group("item").strip().lower()
        val  = _parse_k(m.group("val"))
        if "react agent input" in item:
            ts.react_input = val
        elif "react agent output" in item:
            ts.react_output = val
        elif "react agent total" in item:
            ts.react_total = val
        elif "reviewer input" in item:
            ts.reviewer_input = val
        elif "reviewer output" in item:
            ts.reviewer_output = val
        elif "reviewer total" in item:
            ts.reviewer_total = val
        elif "combined total" in item:
            ts.mcp_combined_total = val

    # Fallback: bare "COMBINED TOTAL" row in MCP table
    if ts.mcp_combined_total is None:
        m = re.search(r"│\s*COMBINED TOTAL\s*│\s*([^│]+?)\s*│", output)
        if m:
            ts.mcp_combined_total = _parse_k(m.group(1))

    return ts


# ---------------------------------------------------------------------------
# Core helpers 
# ---------------------------------------------------------------------------
def find_problems(folder: Path) -> list[Path]:
    """Return all .md files in *folder* (non-recursive), sorted by name."""
    if not folder.exists():
        sys.exit(f"Folder not found: {folder}")
    if not folder.is_dir():
        sys.exit(f"Not a directory: {folder}")
    problems = sorted(folder.glob("*.md"))
    if not problems:
        warn(f"No .md files found in {folder}")
    return problems


def build_commands(solver: str, problem: Path, runner: str) -> list[list[str]]:
    cmds = []
    if runner in ("mcp", "both"):
        cmds.append(["uv", "run", "run-test", solver, "--problem", str(problem)])
    if runner in ("llm", "both"):
        cmds.append(["uv", "run", "python", "tests/run_test_llm.py", solver,
                     "--problem", str(problem)])
    return cmds


def run_command(
    cmd: list[str],
    timeout: int,
    verbose: bool,
) -> tuple[bool, str, str, bool]:
    """
    Run *cmd* with a *timeout*.
    Returns (passed, label, combined_output, timed_out).
    """
    label = " ".join(cmd)
    combined_output = ""
    timed_out = False
    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        combined_output = (result.stdout or "") + "\n" + (result.stderr or "")
        if verbose or result.returncode != 0:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        return result.returncode == 0, label, combined_output, False

    except subprocess.TimeoutExpired:
        warn(f"Timed out after {timeout}s")
        timed_out = True
        return False, f"TIMEOUT: {label}", "", True

    except FileNotFoundError as exc:
        fail(f"Command not found: {exc}")
        return False, f"NOT FOUND: {label}", "", False


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------
def _fmt_tokens(val: Optional[int]) -> str:
    if val is None:
        return "—"
    if val >= 1000:
        return f"{val/1000:.1f}k"
    return str(val)


def _verdict_symbol(verdict: Optional[str], passed: bool) -> str:
    if verdict == "correct":
        return f"{C_GREEN}✓ correct{C_RESET}"
    if verdict == "incorrect":
        return f"{C_RED}✗ incorrect{C_RESET}"
    # No verdict parsed — fall back to exit code
    if passed:
        return f"{C_GREEN}✓ pass{C_RESET}"
    return f"{C_RED}✗ fail{C_RESET}"


def print_summary_table(results: list[TestResult]) -> None:
    """Print a per-test summary with verdict + token usage."""
    if not results:
        return

    header("=" * 60)
    header("  Per-Test Summary")
    header("=" * 60)

    # Column widths
    col_problem = max(len(r.problem) for r in results)
    col_problem = max(col_problem, 7)  # "Problem"

    # Header row
    row_fmt = (
        f"  {{:<{col_problem}}}  {{:<5}}  {{:<14}}  "
        f"{{:>7}}  {{:>7}}  {{:>7}}  {{:>7}}  {{:>8}}"
    )
    divider = "  " + "-" * (col_problem + 5 + 14 + 7*4 + 8 + 7*2 + 10)

    print(row_fmt.format(
        "Problem", "Run", "Verdict",
        "ReactIn", "ReactOut", "RevIn", "RevOut", "Combined",
    ))
    print(divider)

    for r in results:
        t = r.tokens
        combined = t.combined_total or t.mcp_combined_total

        # Strip colour for width calculation in the format string
        verdict_raw = (
            "✓ correct"   if r.verdict == "correct" else
            "✗ incorrect" if r.verdict == "incorrect" else
            ("✓ pass" if r.passed else "✗ fail")
        )
        verdict_col = _verdict_symbol(r.verdict, r.passed)
        # Pad the coloured string to fixed display width
        pad = 14 - len(verdict_raw)
        verdict_col_padded = verdict_col + " " * max(0, pad)

        print(
            f"  {r.problem:<{col_problem}}  "
            f"{r.runner_tag:<5}  "
            f"{verdict_col_padded}  "
            f"{_fmt_tokens(t.react_input):>7}  "
            f"{_fmt_tokens(t.react_output):>7}  "
            f"{_fmt_tokens(t.reviewer_input):>7}  "
            f"{_fmt_tokens(t.reviewer_output):>7}  "
            f"{_fmt_tokens(combined):>8}"
        )

    print(divider)

    # Aggregate token totals
    total_combined = sum(
        (r.tokens.combined_total or r.tokens.mcp_combined_total or 0)
        for r in results
    )
    correct_count = sum(1 for r in results if r.verdict == "correct")
    incorrect_count = sum(1 for r in results if r.verdict == "incorrect")
    unknown_count = len(results) - correct_count - incorrect_count

    print(f"\n  Verdict breakdown:")
    print(f"    {C_GREEN}Correct  :{C_RESET}  {correct_count}")
    print(f"    {C_RED}Incorrect:{C_RESET}  {incorrect_count}")
    if unknown_count:
        print(f"    {C_YELLOW}Unknown  :{C_RESET}  {unknown_count}  (no verdict tag found in output)")
    print(f"\n  Total tokens consumed across all tests: {_fmt_tokens(total_combined)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_tests.py",
        description="Run mcp-solver tests for all .md problem files in a folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--solver", required=True,
        choices=sorted(VALID_SOLVERS),
        metavar="SOLVER",
        help=f"Solver mode to use. Choices: {', '.join(sorted(VALID_SOLVERS))}",
    )
    parser.add_argument(
        "--folder", required=True,
        type=Path,
        metavar="PATH",
        help="Folder containing .md problem files.",
    )
    parser.add_argument(
        "--runner", default="both",
        choices=sorted(VALID_RUNNERS),
        metavar="RUNNER",
        help="Test runner(s) to use: mcp, llm, or both (default: both).",
    )
    parser.add_argument(
        "--timeout", default=600, type=int,
        metavar="SECONDS",
        help="Per-test timeout in seconds (default: 600).",
    )
    parser.add_argument(
        "--stop-on-fail", action="store_true",
        help="Abort the run as soon as one test fails.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Always stream subprocess output (default: only on failure).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve folder relative to the script's directory (project root)
    project_root = Path(__file__).parent.resolve()
    folder = args.folder if args.folder.is_absolute() else project_root / args.folder

    problems = find_problems(folder)

    # -----------------------------------------------------------------------
    # Print run config
    # -----------------------------------------------------------------------
    header("=" * 60)
    header("  mcp-solver Test Runner")
    header("=" * 60)
    info(f"Solver:   {args.solver}")
    info(f"Folder:   {folder}")
    info(f"Runner:   {args.runner}")
    info(f"Problems: {len(problems)}")
    info(f"Timeout:  {args.timeout}s per test")
    info(f"Started:  {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # -----------------------------------------------------------------------
    # Run tests
    # -----------------------------------------------------------------------
    total = 0
    passed = 0
    failed_tests: list[str] = []
    all_results: list[TestResult] = []
    suite_start = time.time()

    for problem in problems:
        cmds = build_commands(args.solver, problem, args.runner)
        for cmd in cmds:
            runner_tag = "MCP" if "run-test" in cmd else "LLM"
            total += 1
            label = f"{args.solver}/{problem.name} ({runner_tag})"

            header(f"[{total}] {label}")
            if args.verbose:
                info(f"Command: {' '.join(cmd)}")

            t_start = time.time()
            success_flag, _, output, timed_out = run_command(
                cmd, timeout=args.timeout, verbose=args.verbose
            )
            elapsed = time.time() - t_start

            # Parse verdict and token stats from combined output
            verdict = parse_verdict(output)
            tokens  = parse_token_stats(output)

            result = TestResult(
                label=label,
                problem=problem.stem,
                runner_tag=runner_tag,
                solver=args.solver,
                passed=success_flag,
                elapsed=elapsed,
                verdict=verdict,
                tokens=tokens,
                timed_out=timed_out,
            )
            all_results.append(result)

            # Per-test inline summary
            verdict_str = (
                f"  verdict={verdict}" if verdict else "  verdict=unknown"
            )
            combined_tok = tokens.combined_total or tokens.mcp_combined_total
            tok_str = (
                f"  tokens={_fmt_tokens(combined_tok)}" if combined_tok else ""
            )

            if success_flag:
                ok(f"{label}  ({elapsed:.1f}s){verdict_str}{tok_str}")
                passed += 1
            else:
                fail(f"{label}  ({elapsed:.1f}s){verdict_str}{tok_str}")
                failed_tests.append(label)
                if args.stop_on_fail:
                    warn("--stop-on-fail is set, aborting.")
                    break

        if args.stop_on_fail and failed_tests:
            break

    # -----------------------------------------------------------------------
    # Original summary
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - suite_start
    failed_count = len(failed_tests)

    header("=" * 60)
    header("  Results")
    header("=" * 60)
    print(f"  {C_BOLD}Total  :{C_RESET}  {total}")
    print(f"  {C_GREEN}Passed :{C_RESET}  {passed}")
    print(f"  {C_RED}Failed :{C_RESET}  {failed_count}")
    print(f"  Elapsed: {total_elapsed:.1f}s")

    if failed_tests:
        print(f"\n{C_RED}Failed tests:{C_RESET}")
        for t in failed_tests:
            print(f"  {C_RED}✗{C_RESET}  {t}")

    # -----------------------------------------------------------------------
    # NEW: per-test summary table with verdicts + token stats
    # -----------------------------------------------------------------------
    print_summary_table(all_results)

    print()
    if failed_tests:
        sys.exit(1)
    else:
        ok("All tests passed ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
