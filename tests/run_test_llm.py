#!/usr/bin/env python3
"""
LLM-only Test Runner
Runs problems directly through the LLM without any solver infrastructure.
"""

import asyncio
import glob
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

from src.mcp_solver.client.llm_factory import LLMFactory
from src.mcp_solver.client.token_callback import TokenUsageCallbackHandler
from src.mcp_solver.client.token_counter import TokenCounter
from tests.run_test import format_box, print_summary, read_problem_file, save_text_files
from tests.test_config import DEFAULT_TIMEOUT

DEFAULT_MODEL = "AT:claude-sonnet-4-20250514"

DIRECT_SOLVE_PROMPT = """\
You are solving a constraint satisfaction/optimization problem directly.

Your task:
1. Read the problem statement carefully.
2. Reason through the constraints and variables.
3. Find a solution, or determine it is unsatisfiable.
4. Report the solution clearly.

Format your response with these exact section markers:

=== MODEL ===
<your reasoning or model here>
=== END MODEL ===

=== SOLUTION ===
<the concrete solution values, or UNSAT if the problem is unsatisfiable>
=== END SOLUTION ===
"""

REVIEW_PROMPT = """\
You are reviewing a solution to a constraint satisfaction/optimization problem.

PROBLEM:
{problem}

MODEL/REASONING:
{model}

SOLUTION:
{solution}

Assess whether the solution is correct given the problem constraints.
Reply with <verdict>correct</verdict> or <verdict>incorrect</verdict>.
"""


def extract_sections(response_text):
    """Extract MODEL and SOLUTION sections from LLM response."""
    model_match = re.search(r"=== MODEL ===(.*?)=== END MODEL ===", response_text, re.DOTALL)
    solution_match = re.search(
        r"=== SOLUTION ===(.*?)=== END SOLUTION ===", response_text, re.DOTALL
    )
    model = model_match.group(1).strip() if model_match else response_text
    solution = solution_match.group(1).strip() if solution_match else response_text
    return model, solution


async def call_reviewer(problem, model, solution, llm_code, token_counter):
    """Call the reviewer LLM to check whether the solution is correct."""
    review_prompt = REVIEW_PROMPT.format(
        problem=problem, model=model, solution=solution
    )

    llm = LLMFactory.create_model(llm_code)
    callback = TokenUsageCallbackHandler(token_counter, agent_type="reviewer")
    config = RunnableConfig(callbacks=[callback])
    response = await llm.ainvoke([HumanMessage(content=review_prompt)], config=config)
    review_text = response.content

    verdict_match = re.search(r"<verdict>(correct|incorrect|unknown)</verdict>", review_text)
    verdict = verdict_match.group(1) if verdict_match else "unknown"
    return verdict, review_text


async def run_test_llm(
    problem_file,
    verbose=False,
    timeout=DEFAULT_TIMEOUT,
    save_results=False,
    llm_code=DEFAULT_MODEL,
):
    """Run a single problem directly through the LLM."""
    problem_name = os.path.basename(problem_file).replace(".md", "")
    print(format_box(f"Testing problem [LLM]: {problem_name}"))

    problem_content = read_problem_file(problem_file)
    start_time = datetime.now()
    token_counter = TokenCounter()

    try:
        llm = LLMFactory.create_model(llm_code)
        messages = [
            SystemMessage(content=DIRECT_SOLVE_PROMPT),
            HumanMessage(content=problem_content),
        ]

        print(f"Calling LLM ({llm_code}) directly...")
        main_callback = TokenUsageCallbackHandler(token_counter, agent_type="main")
        main_config = RunnableConfig(callbacks=[main_callback])
        response = await asyncio.wait_for(
            llm.ainvoke(messages, config=main_config), timeout=timeout
        )
        agent_response = response.content

        print(format_box("SOLUTION RESULT"))
        print(agent_response)

        extracted_model, extracted_solution = extract_sections(agent_response)

        print(format_box("REVIEW RESULT"))
        verdict, review_text = await call_reviewer(
            problem_content, extracted_model, extracted_solution, llm_code, token_counter
        )
        print(review_text)
        print(f"\nVerdict: {verdict}")

        token_counter.print_stats()

        duration = (datetime.now() - start_time).total_seconds()
        print(f"\nCompleted in {duration:.1f}s")

        if save_results:
            output_dir = os.path.join(os.path.dirname(__file__), "results", "llm")
            save_text_files(output_dir, problem_name, extracted_model, agent_response, ".txt")

        return verdict == "correct", Counter()

    except asyncio.TimeoutError:
        print(f"\nTest timed out after {timeout}s: {problem_name}")
        return False, Counter()
    except Exception as e:
        print(f"\nError testing {problem_name}: {e}")
        return False, Counter()


def main():
    parser = argparse.ArgumentParser(
        description="Run problems directly through the LLM"
    )
    parser.add_argument("--problem", help="Path to specific problem file (.md)")
    parser.add_argument("--folder", help="Folder containing .md problem files")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout in seconds per problem (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument("--save", "-s", action="store_true", help="Save response files")
    parser.add_argument(
        "--mc",
        default=DEFAULT_MODEL,
        help=f"Model code (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    if args.problem:
        if not os.path.exists(args.problem):
            print(f"Error: Problem file not found at '{args.problem}'")
            return 1
        problem_files = [args.problem]
    elif args.folder:
        problem_files = sorted(glob.glob(os.path.join(args.folder, "*.md")))
    else:
        print("Error: either --problem or --folder must be specified")
        return 1

    if not problem_files:
        print("Error: No problem files found")
        return 1

    print(f"Found {len(problem_files)} problem(s) to test")

    success_count = 0
    failed_tests = []
    all_tool_calls = Counter()

    for problem_file in sorted(problem_files):
        success, tool_counts = asyncio.run(
            run_test_llm(
                problem_file,
                verbose=args.verbose,
                timeout=args.timeout,
                save_results=args.save,
                llm_code=args.mc,
            )
        )
        if success:
            success_count += 1
        else:
            failed_tests.append(os.path.basename(problem_file))
        all_tool_calls.update(tool_counts)

    return print_summary(
        problem_files, success_count, failed_tests, all_tool_calls, "LLM"
    )


if __name__ == "__main__":
    sys.exit(main())
