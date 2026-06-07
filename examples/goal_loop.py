"""
Example: goal loop with a judge.

A worker attempts the goal, a judge scores the attempt against acceptance
criteria, and the loop repeats with feedback until the judge accepts or the
budget runs out. When the budget runs out, the status is "exhausted", never a
false "done". Runs with stub functions and no API keys.
"""

from agent_orchestrator import Verdict, run_goal

GOAL = "List three prime numbers greater than 10."


# Stub worker that improves across turns as feedback accumulates.
def worker(goal: str, feedback: str) -> str:
    attempts = {
        "": "11, 12, 13",  # 12 is not prime
        "12 is not prime": "11, 13, 15",  # 15 is not prime
        "15 is not prime": "11, 13, 17",  # correct
    }
    return attempts.get(feedback, "11, 13, 17")


def judge(goal: str, output: str) -> Verdict:
    numbers = [int(n) for n in output.replace(",", " ").split()]
    bad = [n for n in numbers if n <= 10 or any(n % d == 0 for d in range(2, n))]
    if bad:
        return Verdict(done=False, feedback=f"{bad[0]} is not prime")
    return Verdict(done=True)


print("=== Goal loop: converges ===")
result = run_goal(GOAL, worker, judge, max_turns=5)
print(f"  status: {result.status}")
print(f"  turns:  {result.turns}")
print(f"  output: {result.output}")
print()

print("=== Goal loop: budget exhausted (never a false done) ===")
stubborn = run_goal(
    GOAL,
    worker=lambda goal, feedback: "11, 12, 13",  # always wrong
    judge=judge,
    max_turns=3,
)
print(f"  status: {stubborn.status}")
print(f"  turns:  {stubborn.turns}")
