"""Unit tests for the LLM judge and evaluation runner."""

from src.eval.llm_judge import LLMJudge


def test_judge_faithfulness_refusal():
    judge = LLMJudge()
    eval_res = judge.evaluate_faithfulness(
        query="What is Cassandra replication?",
        context="PostgreSQL 16 is used for relational persistence.",
        answer="Based on the provided documentation, there is insufficient information to answer this question.",
    )
    assert eval_res.score == 1.0
    assert eval_res.grounded_claims >= 1


def test_judge_faithfulness_heuristic():
    judge = LLMJudge()
    context = "Redis 7.2 Cluster has 6 horizontally sharded master nodes with lookup latencies of 1-3ms."
    grounded_answer = "The Redis cluster has 6 horizontally sharded master nodes."
    ungrounded_answer = "MongoDB Atlas uses 3 replica sets across Azure us-east regions."

    grounded_eval = judge.evaluate_faithfulness("How many nodes in Redis?", context, grounded_answer)
    ungrounded_eval = judge.evaluate_faithfulness("How many nodes in Redis?", context, ungrounded_answer)

    assert grounded_eval.score >= 0.8
    assert ungrounded_eval.score <= 0.3


def test_judge_correctness_refusal():
    judge = LLMJudge()
    eval_res = judge.evaluate_correctness(
        query="What is the Cassandra replication?",
        answer="Based on the provided documentation, there is insufficient information to answer this question.",
        ground_truth="Based on the provided documentation, there is insufficient information to answer this question.",
    )
    assert eval_res.score == 1.0
    assert eval_res.rubric_grade == 5


def test_judge_correctness_heuristic():
    judge = LLMJudge()
    gt = "Distributed rate limiting is enforced via token bucket algorithms backed by Redis."
    correct_ans = "The gateway uses a token bucket algorithm backed by Redis for rate limiting."
    wrong_ans = "Rate limiting uses fixed window counters in memory."

    high_eval = judge.evaluate_correctness("Rate limit algorithm?", correct_ans, gt)
    low_eval = judge.evaluate_correctness("Rate limit algorithm?", wrong_ans, gt)

    assert high_eval.rubric_grade >= 4
    assert low_eval.rubric_grade <= 2
