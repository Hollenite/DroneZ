from __future__ import annotations

from urbanair.env.environment import DroneZEnvironment
from urbanair.training.action_format import build_action_prompt, build_candidate_actions, compact_observation_summary, parse_llm_action


def test_candidate_actions_are_valid_for_initial_easy_observation() -> None:
    env = DroneZEnvironment()
    observation, _ = env.reset("easy")

    candidates = build_candidate_actions(observation)

    assert candidates
    for candidate in candidates:
        routed = env.router.route(env.state, candidate)
        assert routed.is_valid, routed.error_message


def test_parser_handles_markdown_and_extra_text() -> None:
    env = DroneZEnvironment()
    observation, _ = env.reset("easy")
    candidate = build_candidate_actions(observation)[0]
    text = f"```json\n{candidate}\n```\nThis is the safest action."

    result = parse_llm_action(text, observation=observation, candidate_actions=[candidate])

    assert result.valid_json
    assert result.valid_action_shape
    assert result.action == candidate


def test_parser_repairs_action_name_alias_and_missing_params() -> None:
    env = DroneZEnvironment()
    observation, _ = env.reset("easy")
    candidate = build_candidate_actions(observation)[0]

    result = parse_llm_action(
        '{"action_name": "%s", "params": {}}' % candidate["action"],
        observation=observation,
        candidate_actions=[candidate],
    )

    assert result.valid_action_shape
    assert result.repaired
    assert result.action == candidate


def test_candidate_choice_mode_returns_numbered_action() -> None:
    env = DroneZEnvironment()
    observation, _ = env.reset("demo")
    candidates = build_candidate_actions(observation)

    result = parse_llm_action('{"choice": 1}', observation=observation, candidate_actions=candidates)

    assert result.valid_json
    assert result.used_candidate_choice
    assert result.action == candidates[0]


def test_action_prompt_is_compact_and_includes_candidates() -> None:
    env = DroneZEnvironment()
    observation, _ = env.reset("easy")

    prompt = build_action_prompt(observation, candidate_choice=True)

    assert "VALID_CANDIDATES" in prompt
    assert '{"choice": 1}' in prompt
    assert len(compact_observation_summary(observation)) <= len(observation["summary"])
