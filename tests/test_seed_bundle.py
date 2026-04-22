from urbanair.utils.seeding import create_seed_bundle, derive_seed


def test_seed_bundle_is_deterministic() -> None:
    first = create_seed_bundle(101)
    second = create_seed_bundle(101)

    assert first.seed == second.seed == 101
    assert [first.rng.randint(0, 100) for _ in range(5)] == [second.rng.randint(0, 100) for _ in range(5)]


def test_derive_seed_changes_with_offset() -> None:
    assert derive_seed(11, 1) != derive_seed(11, 2)
    assert derive_seed(11, 1) == derive_seed(11, 1)
