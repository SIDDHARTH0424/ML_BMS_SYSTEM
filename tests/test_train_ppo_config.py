from agents import train_ppo


def test_build_agent_uses_constant_configured_learning_rate(monkeypatch):
    captured = {}

    class DummyPPO:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(train_ppo, "PPO", DummyPPO)
    cfg = {
        "policy": "MlpPolicy", "learning_rate": 0.0003, "n_steps": 8,
        "batch_size": 4, "n_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95,
        "clip_range": 0.2, "ent_coef": 0.01, "vf_coef": 0.5,
        "max_grad_norm": 0.5, "policy_kwargs": {"net_arch": [8, 8]},
        "target_kl": 0.01, "seed": 7,
    }
    model = train_ppo.build_agent(None, cfg, "runs/test_tensorboard")
    assert model is not None
    assert captured["learning_rate"] == 0.0003
    assert not callable(captured["learning_rate"])
