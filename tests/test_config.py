from tidalnet.config import load_config

def test_config_inheritance():
    c=load_config("configs/smoke.yaml")
    assert c["training"]["learning_rate"]==0.0005
    assert c["training"]["epochs"]==1
