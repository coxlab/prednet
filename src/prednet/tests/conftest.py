import tensorflow.compat.v1

@pytest.fixture(scope="session", autouse=True)
def disable_v2_behavior():
    tensorflow.compat.v1.disable_v2_behavior()

