from settings import Settings


def test_settings():
    settings = Settings()
    assert settings.ENVIRONMENT == "test"
    assert settings.APP_NAME == "mlops-lab01-test"
    assert settings.API_KEY == "test-api-key"
