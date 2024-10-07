import pytest
from app.ml.feature_extraction import FeatureExtractor
from app.exceptions import FeatureExtractionError

@pytest.fixture
def feature_extractor():
    return FeatureExtractor()

def test_extract_features(feature_extractor):
    url = "https://www.example.com"
    try:
        features = feature_extractor.extract_features(url)
        assert isinstance(features, dict)
        assert len(features) == 21  # Assuming 21 features are extracted
    except FeatureExtractionError as e:
        pytest.fail(f"Feature extraction failed: {str(e)}")

def test_extract_features_with_ip(feature_extractor):
    url = "http://192.168.1.1"
    features = feature_extractor.extract_features(url)
    assert features["use_of_ip"] == 1

def test_extract_features_with_shortening_service(feature_extractor):
    url = "http://bit.ly/example"
    features = feature_extractor.extract_features(url)
    assert features["short_url"] == 1

def test_extract_features_with_suspicious_word(feature_extractor):
    url = "http://example.com/login"
    features = feature_extractor.extract_features(url)
    assert features["sus_url"] == 1