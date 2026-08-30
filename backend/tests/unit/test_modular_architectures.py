# unit test for modular architectures and a-priori op validation
from app.nas.architectures.autoencoders import DenseAutoencoder, VariationalAutoencoder, Conv1DAutoencoder
from app.nas.architectures.routers import DenseRouter, Conv1DRouter
from app.nas.architectures.memory import LSTMBackbone

def test_dense_autoencoder_supported():
    enabled = {"FULLY_CONNECTED", "RELU", "SUB", "SQUARE", "MEAN"}
    assert DenseAutoencoder.is_supported(enabled) is True

def test_conv1d_autoencoder_unsupported_when_missing_conv():
    enabled = {"FULLY_CONNECTED", "RELU", "SUB", "SQUARE", "MEAN"}
    assert Conv1DAutoencoder.is_supported(enabled) is False

def test_lstm_backbone_requires_unidirectional_lstm():
    enabled_no_lstm = {"FULLY_CONNECTED", "RELU"}
    assert LSTMBackbone.is_supported(enabled_no_lstm) is False
    enabled_with_lstm = {"UNIDIRECTIONAL_SEQUENCE_LSTM", "FULLY_CONNECTED", "TANH", "LOGISTIC", "ADD", "MUL"}
    assert LSTMBackbone.is_supported(enabled_with_lstm) is True
