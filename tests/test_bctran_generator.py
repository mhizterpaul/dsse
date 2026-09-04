import pytest
from src.transient.models import TransformerSpec, TransformerWinding, ShortCircuitTest
from src.transient.bctran_generator import BCTRANGenerator


@pytest.fixture
def sample_transformer_spec():
    return TransformerSpec(
        name="test_tx",
        frequency_hz=50.0,
        windings=[
            TransformerWinding("HV", 33.0, 7.5, "Delta", 0.0),
            TransformerWinding("LV", 0.415, 7.5, "Wye", -30.0),
        ],
        short_circuit_tests=[
            ShortCircuitTest(1, 2, z_pos_pu=0.05, losses_pos_kw=10.0, z_zero_pu=0.05, losses_zero_kw=10.0)
        ],
        excitation_current_percent=0.5,
        excitation_loss_kw=25.0,
    )


def test_bctran_supporting_deck_generation(sample_transformer_spec):
    generator = BCTRANGenerator()
    deck = generator.generate_supporting_deck(sample_transformer_spec)
    assert "BEGIN NEW DATA CASE" in deck
    assert "ACCESS MODULE BCTRAN" in deck
    assert "$ERASE" in deck
    assert "$PUNCH" in deck


def test_bctran_punch_extraction_under_wine(sample_transformer_spec):
    generator = BCTRANGenerator()
    punch = generator.generate(sample_transformer_spec)
    assert "$VINTAGE, 1," in punch or "$VINTAGE,1," in punch
    assert "USE RL" in punch
    assert "USE OLD" in punch
