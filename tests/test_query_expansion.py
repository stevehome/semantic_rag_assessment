from src.query_expansion import MockGenerativeModel, QueryExpander


def test_mock_generative_model_returns_text_attr():
    model = MockGenerativeModel()
    resp = model.generate_content("Query: How does the system handle peak load?")
    assert hasattr(resp, "text") and isinstance(resp.text, str)
    assert resp.text  # non-empty


def test_mock_generative_model_candidates_surface_matches_vertex():
    """Vertex's GenerationResponse exposes `.candidates[i].content.parts[j].text`."""
    model = MockGenerativeModel()
    resp = model.generate_content("Query: peak load")
    cand = resp.candidates[0]
    assert cand.content.parts[0].text == resp.text


def test_expansion_adds_domain_synonyms_for_known_triggers():
    expander = QueryExpander()
    out = expander.expand("How does the system handle peak load?")
    lowered = out.lower()
    # At least one of the seeded synonyms should appear.
    assert any(token in lowered for token in
               ("traffic spikes", "autoscaling", "high concurrency", "capacity"))


def test_expansion_is_deterministic():
    expander = QueryExpander()
    assert expander.expand("cache strategy?") == expander.expand("cache strategy?")


def test_expansion_strips_question_framing_when_no_trigger():
    expander = QueryExpander()
    out = expander.expand("What is the vector index choice?")
    assert "?" not in out
    assert "what is" not in out.lower()
