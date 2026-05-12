from unittest.mock import MagicMock

import numpy as np

from src.embedding import LocalEmbedder, MockTextEmbedding, MockVertexEmbeddingModel


def test_mock_vertex_embedding_model_surface():
    """Mirror of:
        from vertexai.language_models import TextEmbeddingModel
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        embs  = model.get_embeddings(["text"])
        embs[0].values
    """
    fake_local = MagicMock(spec=LocalEmbedder)
    fake_local.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    model = MockVertexEmbeddingModel(embedder=fake_local)
    embs = model.get_embeddings(["hello"])

    assert isinstance(embs, list) and len(embs) == 1
    assert isinstance(embs[0], MockTextEmbedding)
    np.testing.assert_allclose(embs[0].values, [0.1, 0.2, 0.3], atol=1e-6)
    fake_local.encode.assert_called_once_with(["hello"])


def test_from_pretrained_constructs_model():
    model = MockVertexEmbeddingModel.from_pretrained("textembedding-gecko@003")
    assert isinstance(model, MockVertexEmbeddingModel)
