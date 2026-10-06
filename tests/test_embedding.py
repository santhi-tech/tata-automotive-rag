from retrieval.retriever import get_embedding

def test_embedding_length():
    emb = get_embedding("Electric cars are the future.")
    assert len(emb) == 768

def test_embedding_type():
    emb = get_embedding("Motorcycles are popular in Asia.")
    assert isinstance(emb, list)
    assert all(isinstance(x, float) for x in emb)
