import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/tf2-preview/nnlm-es-dim50-with-normalization/1")

def get_vectors(sentences):
    embeddings = embed(sentences)
    return embeddings
if __name__ == "__main__":
    embeddings = embed(["gato", "gato y perro"])
    print(embeddings)
