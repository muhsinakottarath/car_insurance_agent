from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder

# Test 1: chat model
generator = OllamaChatGenerator(model="llama3.2")
result = generator.run(messages=[ChatMessage.from_user("Say hello in one sentence.")])
print("Chat model:", result["replies"][0].text)

# Test 2: embedding model
embedder = OllamaTextEmbedder(model="nomic-embed-text")
result = embedder.run(text="Where is my parcel?")
print("Embedding length:", len(result["embedding"]))