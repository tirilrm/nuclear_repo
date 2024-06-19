from transformers import pipeline

sentences = ['Kant was a remarkable figure',
             'Water is made of hydrogen and oxygen',
             'The capital of Sweden is Stockholm']

'''# Sentiment analysis
classifier = pipeline('sentiment-analysis')
for sentence in sentences:
    print(sentence, classifier(sentence))

# Zero-shot classification
classifier = pipeline('zero-shot-classification')
for sentence in sentences:
    print(sentence, classifier(sentence, candidate_labels=['education', 'business', 'science', 'politics']))'''

# Generation
generator = pipeline('text-generation', model='distilgpt2')
generator(
    'I am better. I am a god. You are merely', 
    max_length=150, 
    num_return_sequences=2
)