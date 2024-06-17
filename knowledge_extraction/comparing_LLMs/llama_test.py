from transformers import pipeline

sentences = ['Kant was a remarkable figure',
             'Water is made of hydrogen and oxygen',
             'The capital of Norway is Oslo']


'''# Sentiment analysis
classifier = pipeline('sentiment-analysis')
for sentence in sentences:
    print(sentence, classifier(sentence))

# Zero-shot classification
classifier = pipeline('zero-shot-classification')
for sentence in sentences:
    print(sentence, classifier(sentence, candidate_labels=['education', 'business', 'science', 'politics']))'''

# Generation
generator = pipeline('text-generation', model='meta-llama/Meta-Llama-3-8B')
generator('I am better. I am a God. You are merely')