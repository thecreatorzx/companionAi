# from transformers import pipeline
# sentiment_pipeline = pipeline("sentiment-analysis")
# data = ["I love you", "I hate you","I wanna kiss you"]
# print(sentiment_pipeline(data))

## Model Load Time: 3.01 seconds
## Prediction Time: 0.11 seconds
## Total Execution Time: 3.12 seconds

import time
from transformers import pipeline

start_time = time.time()  # Start timing

classifier = pipeline("text-classification", model='distilbert-base-uncased-finetuned-sst-2-english', return_all_scores=True)

load_time = time.time()  # Capture time after model loading
text = "I love using transformers. The best part is wide range of support and its easy to use"

prediction = classifier(text)

end_time = time.time()  # Capture time after prediction

# Print the time taken
print(f"Model Load Time: {load_time - start_time:.2f} seconds")
print(f"Prediction Time: {end_time - load_time:.2f} seconds")
print(f"Total Execution Time: {end_time - start_time:.2f} seconds")

print(prediction)
