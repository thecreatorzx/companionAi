import time
import rasa.shared.nlu.training_data.message
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.model import get_model
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.registry import component_registry

# Start timing model loading
start_load_time = time.time()

# Load the Rasa NLU model (Change the path to your trained model)
model_path = "models"
model = get_model(model_path)
loaded_model = DefaultV1Recipe.load(model)

load_time = time.time()
print(f"Model Load Time: {load_time - start_load_time:.2f} seconds")

# Test the model with a sample query
test_text = "What is the weather like today?"
start_pred_time = time.time()

# Parse user input
message = Message.build(text=test_text)
result = loaded_model.process([message])

end_pred_time = time.time()

# Print timing results
print(f"Prediction Time: {end_pred_time - start_pred_time:.2f} seconds")
print(f"Total Execution Time: {end_pred_time - start_load_time:.2f} seconds")

# Print model response
print(result)
