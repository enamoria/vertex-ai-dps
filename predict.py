from google.cloud import aiplatform

endpoint = aiplatform.Endpoint(
    endpoint_name="projects/551577981756/locations/us-central1/endpoints/9061251246362460160"
)

# A test example we'll send to our model for prediction
# test_mpg = [1.4838871833555929,
#  1.8659883497083019,
#  2.234620276849616,
#  1.0187816540094903,
#  -2.530890710602246,
#  -1.6046416850441676,
#  -0.4651483719733302,
#  -0.4952254087173721,
#  0.7746763768735953]
test_mpg =  [1, 2, 3, 2, -2, -1, -2, -1, 0]

response = endpoint.predict([test_mpg])

print('API response: ', response)

print('Predicted MPG: ', response.predictions[0][0])