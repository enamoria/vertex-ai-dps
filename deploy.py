from google.cloud import aiplatform

model = aiplatform.Model.upload(
    display_name="mpg-imported",
    artifact_uri="gs://io-vertex-codelab/mpg-model/",
    serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-3:latest"
)

# Deploy to an endpoint
endpoint = model.deploy(
    machine_type = "n1-standard-4"
)