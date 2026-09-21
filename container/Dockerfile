FROM us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest
WORKDIR /

# Copies the trainer code to the docker image.
COPY trainer /trainer

RUN pip install scikit-learn google-cloud-bigquery joblib pandas google-cloud-storage db-dtypes

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.train"]