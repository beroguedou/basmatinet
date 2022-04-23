# Variables
PATH_TO_DATASET := "/path/to/rice_image_dataset/"

# Build the training conda environment
condaenv:
	conda create --name basmatienv --file conda-env.yaml python=3.8.12
	conda activate basmatienv
# Clean the conda environment
clean:
	conda env remove -n basmatienv
# Train a model
.PHONY: train
train:
	python src/train.py ${PATH_TO_DATASET} \
                     --batch-size 16 --nb-epochs 200 \
                     --workers 8 --early-stopping 5  \
                     --percentage 0.1 --cuda

# Build and serve the model
serve:
	docker build -t basmatinet app/. && \
	docker run -d -p 5001:5000 basmatinet
# Make a prediction with a sample of image in the folder images
predict:
	python app/frontend.py --filename "./images/arborio.jpg" --host-ip "0.0.0.0"
# Create the namespaces if they aren't.
gkenvs:
	kubectl apply -f app/k8s/namespace.yaml
# Create  deployment, service for test ong GKE
deploy-test:
	kubectl apply -f app/k8s/basmatinet-deployment.yaml --namespace=mlops-test && \
	kubectl apply -f app/k8s/basmatinet-service.yaml --namespace=mlops-test
# See the service that we create
get-test-svc:
	kubectl get services -n mlops-test
