# Variables
PATH_TO_DATASET := "/path/to/rice_image_dataset/"
HOST_IP := "0.0.0.0"
FILENAME := "./images/arborio.jpg"
PROJECT_ID := ""
HOSTNAME := ""
BATCH_SIZE := 16

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
                     --batch-size ${BATCH_SIZE} --nb-epochs 200 \
                     --workers 8 --early-stopping 5  \
                     --percentage 0.1 --cuda

# Build and serve the model
serve:
	docker build -t basmatinet app/. && \
	docker run -d -p 5001:5000 basmatinet
# Make a prediction with a sample of image in the folder images
predict:
	python app/frontend.py --filename ${FILENAME} --host-ip ${HOST_IP}
#
image-push:
	docker tag basmatinet ${HOSTNAME}/${PROJECT_ID}/basmatinet && \
	docker push ${HOSTNAME}/${PROJECT_ID}/basmatinet
# Make a K8s cluster on GKE and connect to it
k8s-cluster:
	gcloud container clusters create k8s-gke-cluster --num-nodes 3 \
					--machine-type g1-small --zone europe-west1-b && \
	gcloud container clusters get-credentials k8s-gke-cluster \
					--zone us-west1-b --project ${PROJECT_ID}
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
