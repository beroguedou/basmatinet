.ONESHELL:
# Variables
# PATH_TO_DATASET := "/home/beranger/Downloads/Rice_Image_Dataset/"
PATH_TO_DATASET := "data-samples/"
HOST_IP := "0.0.0.0"
FILENAME := "./images/arborio.jpg"
PROJECT_ID := ""
HOSTNAME := ""
BATCH_SIZE := 16


# Train a model
.PHONY: train
train:
	python basmatinet/ml/train.py ${PATH_TO_DATASET} \
                     --batch-size ${BATCH_SIZE} --nb-epochs 200 \
                     --workers 8 --early-stopping 5  \
                     --percentage 0.1 --cuda
# See MLFlow tracking interface
tracking:
	mlflow ui
# Build and serve the model  docker build -t basmatinet basmatinet/app/. &&
build:
	docker build -t basmatinet --build-arg MODEL="basmatinet.pth" -f basmatinet/app/Dockerfile \
	--progress=plain --no-cache .
run:
	docker run -d -p 5001:5001 basmatinet
# Make a prediction with a sample of image in the folder images
predict:
	python basmatinet/app/frontend.py --filename ${FILENAME} --host-ip ${HOST_IP}

# Unitary test
unit-tests:

	pytest -s -q basmatinet/tests/unit_tests/app/test_api_utils.py --datapath none --disable-pytest-warnings
	pytest -s -q basmatinet/tests/unit_tests/app/test_api.py --datapath none --disable-pytest-warnings
	pytest -s -q basmatinet/tests/unit_tests/ml/test_engine.py --datapath ${PATH_TO_DATASET}  --disable-pytest-warnings
	pytest -s -q basmatinet/tests/unit_tests/ml/test_data.py --datapath ${PATH_TO_DATASET}  --disable-pytest-warnings
# Integration test
integration-tests:
	pytest -s -q basmatinet/tests/integration_tests/test_train.py --datapath ${PATH_TO_DATASET}  --disable-pytest-warnings
#
image-push:
	docker tag basmatinet ${HOSTNAME}/${PROJECT_ID}/basmatinet
	docker push ${HOSTNAME}/${PROJECT_ID}/basmatinet
# Make a K8s cluster on GKE and connect to it
k8s-cluster:
	gcloud container clusters create k8s-gke-cluster --num-nodes 3 \
					--machine-type g1-small --zone europe-west1-b
	gcloud container clusters get-credentials k8s-gke-cluster \
					--zone us-west1-b --project ${PROJECT_ID}
# Create the namespaces if they aren't.
gkenvs:
	kubectl apply -f basmatinet/app/k8s/namespace.yaml
# Create  deployment, service for test ong GKE
deploy-test:
	kubectl apply -f basmatinet/app/k8s/basmatinet-deployment.yaml --namespace=mlops-test
	kubectl apply -f basmatinet/app/k8s/basmatinet-service.yaml --namespace=mlops-test
# See the service that we create
get-test-svc:
	kubectl get services -n mlops-test
