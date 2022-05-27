
![Maintainer](https://img.shields.io/badge/maintainer-Béranger%20GUEDOU-lightgreen)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/green?icon=github)](https://github.com/Naereen/badges/)




# Basmatinet

Welcome to this project folks !

Whether you like it or not this project is all about riiiiice or riz in french. It is also about Deep Learning and MLOPS. So if you want to learn to train and deploy a simple model to recognize rice type basing on a photo, then you are at the right place.

<p align="center">
  <img src="./images/logo.jpg" height="50%" width="75%">
</p>



## 0- Project's Roadmap

This project will consist to:

- [x] Train a Deep Learning model with Pytorch.
- [x] Transfert learning from Efficient Net.
- [x] Data augmentation with Albumentation.
- [x] Save trained model with early stopping.
- [x] Track the training with MLFLOW.
- [x] Serve the model with a Rest Api built with Flask.
- [x] Encode data in base64 client side before sending to the api server.
- [x] Package the application in microservice's fashion with Docker.
- [x] Yaml for configurations file.
- [x] Passing arguments anywhere it is possible.
- [x] Orchestrate the prediction service with Kubernetes (k8s) on Google Cloud Platform.
- [x] Pre-commit git hook.
- [x] Logging during training.
- [x] Makefile to facilate some operations.
- [x] Unitary tests with Pytest (Fixtures and Mocks).
- [x] CI/CD pipeline with github actions.
- [ ] Make docstring and typing.




## 1- Download the dataset
The dataset is the Rice Image dataset that we can find on: https://www.muratkoklu.com/datasets/ . It regroups 05 classes of images data that can be used for classification. After downloading it  unzip and get it ready by ensuring that you have the following aborescence. Be sure you save the PATH-TO-DATASET

```bash
Rice_Image_Dataset/
          ├── Arborio
          ├── Basmati
          ├── Ipsala
          ├── Jasmine
          └── Karacadag
```
## 2- Install project's dependencies and packages
1st step: Install

This project was developped in conda environment so if you have conda installed, just use the following command to create the basmatienv with all requirements installed.

```bash
# Create the environment from the configuration
$ conda create --name basmatienv --file conda-env.yaml python=3.8.12
# Activate it
$ conda activate basmatienv
```


2nd step: Clean the conda environment
```bash
$ conda env remove -n basmatienv

```
## 3- Train a basmatinet model
1st step: Train the model
```bash
$ make train PATH_TO_DATASET=[PATH-TO-DATASET] BATCH_SIZE=16
```

2nd step: See the model's tracking metrics.
Open a new terminal:
```bash
# Activate the conda environnement
$ conda activate basmatienv
# Set up MLFlow's server
$ make tracking
```
The training process tracks the metrics with MLFlow tracking. To monitore it the model metrics, go to the url:

## 4- Dockerize the model and push the Docker Image to Google Container Registry

1st step: Let's build a docker images

```bash
# Move into the app directory
$ make serve
# Try an inference to test the endpoint locally
$ make predict FILENAME="./images/arborio.jpg" HOST_IP=[EXTERNAL-IP]
```

2nd step: Let's push the docker image into a Google Container Registry. But you should create a google cloud project to have PROJECT-ID and in this case you HOSTNAME will be "gcr.io" and you should enable GCR Api on google cloud platform.

```bash
# Re-tag the image and push to container registry
$ make image-push HOSTNAME=[HOSTNAME] PROJECT_ID=[PROJECT-ID]
```

## 5- Create a kubernetes cluster
First of all you should enable GKE Api on google cloud platform. And go to the cloud shell or stay on your host if you have gcloud binary already installed.

```bash
# Start a cluster and connect to it
$ make k8s-cluster PROJECT_ID=[PROJECT-ID]

```

## 6- Deploy the application on Kubernetes (Google Kubernetes Engine)
Create the deployement and the service on a kubernetes cluster.

```bash
# Create namespaces on GKE for dev, staging and production environment
$ make gkenvs

# Deploy to Google Cloud Platform
$ make deploy-test

# Check that everything is alright with the following command and look for basmatinet-app in the output
$ make get-test-svc

# The output should look like
NAME             TYPE           CLUSTER-IP    EXTERNAL-IP     PORT(S)          AGE
basmatinet-app   LoadBalancer   xx.xx.xx.xx   xx.xx.xx.xx   5001:xxxx/TCP      2m3s
```
Take the EXTERNAL-IP and test your service with the file frontend.py . Then you can cook your jollof with some basmatinet!!!

```bash
$ make predict FILENAME="./images/arborio.jpg" HOST_IP=[EXTERNAL-IP]
```

## 7- Clean the conda environnement
If you want to delete the conda environment use the following command:

```bash
$ make clean
```

## 8 - Prerequisites for the CI/CD with github actions.
These steps are very important and you won't be able to push or pull request on main branch if there are not realised. You need to perform them in a Google Cloud Console.

### A - Create a GKE cluster
As you want to deploy your deploy your project to a Google Kubernetes Engine, you need to have one. we choose a simple zonal cluster here. You will also need to create 03 variables here:

- $GKE_PROJECT: your GCP project id
- $GKE_CLUSTER: the name you want for your GKE cluster
- $GKE_ZONE: the zone were you want your cluster

```bash
$ gcloud container clusters create $GKE_CLUSTER \
	--project=$GKE_PROJECT \
	--zone=$GKE_ZONE
```

### B - Enabling useful APIs
Enable container registry to be able to push your docker image in GCR. And enable k8s to deploy container based application.

```bash
$ gcloud services enable \
	containerregistry.googleapis.com \
	container.googleapis.com
```

### C - Configure a service account 
Create a variable named $SA_NAME for your service account. And copy its email to create a new variable $SA_EMAIL.

```bash
# Create service account
$ gcloud iam service-accounts create $SA_NAME
# verify if the service account is created and copy its email
$ gcloud iam service-accounts list
```

### D - Give the service accounts necessary permissions
As you want to connect programmatically with your service account, it needs some permissions defined by some roles. You can give more restrictive roles to respect least privileges principle.

```bash
$ gcloud projects add-iam-policy-binding $GKE_PROJECT \
	--member=serviceAccount:$SA_EMAIL \
	--role=roles/container.admin
$ gcloud projects add-iam-policy-binding $GKE_PROJECT \
	--member=serviceAccount:$SA_EMAIL \
	--role=roles/storage.admin
$ gcloud projects add-iam-policy-binding $GKE_PROJECT \
	--member=serviceAccount:$SA_EMAIL \
	--role=roles/container.clusterViewer
```

### E - Download the service account key and encode it like a secret

```bash
# Create a service account
$ gcloud iam service-accounts keys create key.json --iam-account=$SA_EMAIL
# Encode the service account key
$ export GKE_SA_KEY=$(cat key.json | base64)
# Show the value of the encoded key 
$ echo $GKE_SA_KEY
```

### F - Store the service account encoded key 
Copy the encoded key and store as a secret named GKE_SA_KEY it your github settings secrets interface. 

### G - Store your GCP project name
Store the name of your project as a secret named GKE_PROJECT in your github settings secrets interface.
