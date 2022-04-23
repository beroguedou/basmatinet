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
	kubectl apply -f app/k8s/basmatinet-deployment.yaml --namespace=mlops-test
	kubectl apply -f app/k8s/basmatinet-service.yaml --namespace=mlops-test
get-test-svc:
	kubectl get services -n mlops-test
