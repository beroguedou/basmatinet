# Basmatinet

Ce projet est un projet pédagogique pour aider toute personne désireuse à apprendre les principaux composants pour entraîner un système de Deep Learning destiné à aller en production se basant sur un jeu de données de détection de différentes variétés de Riz. Alors vous êtes plutôt "Nigerian Jollof Or Senegalese Jollof ?? ". 

<p align="center">
  <img src="./app/arborio.jpg" height="25%" width="50%">
</p>

Ce projet traite:

- [x] L'entrainement d'un modèle de Deep Learning avec Pytorch.
- [x] Transfert learning à partir d'Efficient Net.
- [x] Sauvegarde du modèle entraîné.
- [x] Rest Api avec Flask pour servir le modèle entraîné.
- [x] Encryptage des données par un client et décryptage dans le serveur pour assurer la confidentialité.
- [x] Conteneurisation de l'application Flask en microservices Docker.
- [ ] Orchestration du service de prédiction avec Kubernetes (k8S) sur Google Cloud.
- [ ] Sauvegarder des images et de leur prédictions dans une base de données PostgreSQL.
- [ ] Tests Unitaires avec Pytest (Fixtures et Mocks y sont utilisés).
- [ ] Logging pendant l'entraînement et les prédictions
- [ ] Détection de Drift sur les images en entrées et sur les prédictions en sortie du système. 
- [ ] Monitoring avec Grafana.


