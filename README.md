# P8 - Segmentation sémantique d'images routières - Dataset Cityscapes - projet véhicule autonome

Done using : Python, Tensorflow/Keras, Numpy, OpenCV, skimage - IoU and composite metrics and losses from segmentation_models library<br><br>

Original video after resizing | Segmented video 
:-: | :-:
<video src='https://user-images.githubusercontent.com/77936631/154296329-89562978-40cf-46c3-b86b-16784f7a9c69.mp4'/> | <video src='https://user-images.githubusercontent.com/77936631/154296537-c23b082b-2a3a-493b-95cf-83e2e556de0b.mp4'/>

<br>

- Résumé de l'activité dans la présentation ppt
- Recherche documentaire
- Comparaison de métriques et architectures
- Essais d'architectures alternatives (les modèles retenus pour la comparaison sont intégrés aux scripts d'entraînement)
- Comparatif et optimisation des apports de différents procédés d'augmentation de données visuelles
- Traitement de vidéos routières
- Développement et déploiement d'une API et d'une webapp via Flask
- Explicatif des modèles testés, de la démarche suivie et des résultats dans la note technique au format word

Le notebook de contrôle permet de réaliser un entraînement et la sauvegarde d'un modèle dans le cloud AzureML.<br>
Le notebook API Client montre comment utiliser les routes définies dans l'app Flask.<br>
Les fichiers d'entraînement partagés illustrent les types d'architectures et d'augmentation de données mis en oeuvre
