# P8 - Segmentation sémantique d'images routières - Dataset Cityscapes - projet véhicule autonome

Done using : Python, Tensorflow/Keras, Numpy, OpenCV, skimage - IoU and composite metrics and losses from segmentation_models library<br><br>

Original video after resizing | Segmented video 
:-: | :-:
<video src='https://user-images.githubusercontent.com/77936631/154293201-5360b05d-1e6a-4a8f-a035-f78c211f39f8.mp4' width=100/> | <video src='https://user-images.githubusercontent.com/77936631/154293257-16f8b24d-307e-4461-86ad-6d4e8d5ed5df.mp4' width=100/>

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
