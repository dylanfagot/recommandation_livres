# -*- coding: utf-8 -*-
"""
Ce fichier permet de lancer l'analyse des CSV contenus dans un dossier
"Book reviews" contenu a la racine du projet, sur la base de fonctions
contenues dans le fichier fonctions_recommandation.py

@auteur: Dylan Fagot
"""

import fonctions_recommandation as f

if __name__ == "__main__" :
    
    print("Etape 1 : Construction de la matrice d'utilite")
    # Obtention de la matrice d'utilite depuis le fichier des notes
    matrice_utilite, liste_isbn, liste_utilisateurs, statistiques = f.lire_matrice_utilite_depuis_csv("Book reviews\\BX-Book-Ratings.csv")    
    
    print("Etape 2 : Correspondances isbn / titres des livre")
    # On recupere le titre de chaque livre si disponible (sinon on garde isbn)
    liste_livres = f.retourner_noms_livres("Book reviews\\BX_Books.csv", liste_isbn)
    
    print("Etape 3 : Décompositions en valeurs singulières")
    # SVD tronquee sur la matrice d'utilite
    matrice_reduite = f.reduire_matrice_utilite_via_SVD(matrice_utilite)
    
    print("Etape 4 : Calcul des similarites")
    # Calcul des similarites entre les livres
    similarites_livres = f.estimer_similarites(matrice_reduite)
    
    print("Etape 5 : Recommandation de livre")
    # Nombre de recommandations a faire
    n_recommandations = 3
    # Livre de reference
    livre = "The Hobbit : The Enchanting Prelude to The Lord of the Rings"
    # Recommandations sur la base des similarites entre livres
    liste_noms, liste_proximites = f.lister_recommandations(liste_livres, similarites_livres, livre, n_recommandations)
    # Affichage des resultats
    print("Recommandations de livres proches de {} :".format(livre))
    for i in range(n_recommandations):
        print("{}, score = {}".format(liste_noms[i], liste_proximites[i]))
        
    print("Etape 6 : analyse des valeurs singlières")
    f.analyser_valeurs_singulieres(matrice_reduite)
