# -*- coding: utf-8 -*-
"""
Created on Mon May  8 13:22:15 2023

@author: dylan
"""

import csv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import lil_matrix


def lire_matrice_utilite_depuis_csv(chemin_csv):
    """
    Cette fonction lit le fichier csv donne par le chemin et le nom du fichier
    et le convertit en objet Pandas

    Parameters
    ----------
    chemin_dossier : chemin vers les fichiers CSV contenant les notes a lire

    Returns
    -------
    matrice_utilite : matrice d'utilite au format optimise (scipy.sparse)
    liste_livres_significatifs : liste des livres identifies
    liste_utilisateurs_unique : liste des utilisateurs identifies
    statistiques : dictionnaire de statistiques sur les donnees lues

    """
   
    with open(chemin_csv, newline = "") as fichier_csv:

       # Extraction du contenu du CSV
       contenu_csv = csv.reader(fichier_csv, delimiter=";")
       
       # On prepare des listes pour remplir la matrice d'utilite
       liste_livres_note_non_nulle = []
       liste_utilisateurs_note_non_nulle = []
       liste_valeurs_note_non_nulle = []
       
       # Dicitonnaire pour les decomptes du nombre de note pour chaque livre
       decompte = {}
       
       # On parcourt le fichier pour recuperer les differentes notes
       for (i_ligne, ligne) in enumerate(contenu_csv):
           if (i_ligne == 0):
               print("Lecture du fichier {}...".format(chemin_csv))
           else:
               # Lecture d'une ligne de donnee, on sauvegarde
               # utilisateur, livre, note (de type str)
               utilisateur, isbn, note = ligne
               
               # Si la note est non-nulle (vraie evaluation)
               # On stocke util/livre/note dans des listes
               if note != "0":
                   liste_livres_note_non_nulle.append(isbn)
                   liste_utilisateurs_note_non_nulle.append(utilisateur)
                   liste_valeurs_note_non_nulle.append(int(note))
                   
                   # Mise a jour du decompte de notes pour le livre
                   if isbn in decompte:
                       # Si le decompte a deja ete cree, on incremente
                       decompte[isbn] += 1
                   else:
                       # Sinon, on initialise ce decompte a 1
                       decompte[isbn] = 1
       
    # On liste les utilisateurs et les livres en vue de les compter 
    liste_livres_unique = np.unique(liste_livres_note_non_nulle).tolist()
    
    # On compte le nombre de notes par livre, en vue de conserver
    # uniquement les livres ayant suffisament de notes
    liste_livres_significatifs = []
    
    # Pour chaque livre
    for i_livre, livre in enumerate(liste_livres_unique):
        
        # On compte le nombre de notes associees au livre courant
        nombre_notes_livre = decompte[livre]
        
        # Si le livre a au moins 10 notes, on le conserve
        if nombre_notes_livre > 10:
            liste_livres_significatifs.append(livre)
 
    # Apres avoir identifie les livres avec le nombre minimal de notes,
    # on retire les notes inexploitables des donnees lues
    liste_livres_note_non_nulle_copie = liste_livres_note_non_nulle.copy()
    liste_utilisateurs_note_non_nulle_significatif = []
    liste_livres_note_non_nulle_significatif = []
    liste_valeurs_note_non_nulle_significatif = []
    
    for (i_livre, livre) in enumerate(liste_livres_note_non_nulle_copie):
        # Si le livre a un nombre de note significatif, on conserve 
        # son couple utilisateur/evaluation
        if liste_livres_significatifs.count(livre) != 0:
            liste_utilisateurs_note_non_nulle_significatif.append(liste_utilisateurs_note_non_nulle[i_livre])
            liste_livres_note_non_nulle_significatif.append(liste_livres_note_non_nulle[i_livre])
            liste_valeurs_note_non_nulle_significatif.append(liste_valeurs_note_non_nulle[i_livre])
    
    # On liste les utilisateurs
    liste_utilisateurs_unique = np.unique(liste_utilisateurs_note_non_nulle_significatif).tolist()
    
    # On compte le nombre d'utilisateurs et de livres
    # apres suppression des notes nulles et des livres pas assez evalues
    nombre_utilisateurs = len(liste_utilisateurs_unique)
    nombre_livres = len(liste_livres_significatifs)
    
    # On construit une matrice vide initialement, au format sparse
    matrice_utilite = lil_matrix((nombre_livres, nombre_utilisateurs), dtype=int)
    
    # On la remplit en parcourant les donnees lues
    n_valeurs_significatif = len(liste_valeurs_note_non_nulle_significatif)
    for (i_note, note) in enumerate(liste_valeurs_note_non_nulle_significatif):
        
        # Affichage de l'avancement de la completion de la matrice
        if np.mod(i_note, 1000) == 0:
            print("chargement note {}/{}".format(i_note, n_valeurs_significatif))
        
        # On identifie ligne/colonne ou placer la note
        ligne_note = liste_livres_significatifs.index(liste_livres_note_non_nulle_significatif[i_note])
        colonne_note = liste_utilisateurs_unique.index(liste_utilisateurs_note_non_nulle_significatif[i_note])
        
        # On l'insere et on passe a la suivante
        matrice_utilite[ligne_note, colonne_note] = note
    
    # On prepare la matrice sparse pour la rendre utilisable par scikitlearn   
    matrice_utilite = matrice_utilite.tocsr()
    matrice_utilite.sort_indices()
    
    # Calcul de statistiques sur les donnees
    statistiques = {}
    statistiques["taux completude"] = (i_note+1) / (nombre_livres * nombre_utilisateurs) * 100
    statistiques["taux livres significatifs"] = len(liste_livres_significatifs) / len(liste_livres_unique) * 100
    statistiques["taux notes zero"] = (i_note+1) / (i_ligne+1) * 100
    
    return matrice_utilite, liste_livres_significatifs, liste_utilisateurs_unique, statistiques


def reduire_matrice_utilite_via_SVD(matrice_utilite_sparse):
    """
    Cette fonction permet de calculer la SVD tronquee de la matrice en entree
    sous la forme M = UxSxV'

    Parameters
    ----------
    matrice_utilite_sparse : matrice d'utilite au format scipy.sparse csr

    Returns
    -------
    matrice_reduite : resultat de la SVD tronquee (UxS)

    """
    
    # On realise la SVD tronquee sur une dimension d'arrivee egale
    # a la racine carree du nombre de lignes
    n_lignes = matrice_utilite_sparse.shape[0]
    svd = TruncatedSVD(n_components = int(np.sqrt(n_lignes)))

    # On extrait de la SVD tronquee les donnees reduites
    matrice_reduite = svd.fit_transform(matrice_utilite_sparse)
    
    return matrice_reduite


def estimer_similarites(matrice_reduite):

    """
    Cette fonction permet de calculer les similarites dans un ensemble 
    de vecteurs : similarite = A.B / ||A|| * ||B|| = cos(angle(A,B))

    Parameters
    ----------
    matrice_reduite : matrice resultant de la reduction via SVD tronquee

    Returns
    -------
    smilarite_consinus : matrice symetrique de similarites

    """    

    # On evalue la ressemblance entre les livres en dimension reduite
    similarite_cosinus = cosine_similarity(matrice_reduite)
    
    return similarite_cosinus


def lister_recommandations(liste_livres, similarites_livres, livre, nombre_recommandations):
    
    """
    Cette fonction permet d'effectuer plusieurs recommandations sur la base
    d'un livre de reference et des similarites calculees entre les livres

    Parameters
    ----------
    liste_livres : ensemble des livres disponibles
    similarites_livres : matrice de similarites entre les livres
    livre : livre de reference, sur lequel sont basees les recommandations
    nombre_recommandations : nombre de livres a recommander

    Returns
    -------
    liste_noms : liste des livres recommandes
    liste_proximites : liste des similarites associees a chaque recommandation

    """
    
    # On regarde pour le livre donne sa ressemblance par rapport aux autres
    i_livre = liste_livres.index(livre)
    similarites = similarites_livres[:,i_livre]
    # On force le terme du livre de reference a -1 pour l'exclure des recommandations
    similarites[i_livre] = -1  
    
    # On trie par ordre decroissant de ressemblance les nombres_recommandations
    # livres identifies : ressemblance = 1 : livre tres proche,
    # ressemblance = -1 livre tres eloigne
    similarites_copiees = np.copy(similarites)
    indices = similarites_copiees.argsort()[::-1][:nombre_recommandations]
    
    # Initialisation des listes
    liste_noms = []
    liste_proximites = []
    
    # Pour chaque recommandation, on recupere le titre et la ressemblance
    for r in range(nombre_recommandations):
        liste_noms.append(liste_livres[indices[r]])
        liste_proximites.append(similarites[indices[r]])
    
    return liste_noms, liste_proximites


def retourner_noms_livres(chemin_csv, liste_isbn):
    
    """
    Cette fonction permet de recuperer le titre de plusieurs livres
    sur la base de leurs identifiants isbn

    Parameters
    ----------
    chemin_csv : chemin vers le fichier de donnees isbn/livres
    liste_isbn : isbn des livres dont on souhaite recuperer les titres

    Returns
    -------
    liste_titres : liste des titres des livres

    """
    
    with open(chemin_csv, newline = "") as fichier_csv:

       # Extraction du contenu du CSV
       contenu_csv = csv.reader(fichier_csv, delimiter=";")
       
       # On initialise les titres sur les isbn
       titres = {}
       for isbn in liste_isbn:
           titres[isbn] = isbn
       
       # On parcourt le fichier pour recuperer les differents titres
       for (i_ligne, ligne) in enumerate(contenu_csv):
           if (i_ligne == 0):
               print("Lecture du fichier {}...".format(chemin_csv))
           else:
                # Lecture d'une ligne, seuls les deux premiers champs
                # sont d'interet
                isbn_lu, titre_lu = ligne[:2]
                
                # Si un des isbn d'interet apparait, on recupere le titre associe
                if isbn_lu in liste_isbn:
                    titres[isbn_lu] = titre_lu
       
       # On met les different titres lus dans une liste
       liste_titres = [titres[isbn] for isbn in liste_isbn]
        
    return liste_titres
 