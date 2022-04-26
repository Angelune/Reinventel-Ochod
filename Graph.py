import numpy as np 
import math

from regex import R
import Gradient as grad


class Graph:

    def sigmoid(self,x):
        return np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
    
    def sigmPrime (self, x):
    # dérivé en x de la sigmoide 
        return x*(1-x)

    def marche (self, x) :
        if x<0.5:
            return 0
        else :
            return 1
        
    def __init__(self):
        self.poids = [] # liste de nbCouches array (tableaux)
        self.noeud = [] # liste de nbCouches array (listes)
        self.gradientNoeuds = [] # liste de nbCouches gradients (listes)
        self.gradientPoids = [] # liste de nbCouches gradients (listes)
        self.nbCouches = 0
        self.pas = 0
        self.history = []

               
        

    def parametrer(self):
    #TODO 
    #Pas besoin d'initialiser les noeuds ils vont être recalculer anyways. A part pour l'entrée.
    #Pertinance de construire un tableau des dimensions pour s'assurer qu'on part pas dans le décors ? 




        print("Combien de couches a le réseaux, entrées et sorties incluses")
        self.nbCouches = int(input())

        print("Quel pas souhaitez-vous avoir pour l'apprentissage ?")
        self.pas = float(input())

        print("Combien y a t'il d'entrées ?")
        nbNeuronesPreced = int(input ())
        self.noeud.append(np.zeros(nbNeuronesPreced)) #pour l'instant remplis de 0, serons remplis lors du passage en avant ? 

        for i in range(1,self.nbCouches):
            txt = "Combien de neurones possède la couche {}"
            print(txt.format(i))
            nbNeurones = int(input ())

            #A remplacer par tableau de dimensions ? 
            self.noeud.append(np.ones(nbNeurones)) #Initialisés à 1 seront calculé dans le premier passage en avant.  
           
            self.poids.append(np.random.random_sample((nbNeuronesPreced +1,nbNeurones))) #poids initialisés aléatoirement entre 0 et 1.
            
            self.gradientNoeuds.append(np.ones(nbNeurones)) #A-t-on vraiment besoin du gradient du poids ? 
            
            self.gradientPoids.append(np.ones((nbNeuronesPreced +1,nbNeurones)))
            
            nbNeuronesPreced = nbNeurones

        #Verifier que la dernière couche est ok 
       

    def parametrerPoidsFixes(self, lesPoids):
    #TODO

    #Initialiser les dimensions toutes seules
    #Changer la taille des matrices poids pour pas qu'ils prennent en compte la biais. ( puis on fait le .append)
    #initialiser les deux matrices gradients avec la bonne taille 
    #Pas besoin d'initialiser les noeuds ils vont être recalculer anyways. A part pour l'entrée.
    #Pertinance de construire une matrice des dimentions pour s'assurer qu'on part pas dans le décors ? 


        print("parametrer - Combien de couches a le réseaux, entrées et sorties incluses")
        self.nbCouches = int(input())

        print("Quel pas souhaitez-vous avoir pour l'apprentissage ?")
        self.pas = float(input())

        print("Combien y a t'il d'entrées ?")
        nbNeuronesPreced = int(input ())
        self.noeud.append(np.zeros(nbNeuronesPreced)) #pour l'instant remplis de 0, serons remplis lors du passage en avant ? 
        
        self.poids = lesPoids

        for i in range(1,self.nbCouches):
            txt = "Combien de neurones possède la couche {}"
            print(txt.format(i))
            nbNeurones = int(input ())
            
            self.noeud.append(np.ones(nbNeurones)) #Initialisés à 1 seront calculé dans le premier passage en avant.  
            
            self.gradientNoeuds.append(np.ones(nbNeurones)) #A-t-on vraiment besoin du gradient du biais ? 
            
            self.gradientPoids.append(np.ones((nbNeuronesPreced +1,nbNeurones)))
            
            nbNeuronesPreced = nbNeurones
    


    def passageForward(self, entree):
    #TODO
    # Attention les matrices poids vont changer de taille
    # Verifier ques le dimensions sont correctes avec le tableau de dimensions ?  
    
    # entrée est un array qui contient les valeurs de l'entrée. Il faut que la taille corresponde. Du coup a la fin du backward faut qu'on ai enlevé le biais de la couche d'entrée ? ou osef parcequ'on redonne la meme et qu'on la modifie pas 
    # retourne un vecteur des sortie (de 1 ou 0)
        if(len(entree) == len(self.noeud[0])):
            self.noeud[0] = np.append(entree,1)
        else: print ("la taille de l'entrée donnée ne correspond pas à celle attendu par le réseau")
        
        for i in range(1,self.nbCouches):
            self.noeud[i] = self.noeud[i-1].dot(self.poids[i-1])
            self.noeud[i] = self.sigmoid(self.noeud[i])
            if (i != self.nbCouches - 1):
                self.noeud[i]= np.append(self.noeud[i],1) #on fixe le biais à 1 
        
        return self.noeud[-1] 
   
   
    def passageBackward(self):
    # Permets de calculer les gradients associés a chaques poids. Ils sont rangés dans une matrice de la même 
    # taille que celle des poids ()

    #TODO
    # Calculer les gradients de noeuds
    # Attention a enlever le biais. pour faire le calcul pour la couche d'avant sans pour autant modifer le vecteur ? A moins que osef

    # Calculer les gradients des poids grâce au gradient des noeuds. 
    #ATTENTION j'ai pas tant de gradients que ça, pas sure des indices

               
        for i in range(self.nbCouches-3,-1,-1):
            
            self.gradientPoids[i+1] = np.transpose(self.gradientNoeuds[i+2]*self.sigmPrime(self.noeud[i+2])).dot(self.noeud[i+1])
            self.gradientNoeuds[i] = (self.gradientNoeuds[i+2]*self.sigmPrime(self.noeud[i+2])).dot(np.transpose(self.poids[i+1]))

            del self.noeud[i+1][-1] 

            print (self.gradientNoeuds[i])
            print (self.noeud[i+1])
            #assert self.gradientNoeuds[i].shape == self.noeud[i][:-1].shape

    
        self.gradientPoids[0] = np.transpose(self.gradientNoeuds[1]*self.sigmPrime(self.noeud[1])).dot(self.noeud[0])

    def apprentissage(self,entree,labels,trainRate=0.2):
    # mets a jour les poids en fonction des poids et des gradients associés pdt les deux passages
    # batch de 1
    #Forme des data : tableau de 3 colonnes = 2 positions entrée et valeur attendue 

        train_size = math.floor(trainRate*len(entree))
        x_train = entree[:train_size,:]
        y_train = labels[:train_size]
        #y_train = labels[:train_size,:]

        x_test = entree[train_size:,:]
        y_test = labels[train_size:]
        #y_test = labels[train_size:,:]

        for i in range(0,len(x_train)):
            y_predi = self.marche(self.passageForward(x_train[i,:]))
            self.passageBackward()

            #calcul de l'erreur
            E = y_train[i,:] - y_predi
            self.history = abs(E)

            #Mise à jour des poids
            for j in range(0,self.nbCouches-1):
                self.poids[j] = self.poids[j] - self.pas*E*self.gradientPoids[j]

        # L'apprentissage est fini on test sur les données de test pour connaitre l'efficacité
        taux_erreur = 0
        for i in range(0,len(x_test)):
            y_predi = self.passageForward(x_test(i))
            
            #calcul de l'erreur
            E = y_test(i) - y_predi
            taux_erreur = taux_erreur + abs(E)
        taux_erreur = taux_erreur/len(x_test)

        return taux_erreur

    def predire(self, x):

        sortie =  self.apprentissage(x)
        if len(sortie)==1 :
            prediction = self.marche(sortie)
            if  prediction == 1:
                reponse = " positif" 
            elif prediction == 0 :
                reponse = " négatif"
            else :
                reponse = " pbl : la réponse est différente de 0 ou 1"
        elif len(sortie)>1 :
            choix = np.where(sortie = max(sortie))
            prediction = np.eye(1,len(sortie),choix)
            txt = " la couche {}"
            reponse = txt.format(choix)
        else : 
            reponse = "pbl : la taille du vecteur prédiction est moins que 1"
        
        print( "La prédiction est :" + reponse)
        return prediction

            

        

            
            



