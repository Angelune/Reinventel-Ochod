import numpy as np 
import Gradient as grad


class Graph:

    def sigmoid(self,x):
        return np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
    
    def sigmPrime (self, x):
    # dérivé en x de la sigmoide 
        return x*(1-x)
        
    def __init__(self):
        self.poids = [] # liste de nbCouches array (tableaux)
        self.biais = []
        self.noeud = [] # liste de nbCouches array (listes)
        self.gradientNoeuds = [] # liste de nbCouches gradients (listes)
        self.gradientPoids = [] # liste de nbCouches gradients (listes)
        self.nbCouches = 0
        self.pas = 0

               
        

    def parametrer(self):
    #TODO
    #Changer la taille des matrices poids pour pas qu'ils prennent en compte la biais. ( puis on fait le .append)
    #initialiser les deux matrices gradients avec la bonne taille 
    #Pas besoin d'initialiser les noeuds ils vont être recalculer anyways. A part pour l'entrée.
    #Pertinance de construire une matrice des dimentions pour s'assurer qu'on part pas dans le décors ? 




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

            
            self.noeud.append(np.ones(nbNeurones)) #Initialisés à 1 seront calculé dans le premier passage en avant.  
            self.biais.append(np.random.random_sample(nbNeurones))
            self.poids.append(np.random.random_sample((nbNeurones,nbNeuronesPreced))) #poids initialisés aléatoirement entre 0 et 1.
            #TODO
            #self.gradient.append(np.ones(nbNeurones))
            #self.gradient.append(grad.Gradient(nbNeurones))
            
            nbNeuronesPreced = nbNeurones
    

    def parametrerPoidsFixes(self, lesPoids):
    #TODO
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

        for i in range(1,self.nbCouches-1):
            txt = "Combien de neurones possède la couche {}"
            print(txt.format(i))
            nbNeurones = int(input ())
            self.noeud.append(np.ones(nbNeurones+1)) #Initialisés à 1 seront calculé dans le premier passage en avant. Le +1 est pour le biais.                
            #TODO
            #self.gradient.append(np.ones(nbNeurones+1))
            #self.gradient.append(np.ones(nbNeurones+1))
            nbNeuronesPreced = nbNeurones
        
        print("Combien de neurones possède la dernière couche")
        nbNeurones = int(input ())
        self.noeud.append(np.ones(nbNeurones)) #Initialisés à 1 seront calculé dans le premier passage en avant.                
        #TODO
        #self.gradient.append(np.ones(nbNeurones))
        #self.gradient.append(np.ones(nbNeurones))
    


    def passageForward(self, entree):
    #TODO
    # Attention les matrices poids vont changer de taille 
    
    # entrée est un array qui contient les valeurs de l'entrée. Il faut que la taille corresponde 
        
        if(len(entree) == len(self.noeud[0])):
            self.noeud[0] = np.append(entree,1)
        else: print ("la taille de l'entrée donnée ne correspond pas à celle attendu par le réseau")
        
        for i in range(1,self.nbCouches):
            self.noeud[i] = self.poids[i-1].dot(self.noeud[i-1])
            self.noeud[i] = self.sigmoid(self.noeud[i])
            if (i != self.nbCouches - 1):
                self.noeud[i][len(self.noeud[i])-1] = 1 #on fixe le poid à 1 
        
   
   
    def passageBackward(self):
    # Permets de calculer les gradients associés a chaques poids. Ils sont rangés dans une matrice de la même 
    # taille que celle des poids ()


               
        for i in range(self.nbCouches-2,0,-1):
            
            self.gradient[i] = (self.gradient[i+1]*self.sigmPrime(self.noeud[i+1])).dot(self.poids[i])

            print (self.gradient[i])




        
            #assert self.gradient[i].shape == self.nbCouches

    def Apprentissage(self):
    #mets a jour les poids en fonction des poids et des gradients associés pdt les deux passages 

        for i in range(1,self.nbCouches):
            print ('')
            



