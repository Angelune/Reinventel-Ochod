import numpy as np 
import Gradient as grad


class Graph:

    def sigmoid(self,x):
        return np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
    
    def sigmPrime (self, x):
    # dérivé en x de la sigmoide 
        return self.sigmoid(x)*(1-self.sigmoid(x))
        
    def __init__(self):
        self.poids = [] # liste de nbCouches array (tableaux)
        self.noeud = [] # liste de nbCouches array (listes)
        self.gradient = [] # liste de nbCouches gradients (listes)
        # self.activation = [] # liste de nbCouches fonctions (une seule fonction par couche) / pour l'instant que des fonctions sigmoides 
               
        

    def parametrer(self):
        print("Combien de couches a le réseaux, entrées et sorties incluses")
        nbCouches = int(input())

        print("Quel pas souhaitez-vous avoir pour l'apprentissage ?")
        self.pas = float(input())

        print("Combien y a t'il d'entrées ?")
        nbNeuronesPreced = int(input ())
        self.noeud.append(np.zeros(nbNeuronesPreced)) #pour l'instant remplis de 0, serons remplis lors du passage en avant ? 

        for i in range(1,nbCouches):
            txt = "Combien de neurones possède la couche {}"
            print(txt.format(i))
            nbNeurones = int(input ())

            
            self.noeud.append(np.ones(nbNeurones+1)) #Initialisés à 1 seront calculé dans le premier passage en avant. Le +1 est pour le biais.  
            self.poids.append(np.random.random_sample((nbNeurones+1,nbNeuronesPreced+1))) #poids initialisés aléatoirement entre 0 et 1. ATTENTION PAS DE BIAIS SUR LA DERNIERE COUCHE
            self.gradient.append(np.zeros(nbNeurones+1))
            
            #self.gradient.append(grad.Gradient(nbNeurones))
            
            nbNeuronesPreced = nbNeurones
    

    def parametrerPoidsFixes(self, lesPoids):
        print("parametrer - Combien de couches a le réseaux, entrées et sorties incluses")
        nbCouches = int(input())

        print("Quel pas souhaitez-vous avoir pour l'apprentissage ?")
        self.pas = float(input())

        print("Combien y a t'il d'entrées ?")
        nbNeuronesPreced = int(input ())
        self.noeud.append(np.zeros(nbNeuronesPreced)) #pour l'instant remplis de 0, serons remplis lors du passage en avant ? 
        
        self.poids = lesPoids

        for i in range(1,nbCouches-1):
            txt = "Combien de neurones possède la couche {}"
            print(txt.format(i))
            nbNeurones = int(input ())
            self.noeud.append(np.ones(nbNeurones+1)) #Initialisés à 1 seront calculé dans le premier passage en avant. Le +1 est pour le biais.                
            self.gradient.append(np.ones(nbNeurones+1))
            nbNeuronesPreced = nbNeurones
        
        txt = "Combien de neurones possède la dernière couche"  
        nbNeurones = int(input ())
        self.noeud.append(np.ones(nbNeurones)) #Initialisés à 1 seront calculé dans le premier passage en avant.                
        self.gradient.append(np.ones(nbNeurones))
    


    def passageForward(self, entree):
    # entrée est un array qui contient les valeurs de l'entrée. Il faut que la taille corresponde 
      
        #if (entree is np.array): print("c'est reconnu array")
        #else :  print ("c'est pas")
        
        if(len(entree) == len(self.noeud[0])):
            self.noeud[0] = np.append(entree,1)
        else: print ("la taille de l'entrée donnée ne correspond pas à celle attendu par le réseau")
        
        nbCouches = len(self.noeud)
        for i in range(1,nbCouches):
            self.noeud[i] = self.poids[i-1].dot(self.noeud[i-1])
            self.noeud[i] = self.sigmoid(self.noeud[i])
            self.noeud[i][len(self.noeud[i])-1] = 1 #on fixe le poid à 1 
        
    def passageBackward(self):
    #passer les paramètres nbCouche et tout en paramètres pour pas les redemander à chaque fois
       
        nbCouches = len(self.noeud)
        for i in range(nbCouches-1,0,-1):
            
            self.gradient[i] = self.gradient[i+1].dot(self.poids[i])
            #self.noeud[i][len(self.noeud[i])] = 1 #on fixe le poid à 1
