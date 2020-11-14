import os
import sys
import math
import itertools as it
import matplotlib.pyplot as plt
import networkx as nx
import random as rdm
import statistics as st
from steinlib.instance import SteinlibInstance
from steinlib.parser import SteinlibParser


stein_file = "test.stp"

#librairies :
# python3 -m pip install steinlib
# python3 -m pip install networkx
# doc nx https://networkx.github.io/documentation/stable/index.html

# draw a graph in a window
def print_graph(graph,terms=None,sol=None):

    pos=nx.kamada_kawai_layout(graph)
    
    nx.draw(graph,pos)
    if (not (terms is None)):
        nx.draw_networkx_nodes(graph,pos, nodelist=terms, node_color='r')
    if (not (sol is None)):
        nx.draw_networkx_edges(graph,pos, edgelist=sol, edge_color='b')
    plt.show()
    
    return


def drawSolGraph(graph,terms, sol):
    #arretes du graphe
    edges = list(nx.edges(graph))
    #arretes du sous graphe
    edgesSub = []
    #on selectionne les arretes à 1 dans 'sol'
    for i in range(len(edges)):
        if(sol[i]):
            edgesSub.append(edges[i])
    
    #sous graphe à partir des arretes selectionees 
    subGraph = nx.edge_subgraph(graph,edgesSub)
    
    sg_terms = []
    for t in terms:
        if(subGraph.has_node(t)):
            sg_terms.append(t)
    
    print_graph(subGraph,sg_terms)
    
def init(graph):
    Es = []
    for i in range(len(nx.edges(graph))):
        Es.append(rdm.randint(0, 1))
    return Es

        
def neighbor(sol,graph):
    res = sol.copy()
    rdmI = rdm.randint(0, len(sol)-1)
    res[rdmI] = 1-res[rdmI]
    
    for i in range(rdmI):
        if rdm.random() < 1/len(res):
            res[i] = 1 - res[i]
    
    for i in range(rdmI+1, len(res)-1):
        if rdm.random() < 1/len(res):
            res[i] = 1 - res[i]
            
    return res
    
def eval_recuit(sol, graph, terms):
    #arretes du graphe
    edges = list(nx.edges(graph))
    #arretes du sous graphe
    edgesSub = []
    #on selectionne les arretes à 1 dans 'sol'
    for i in range(len(edges)): 
        if(sol[i]):
            edgesSub.append(edges[i])
    
    #sous graphe à partir des arretes selectionees 
    subGraph = nx.edge_subgraph(graph,edgesSub)
    
    #somme des poids de ce sous-graphe
    sumW = subGraph.size(weight="weight")
    #nombre de termes non-reliés
    nbTermsNR = 0
    
    #termes présents dans le sous-graphe
    sg_terms = []
    for t in terms:
        if(subGraph.has_node(t)):
            sg_terms.append(t)
        else:
            #print(t," is not in subGraph ! ")
            # si le noeud n'est pas dans le sous-graphe alors il n'est pas relié
            nbTermsNR+=1
            
    linked = False
    for t1 in sg_terms:
        for t2 in sg_terms:
            if t1!=t2 and nx.has_path(subGraph,t1,t2) :
                #print(t1,"is linked with",t2)
                linked = True
                break
        
        if not linked:
            nbTermsNR+=1
    #print_graph(subGraph, sg_terms)
    #print(nbTermsNR,"node not linked")
    
    Mt = graph.size()
    Mc = graph.size()/2
    
    nbCC = len(list(nx.connected_components(subGraph)))
    
    res = sumW + Mt*nbTermsNR + Mc*(nbCC-1)
    
    #print("sumW :",sumW,"NR :",nbTermsNR,"nbCC :",nbCC,"eval :",res)
    
    return res
    
def recuit(graph, terms, T, Tmin, deltaT):
    
    proba = 1
    keys = []
    values = []
    cpt=0
    
    # Solution initiale
    I = init(graph)
    val_I = eval_recuit(I, graph, terms)
    
    #Solution minimale
    minI = I
    val_minI = val_I
    
    while T > Tmin:
            
        nI = neighbor(I, graph)
        val_nI = eval_recuit(nI, graph, terms)
        
        if val_nI < val_I:
            proba = 1
        else:
            proba = math.exp((-(val_nI-val_I))/T)
        
        if  rdm.random() <= proba:
            #on met à jour I
            I = nI
            val_I = val_nI    
            #on met à jour la solution min
            if(val_I < val_minI):
                minI = I
                val_minI = val_I
        
        # mise à jour de la température
        T = deltaT * T
        
        keys.append(cpt)
        values.append(val_I)
        #print(T,val_I,sep=" | ")
        cpt+=1
    
    # on ajoute la valeur min rencontrée comme si c'était un itération en plus
    keys.append(cpt+1)
    values.append(val_minI)
    
    #drawPlot(keys,values)
    #drawSolGraph(graph, terms, I)
    
    return keys, values, minI
    
# class used to read a steinlib instance
class MySteinlibInstance(SteinlibInstance):

    def __init__(self):
        self.my_graph = nx.Graph()
        self.terms = []


    def terminals__t(self, line, converted_token):
        self.terms.append(converted_token[0])

    def graph__e(self, line, converted_token):
        e_start = converted_token[0]
        e_end = converted_token[1]
        weight = converted_token[2]
        self.my_graph.add_edge(e_start,e_end,weight=weight)

def drawPlot(names, values):
    plt.plot(names, values)
    plt.show()


def computeInstance(graph, terms, Tinit, Tmin, deltaT, path=None):
    
    NB_POINTS = 15
    
    keys = []
    values = []
    interMin = []
    interMax = []
    
    # on construit le nom du fichier en fonction des paramètres
    if (not (path is None)):
        
        strTmin = str(Tmin).split('.')[1]
        strDeltaT = str(deltaT).split('.')[1]
        path = "runs/"+path
        
        # si le dossier n'existe pas, on le créé
        if not os.path.exists(path):
            os.makedirs(path)
        
        path += "/"+str(Tinit)+"_"+strTmin+"_"+strDeltaT
        print(path)
        
        
        # test si le fichier n'existe pas déjà 
        if os.path.isfile(path):
            raise NameError("ComputeInstance : le fichier "+path+" existe déjà")
        # ouverture fichier
        f = open(path,"a")
    
    #remplissage keys et values avec la 1ère éxécution
    # k : keys from recuit
    # v : values from recuit
    # s : solution
    k, v, s = recuit(graph, terms, Tinit, Tmin, deltaT)
    
    
    #on récupère 10 points parmis ceux explorés
    for i in range(1,NB_POINTS):
        id = i*round(len(k)/NB_POINTS)
        keys.append(k[id])
        
    keys.append(k[len(k)-1])
    
    
    # on initialise values avec les valeurs de la 1ère exécution
    cpt = 0
    old_xPt = 0 
    #on parcoure les 10 points sur l'axe x qui ont été choisis (xPt)
    for xPt in keys:
        vals = 0
        #pour chaque xPt on parcoure les valeurs entre lui et l'ancien point (old_xPt)
        for j in range(old_xPt, xPt):
            
            if(j==old_xPt):
                # initialisation intervalle de confiance
                interMin.append(v[j])
                interMax.append(v[j])
            else:
                #maj intervalle de confiance
                if(v[j] < interMin[cpt]):
                    interMin[cpt] = v[j]
                if(v[j] > interMax[cpt]):
                    interMax[cpt] = v[j]
                
            vals+= v[j]
        
        # moyenne des valeurs entre old_xPt et xPt
        vals = vals / (xPt-old_xPt)
        values.append(vals)
        #maj old_xPt et cpt
        old_xPt = xPt
        cpt+=1
    
    print("1 | val :",v[len(v)-1])
    
    # on ajoute les valeurs des autres éxécutions dans values
    for i in range(99):
        # k : keys from recuit
        # v : values from recuit
        # s : solution
        k, v, s = recuit(graph, terms, Tinit, Tmin, deltaT)
        print(i+2, "| val :",v[len(v)-1])    
        # on initialise values avec les valeurs de la 1ère exécution
        cpt = 0
        old_xPt = 0 
        #on parcoure les 10 points sur l'axe x qui ont été choisis (xPt)
        for xPt in keys:
            vals = 0
            #pour chaque xPt on parcoure les valeurs entre lui et l'ancien point (old_xPt)
            for j in range(old_xPt, xPt):
            
                #maj intervalle de confiance
                if(v[j] < interMin[cpt]):
                    interMin[cpt] = v[j]
                if(v[j] > interMax[cpt]):
                    interMax[cpt] = v[j]
                
                vals+= v[j]
        
            # moyenne des valeurs entre old_xPt et xPt
            vals = vals / (xPt-old_xPt)
            # on ajoute la valeur moyennée pour cette courbe à celles des autre courbes
            values[cpt] += (vals)
            #maj old_xPt et cpt
            old_xPt = xPt
            cpt+=1
    
    
    for cpt in range(len(values)):
        # moyenne sur les 100 courbes
        values[cpt] = values[cpt] / 100
        #maj interMin/Max pour avoir une position relative a values et non absolue
        interMax[cpt] = interMax[cpt] - values[cpt]
        interMin[cpt] = values[cpt] - interMin[cpt]
        # on enregistre les données calculées ligne par ligne (key val interMin interMax)
        if (not (path is None)):
            f.write(str(keys[cpt])+" "+str(values[cpt])+" "+str(interMin[cpt])+" "+str(interMax[cpt])+"\n")
    
    #fermeture fichier de données
    if (not (path is None)):
        f.close()
    #dessin de la courbe avec les errorbars
    plt.plot(keys, values)
    plt.errorbar(keys, values, yerr=[interMin, interMax], fmt='none') #capsize=5
    # plt.show()
    

def plotFromFile(Tinit, Tmin, deltaT, path):
    
    strTmin = str(Tmin).split('.')[1]
    strDeltaT = str(deltaT).split('.')[1]
    path ="runs/"+path+"/"+str(Tinit)+"_"+strTmin+"_"+strDeltaT
    print(path)
    # test si le fichier existe 
    if not os.path.isfile(path):
        raise NameError("plotFromFile : le fichier "+path+" n'existe pas")
    
    # ouverture fichier
    f = open(path,"r")
    
    # récupération des données
    keys = []
    values = []
    interMin = []
    interMax = []
    for l in f:
        # on récupère les valeurs dans la ligne du fichier
        lTab = l.split(" ")
        key = int(lTab[0])
        val = float(lTab[1])
        iMin = float(lTab[2])
        iMax = float(lTab[3])
        # on remplit les tableaux
        keys.append(key)
        values.append(val)
        interMin.append(iMin)
        interMax.append(iMax)
    # fermeture fichier
    f.close()


    # courbe avec errorbars
    plt.plot(keys,values, color='blue')
    plt.errorbar(keys, values, yerr=[interMin, interMax], fmt='none', color='red') #capsize=5
    
    # légendes et titre
    plt.xlabel('number of iterations')
    plt.ylabel('average solution evaluation')
    plt.title("Estimations avec $T_{init}="+ str(Tinit) +"$ $T_{min}="+ str(Tmin) +"$ et $\Delta_{T}="+ str(deltaT) +"$")

    #affichage des valeurs de la dernière itération
    iEnd = len(keys)-1
    plt.text(
        0.92,0.60, 
        "dernière itération :", 
        transform=plt.gcf().transFigure
    )
    plt.text(
        0.92,0.56,
        "min : "+str(round(values[iEnd]-interMin[iEnd], 2)),
        transform=plt.gcf().transFigure
    )
    plt.text(
        0.92,0.52, 
        "val : "+str(round(values[iEnd], 2)), 
        transform=plt.gcf().transFigure
    )
    plt.text(
        0.92,0.48, 
        "max : "+str(round(values[iEnd]+interMax[iEnd], 2)), 
        transform=plt.gcf().transFigure
    )

    # dessin
    plt.show()
    
    
if __name__ == "__main__":
    my_class = MySteinlibInstance()
    # print_graph(graph,terms)
    # sol=approx_steiner(graph,terms)
    # print_graph(graph,terms,sol)
    # print(eval_sol(graph,terms,sol))
    
    
    with open("B/b01.stp") as my_file:
        my_parser = SteinlibParser(my_file, my_class)
        my_parser.parse()
        terms = my_class.terms
        graph = my_class.my_graph
        
        # print("sumW",(graph.size(weight="weight")))
        
        folder = "b01"
        
        computeInstance(graph, terms, 10000, 0.01, 0.999, folder)
        plotFromFile(10000, 0.01, 0.999, folder)
        
        computeInstance(graph, terms, 13000, 0.01, 0.999, folder)
        plotFromFile(13000, 0.01, 0.999, folder)
        
        computeInstance(graph, terms, 15000, 0.01, 0.999, folder)
        plotFromFile(15000, 0.01, 0.999, folder)
        