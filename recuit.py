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
    # print("solution :",I)
    # print("val :",val_I)
    
    #drawPlot(keys,values)
    #drawSolGraph(graph, terms, I)
    
    return keys, values, val_minI
    
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


def computeInstance(path, graph, terms, T, Tmin, deltaT):
    
    NB_POINTS = 15
    
    keys = []
    values = []
    interMin = []
    interMax = []
    
    # on construit le nom du fichier en fonction des paramètres
    strTmin = str(Tmin).split('.')[1]
    strDeltaT = str(deltaT).split('.')[1]
    path += "/"+str(T)+"_"+strTmin+"_"+strDeltaT
    print(path)
    # f = open(path,"a")
    
    #remplissage keys et values avec la 1ère éxécution
    k, v, r = recuit(graph, terms, T, Tmin, deltaT)
    
    
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
        
        #maj interMin/Max pour avoir une position relative a vals et non absolue
        interMax[cpt] = interMax[cpt] - vals
        interMin[cpt] = vals - interMin[cpt]
        
        
        old_xPt = xPt
        cpt+=1
    
    print("1 | val :",r)
    
    #on ajoute les valeurs des autres éxécutions dans values
    # for i in range(100):
    #     # k : keys from recuit
    #     # v : values from recuit
    #     # r : result from recuit
    #     k, v, r = recuit(graph, terms, T, Tmin, deltaT)
    #     print(i+2, "| val :",r)    
    #     # parcours des valeurs explorées par recuit
    #     for j in keys:
    #         values[j] += v[j] 
    #         #maj val_min de l'intervalle
    #         if(v[j] < interMin[j]):
    #             interMin[j] = v[j]
    #         #maj val_max de l'intervalle
    #         if(v[j] > interMax[j]):
    #             interMax[j] = v[j]
                    
    #calul des moyennes de valeurs
    # for i in range(len(values)):
    #     values[i] = values[i] / 100
    
    
    # f.write((str)(val)+"\n")
    # f.close()
    
    
    plt.plot(keys, values)
    plt.errorbar(keys, values, yerr=[interMin, interMax], fmt='none') #capsize=5
    
    plt.show()
    

def plotFromFile(path, Tinit, Tmin, deltaT):
    
    # récupération des données
    
    f = open(path,"r")
    cpt=0
    names = []
    values = []
    for l in f:
        names.append(cpt)
        val = (float)(l)
        values.append(val)
        cpt+=1
    f.close()

    ecart_type = st.stdev(values)
    moy = st.mean(values)
    
    #choix de l'intervalle de confiance
    intervale = ecart_type
    
    # courbe principale
    plt.plot(names,values)
    
    # Moyenne
    plt.plot([0,100], [moy,moy], '--', color='green')
    plt.text(101,moy,str(round(moy, 2)), color='green')
    
    # intervalle de confiance
    uBound = str(round(moy+intervale, 2))
    dBound = str(round(moy-intervale, 2))
    plt.plot([0,100],[moy+intervale,moy+intervale], '--', color='red')
    plt.plot([0,100],[moy-intervale,moy-intervale], '--', color='red')
    plt.text(101, moy+intervale, uBound, color='red')
    plt.text(101, moy-intervale, dBound, color='red')
    
    # plt.text(0,moy-intervale-4,"$-\sigma$")
    
    
    # légendes et titre
    plt.xlabel('number of runs')
    plt.ylabel('solution evaluation')

    # plt.legend(['courbe des résultats', 'moyenne', 'intervalle de confiance ($\sigma$)'], bbox_to_anchor=(1,1), loc='upper left')
    
    plt.title("Estimations avec $T_{init}="+ str(Tinit) +"$ $T_{min}="+ str(Tmin) +"$ et $\Delta_{T}="+ str(deltaT) +"$")

    # plt.text(0.92,0.61,"Points dans l'intervalle \nde confiance :"+str(inInter)+" %", transform=plt.gcf().transFigure)
    # plt.text(0.92,0.49,"$\sigma =$"+str(intervale), transform=plt.gcf().transFigure)
    # plt.text(0.92,0.44,"- "+str(moy-intervale), transform=plt.gcf().transFigure)
    # plt.text(0.92,0.40,"+ "+str(moy+intervale), transform=plt.gcf().transFigure)
    
    plt.margins(x=0.01)
    plt.show()
    
    # Prints
    
    inInter = 0
    for i in values:
        if(i>=moy-intervale and i<=moy+intervale):
            inInter+=1
    
    print("écart type :", ecart_type)
    print("moyenne:", moy)
    print("dans l'intervale :",inInter,"%")
    print()
    
    
if __name__ == "__main__":
    my_class = MySteinlibInstance()
    # print_graph(graph,terms)
    # sol=approx_steiner(graph,terms)
    # print_graph(graph,terms,sol)
    # print(eval_sol(graph,terms,sol))
    
    
    with open("test.stp") as my_file:
        my_parser = SteinlibParser(my_file, my_class)
        my_parser.parse()
        terms = my_class.terms
        graph = my_class.my_graph
        
        # print_graph(graph,terms)
        # sol=approx_steiner(graph,terms)
        # print(sol)
        # print_graph(graph,terms,sol)
        # print(eval_sol(graph,terms,sol))
        
        
        # print("sumW",(graph.size(weight="weight")))
        
        # recuit(graph, terms, 1000, 0.1, 0.999)
        # computeInstance("newNeigh/450000_1_999", graph, terms, 550000, 0.1, 0.999)
        # plotFromFile("newNeigh/450000_1_999", 550000, 0.1, 0.999)
        
        # computeInstance("newNeigh/455_11_999", graph, terms, 455, 0.11, 0.999)
        # plotFromFile("newNeigh/455_1_999", 455, 0.1, 0.999)
     
        # plotFromFile("newNeigh/100000_1_999", 100000, 0.1, 0.999)
        # plotFromFile("newNeigh/10000_1_997", 10000, 0.1, 0.997)
        # plotFromFile("newNeigh/10000_1_998", 10000, 0.1, 0.998)
        # plotFromFile("newNeigh/10000_1_999", 10000, 0.1, 0.999)
        # plotFromFile("newNeigh/1000_1_999", 1000, 0.1, 0.999)
        # plotFromFile("newNeigh/454_1_999", 454, 0.1, 0.999)
        # plotFromFile("newNeigh/455_1_999", 455, 0.1, 0.999)
        # plotFromFile("newNeigh/456_1_999", 456, 0.1, 0.999)
        
        # computeInstance("b02/10000_1_999", graph, terms, 10000, 0.1, 0.999)
        # plotFromFile("b02/10000_1_999", 10000, 0.1, 0.999)
        
        # computeInstance("new_b02", graph, terms, 4550, 0.01, 0.999)
        # plotFromFile("b02/4550_01_999", 4550, 0.01, 0.999)
     
        computeInstance("test", graph, terms, 1000, 0.1, 0.99)