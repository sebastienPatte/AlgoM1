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


# verify if a solution is correct and evaluate it
def eval_sol(graph,terms,sol):

    if (len(sol)==0):
        print("Error: empty solution")
        return -1

    graph_sol = nx.Graph()
    for (i,j) in sol:
        graph_sol.add_edge(i,j,weight=graph[i][j]['weight'])

    # is sol a tree
    if (not (nx.is_tree(graph_sol))):
        print ("Error: the proposed solution is not a tree")
        return -1

    # are the terminals covered
    for i in terms:
        if not i in graph_sol:
            print ("Error: a terminal is missing from the solution")
            return -1

    # cost of solution
    cost = graph_sol.size(weight='weight')

    return cost


    

# compute an approximate solution to the steiner problem
def approx_steiner(graph,terms):
    res = []    
    Gt = nx.complete_graph(terms)
    
    #calcul de tout les plus court chemins
    path = dict(nx.all_pairs_shortest_path(graph))

    #pour chaque combinaison de 2 noeuds
    for (i,j) in it.combinations(terms,2):
        #on met le poids au plus court chemin dans Gt
        Gt.add_edge(i,j,weight=nx.path_weight(graph,path[i][j],weight="weight"))

    #arbre couvrant minimum de Gt        
    A = list(nx.minimum_spanning_tree(Gt).edges(data=True))
    
    for i in range(len(A)):
        pcc = path[A[i][0]][A[i][1]];
        for j in range(len(pcc)-1):
            
            pair = (pcc[j], pcc[j+1])    
            res.append(pair)
    
    print_graph(Gt,terms)
    
    return res 


################################################################################

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
        #Es.append(1)
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
    
    # variation de la température
    
    proba = 1
    
    keys = []
    values = []
    
    cpt=0
    
    I = init(graph)
    val_I = eval_recuit(I, graph, terms)
    
    #init sol minimale
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
    print("nbIt",cpt)
    return val_minI
    
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
    names = []
    values = []
    f = open(path,"a")
    
    for i in range(100):
        val = recuit(graph, terms, T, Tmin, deltaT)
        values.append(val)
        names.append(i)
        print(path, i, val)
        f.write((str)(val)+"\n")
    #drawPlot(names,values)
    f.close()

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
    
    
    with open("B/b05.stp") as my_file:
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
        
        computeInstance("b05/455_1_999", graph, terms, 455, 0.1, 0.999)
        plotFromFile("b05/455_1_999", 455, 0.1, 0.999)
     
        