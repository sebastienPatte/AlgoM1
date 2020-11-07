import os
import sys
import math
import itertools as it
import matplotlib.pyplot as plt
import networkx as nx
import random as rdm
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
    #i = rdm.randint(0, len(sol)-1)
    # res[i] = 1-res[i]
    
    for i in range(len(res)-1):
        if rdm.random() < 0.2:
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
    
    #print_graph(graph,terms,edgesSub)
    
    
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
    
    Mt = graph.size() + 100
    Mc = 100
    
    nbCC = len(list(nx.connected_components(subGraph)))
    
    res = sumW + Mt*nbTermsNR + Mc*(nbCC-1)
    
    #print("sumW :",sumW,"NR :",nbTermsNR,"nbCC :",nbCC,"eval :",res)
    
    return res
    
def recuit(graph, terms, Tmin):
    
    # température
    T = 100000
    # variation de la température
    deltaT = 0.99
    I = init(graph)
    
    proba = 1
    
    keys = []
    values = []
    
    cpt=0
    
    
    while T > Tmin:
        
        val_I = eval_recuit(I, graph, terms)
        nI = neighbor(I, graph)
        val_nI = eval_recuit(nI, graph, terms)
        
        if val_nI < val_I:
            I = nI
            val_I = val_nI    
        else:
            proba = math.exp((-(val_nI-val_I))/T)
            if  rdm.random() <= proba:
                #on met à jour I
                I = nI
                val_I = val_nI    
            
        # mise à jour de la température
        T = deltaT * T
        
        keys.append(cpt)
        values.append(val_I)
        #print(T,val_I,sep=" | ")
        cpt+=1
    # print("solution :",I)
    # print("val :",val_I)
    
    #drawPlot(keys,values)
    drawSolGraph(graph, terms, I)
    return val_I
    
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


if __name__ == "__main__":
    my_class = MySteinlibInstance()
    # print_graph(graph,terms)
    # sol=approx_steiner(graph,terms)
    # print_graph(graph,terms,sol)
    # print(eval_sol(graph,terms,sol))
        
    directory = "./B/"
    
    dirList = sorted(os.listdir(directory))
    
    
    names = []
    values = []
    
    # for filename in dirList:
        
    #     filepath = directory+filename
        
    #     with open(filepath) as my_file:
    #         my_parser = SteinlibParser(my_file, my_class)
    #         my_parser.parse()
    #         terms = my_class.terms
    #         graph = my_class.my_graph
        
        
    #     print(filename,"sumW",(graph.size(weight="weight")))
    #     val = recuit(graph,terms,0.01)
    #     print(val)
    #     names.append(filename)
    #     values.append(val)
    
    
    with open("test.stp") as my_file:
        my_parser = SteinlibParser(my_file, my_class)
        my_parser.parse()
        terms = my_class.terms
        graph = my_class.my_graph
        
        
        
        print("sumW",(graph.size(weight="weight")))
        for i in range(30):
            print(recuit(graph,terms,0.01))
        
        
        
    